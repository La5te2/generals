"""Batched legal observations and learner-versus-frozen-opponent rollouts.

Each episode has one learner seat and one immutable opponent slot. Only learner
transitions enter PPO. Opponent commands/views are separately stored as labels.
"""

import time
from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from features import action, empty, encode, graph, observe
from generals.core.game import create_initial_state, step
from generals.core.grid import generate_grid
from rewards import material, potential, shape_reward
from rollout import Record, Rollout, bits, pack


@dataclass(frozen=True)
class Run:
    side: int = 24
    distance: int = 17
    environments: int = 512
    pool: int = 4096
    steps: int = 512
    horizon: int = 1024
    gamma: float = 1.0
    gae: float = 0.95
    shaping: float = 1.0


class Carry(NamedTuple):
    state: object
    memory: object
    inputs: object
    key: jax.Array
    opponent: jax.Array


def raw(state, player, topology):
    owners = jnp.where(state.ownership[0], 0, jnp.where(state.ownership[1], 1, -1))
    return observe(
        state.armies,
        owners,
        state.mountains,
        state.castles,
        state.generals,
        state.time,
        player,
        (False, False, True),
        jnp.ones(state.armies.shape, bool),
        topology,
    )


def views(states, topology):
    return jax.vmap(lambda s: jax.vmap(lambda p: raw(s, p, topology))(jnp.arange(2)))(
        states
    )


def encode_players(obs, memory, previous, topology):
    return jax.vmap(jax.vmap(encode, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, None))(
        obs, memory, previous, topology
    )


def choose(done, new, old):
    return jax.tree.map(
        lambda a, b: jnp.where(
            done.reshape(done.shape + (1,) * (a.ndim - done.ndim)), a, b
        ),
        new,
        old,
    )


def seat_view(inputs, seats):
    return jax.tree.map(lambda x: x[jnp.arange(len(seats)), seats], inputs)


def labels(states, seats, opponent_commands, enemy_inputs):
    def one(state, player):
        general = (state.generals & state.ownership[1 - player]).reshape(-1)
        return {
            "general": jnp.where(general.any(), jnp.argmax(general), -1),
            "owner": jnp.where(
                state.ownership[player], 1, jnp.where(state.ownership[1 - player], 2, 0)
            )
            .reshape(-1)
            .astype(jnp.uint8),
            "army": jnp.log1p(state.armies.reshape(-1)) / jnp.log(256.0),
        }

    result = jax.vmap(one)(states, seats)
    result.update(
        enemy_action=opponent_commands,
        enemy_flags=bits(enemy_inputs.cells[..., :9]).astype(jnp.uint16),
        enemy_army=enemy_inputs.cells[..., 9],
    )
    return result


@eqx.filter_jit
def initial_maps(config, keys):
    castles = (1, max(2, config.side * config.side // 50))
    maps = jax.vmap(
        lambda key: generate_grid(
            key,
            grid_dims=(config.side, config.side),
            pad_to=config.side,
            num_castles_range=castles,
            min_generals_distance=config.distance,
        )
    )(keys)
    return jax.vmap(create_initial_state)(maps)


class Environment:
    def __init__(self, config, history, seed, inference_batch=64, rollout_chunk=16):
        self.config, self.history = config, history
        self.inference_batch, self.rollout_chunk = inference_batch, rollout_chunk
        self.topology = graph(config.side, config.side)
        self.seats = jnp.arange(config.environments) % 2
        self.rows = jnp.arange(config.environments)
        keys = jax.random.split(jax.random.PRNGKey(seed), config.pool)
        pieces = [
            initial_maps(config, keys[i : i + 128]) for i in range(0, config.pool, 128)
        ]
        self.pool = jax.tree.map(lambda *x: jnp.concatenate(x), *pieces)
        states = jax.tree.map(lambda x: x[self.rows % config.pool], self.pool)
        previous = jnp.full((config.environments, 2), config.side**2 * 8)
        inputs, memory = encode_players(
            views(states, self.topology), self.blank(), previous, self.topology
        )
        self.carry = Carry(
            states,
            memory,
            inputs,
            jax.random.PRNGKey(seed + 1),
            jnp.zeros(config.environments, jnp.int32),
        )
        self.buffer = Rollout(
            config.steps, config.environments, config.side**2, history
        )
        self.advance = eqx.filter_jit(self._advance)
        self.predict = eqx.filter_jit(self._predict)

    def blank(self):
        return jax.tree.map(
            lambda x: jnp.broadcast_to(x, (self.config.environments, 2, *x.shape)),
            empty(self.config.side**2, self.history),
        )

    def _predict(self, model, inputs):
        def one(x):
            output = model(x, self.topology)
            return output.logits, output.value

        return jax.lax.map(one, inputs, batch_size=self.inference_batch)

    def _enemy_actions(self, bank, inputs, ids):
        """Compact by policy slot so weights are shared across each inference batch."""
        arrays, static = eqx.partition(bank, eqx.is_array)
        capacity = jax.tree.leaves(arrays)[0].shape[0]
        count, size = len(ids), min(len(ids), self.inference_batch)
        padded = ((count + size - 1) // size) * size

        def slot_step(slot, result):
            selected = ids == slot
            indices = jnp.nonzero(selected, size=padded, fill_value=0)[0]
            length = selected.sum()
            opponent = eqx.combine(jax.tree.map(lambda x: x[slot], arrays), static)

            def batch_step(batch, result):
                rows = jax.lax.dynamic_slice(indices, (batch * size,), (size,))
                valid = batch * size + jnp.arange(size) < length
                x = jax.tree.map(lambda a: a[rows], inputs)
                logits, _ = self._predict(opponent, x)
                commands = jnp.argmax(logits, -1).astype(jnp.int32)
                return result.at[rows].add(jnp.where(valid, commands, 0))

            return jax.lax.fori_loop(0, (length + size - 1) // size, batch_step, result)

        return jax.lax.fori_loop(0, capacity, slot_step, jnp.zeros(count, jnp.int32))

    def _advance(self, model, bank, eligible, carry, count):
        cfg, topology = self.config, self.topology

        def tick(carry, _):
            state, memory, inputs, key, opponent = carry
            key, sample_key, reset_key, rival_key = jax.random.split(key, 4)
            ours, theirs = (
                seat_view(inputs, self.seats),
                seat_view(inputs, 1 - self.seats),
            )
            logits, value = self._predict(model, ours)
            logp = jax.nn.log_softmax(logits)
            commands = jax.random.categorical(sample_key, logp).astype(jnp.int32)
            enemy = self._enemy_actions(bank, theirs, opponent)
            actions = jnp.zeros((cfg.environments, 2), jnp.int32)
            actions = (
                actions.at[self.rows, self.seats]
                .set(commands)
                .at[self.rows, 1 - self.seats]
                .set(enemy)
            )
            moves = jax.vmap(jax.vmap(lambda a: action(a, cfg.side, cfg.side**2)))(
                actions
            )
            following, info = jax.vmap(lambda s, a: step(s, a, general_trade=True))(
                state, moves
            )
            terminal = info.is_done
            truncated = (following.time >= cfg.horizon) & ~terminal
            ended = terminal | truncated
            outcome = jnp.where(
                terminal,
                jnp.where(
                    info.winner == self.seats,
                    1.0,
                    jnp.where(info.winner == 1 - self.seats, -1.0, 0.0),
                ),
                0.0,
            )
            phi = jax.vmap(potential)(state)[self.rows, self.seats]
            next_phi = jax.vmap(potential)(following)[self.rows, self.seats]
            reward = shape_reward(
                outcome, phi, next_phi, terminal, cfg.gamma, cfg.shaping
            )
            next_inputs, next_memory = encode_players(
                views(following, topology), memory, actions, topology
            )
            bootstrap = jax.lax.cond(
                truncated.any(),
                lambda: self._predict(model, seat_view(next_inputs, self.seats))[1],
                lambda: jnp.zeros_like(value),
            )

            def reset():
                indices = jax.random.randint(
                    reset_key, (cfg.environments,), 0, cfg.pool
                )
                fresh = jax.tree.map(lambda x: x[indices], self.pool)
                new_inputs, new_memory = encode_players(
                    views(fresh, topology),
                    self.blank(),
                    jnp.full_like(actions, cfg.side**2 * 8),
                    topology,
                )
                return (
                    choose(ended, fresh, following),
                    choose(ended, new_memory, next_memory),
                    choose(ended, new_inputs, next_inputs),
                )

            next_state, next_memory, next_inputs = jax.lax.cond(
                ended.any(), reset, lambda: (following, next_memory, next_inputs)
            )
            sampled_opponent = jax.random.categorical(
                rival_key, jnp.where(eligible, 0.0, -jnp.inf), shape=(cfg.environments,)
            ).astype(jnp.int32)
            next_carry = Carry(
                next_state,
                next_memory,
                next_inputs,
                key,
                jnp.where(ended, sampled_opponent, opponent),
            )
            flags, values = pack(ours)
            record = Record(
                flags,
                values,
                ours.stats[..., -1, :],
                ours.time_valid.sum(-1),
                commands,
                jnp.take_along_axis(logp, commands[:, None], -1)[:, 0],
                value,
                bootstrap,
                reward,
                terminal,
                truncated,
                ours.legal[..., :-1].any(-1),
                labels(state, self.seats, enemy, theirs),
                outcome,
                jax.vmap(material)(state)[self.rows, self.seats],
            )
            return next_carry, record

        return jax.lax.scan(tick, carry, None, length=count)

    def collect(self, model, bank, eligible, progress=None):
        buffer = self.buffer
        buffer.begin(seat_view(self.carry.inputs, self.seats))
        started = time.monotonic()
        wins = losses = completed = truncated = 0
        material_sum = np.zeros(3)
        for offset in range(0, self.config.steps, self.rollout_chunk):
            count = min(self.rollout_chunk, self.config.steps - offset)
            self.carry, records = self.advance(model, bank, eligible, self.carry, count)
            records = jax.device_get(records)
            buffer.write(offset, records)
            wins += np.count_nonzero(records.outcome > 0)
            losses += np.count_nonzero(records.outcome < 0)
            completed += np.count_nonzero(records.terminal)
            truncated += np.count_nonzero(records.truncated)
            material_sum += records.material.sum(axis=(0, 1))
            if progress and (
                (offset + count) % 64 == 0 or offset + count == self.config.steps
            ):
                progress(
                    {
                        "phase": "rollout",
                        "steps": offset + count,
                        "total": self.config.steps,
                        "seconds": time.monotonic() - started,
                    }
                )
        following = np.asarray(
            self.predict(model, seat_view(self.carry.inputs, self.seats))[1]
        )
        accumulated = np.zeros(buffer.lanes, np.float32)
        for t in range(self.config.steps - 1, -1, -1):
            next_value = np.where(buffer.truncated[t], buffer.bootstrap[t], following)
            delta = (
                buffer.reward[t]
                + self.config.gamma * ~buffer.terminal[t] * next_value
                - buffer.value[t]
            )
            accumulated = (
                delta
                + self.config.gamma
                * self.config.gae
                * ~(buffer.terminal[t] | buffer.truncated[t])
                * accumulated
            )
            buffer.advantage[t] = accumulated
            following = buffer.value[t]
        buffer.returns[:] = buffer.advantage + buffer.value
        return buffer, {
            "wins": int(wins),
            "losses": int(losses),
            "completed": int(completed),
            "truncated": int(truncated),
            "advantage_std": float(buffer.advantage.std()),
            "mean_shaped_reward": float(buffer.reward.mean()),
            "mean_material": (material_sum / buffer.count).tolist(),
        }
