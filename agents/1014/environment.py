"""Batched legal observations and learner-versus-frozen-opponent rollouts.

Each episode has one learner seat and one immutable opponent slot. Only learner
transitions enter PPO. Hidden state and opponent commands supply training labels.
"""

import time
from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from critic import Oracle, expectation, trace_targets
from features import action, empty, encode, graph, observe
from generals.core.game import create_initial_state, step
from generals.core.grid import generate_grid
from rewards import material, potential, reward_parts, shape_reward
from rollout import Record, Rollout, bits, pack
from economy import HORIZONS, transition as economy_transition


@dataclass(frozen=True)
class Run:
    side: int = 24
    distance: int = 17
    environments: int = 512
    pool: int = 4096
    steps: int = 512
    horizon: int = 1024
    gamma: float = 1.0
    trace_lambda: float = 0.95
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


def labels(states, seats, opponent_commands):
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
    )
    return result


def oracle(states, seats, commands):
    """Full simulator facts exclusively for the centralized training critic."""

    def one(state, player, command):
        own, enemy = state.ownership[player], state.ownership[1 - player]
        facts = jnp.stack(
            (
                own,
                enemy,
                ~(own | enemy),
                state.mountains,
                state.castles,
                state.generals,
            ),
            -1,
        ).reshape(-1, 6)
        return Oracle(
            bits(facts).astype(jnp.uint8),
            jnp.log1p(state.armies.reshape(-1)) / jnp.log(256.0),
            command,
        )

    return jax.vmap(one)(states, seats, commands)


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
        self.bootstrap = eqx.filter_jit(self._bootstrap)

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

    def _q_values(self, critic, inputs, truth):
        return jax.lax.map(
            lambda pair: critic(*pair, self.topology),
            (inputs, truth),
            batch_size=self.inference_batch,
        )

    def _bootstrap(self, model, critic, bank, state, inputs, opponent, seats=None):
        seats = self.seats if seats is None else seats
        ours, theirs = seat_view(inputs, seats), seat_view(inputs, 1 - seats)
        commands = self._enemy_actions(bank, theirs, opponent)
        logits, _ = self._predict(model, ours)
        return expectation(
            self._q_values(critic, ours, oracle(state, seats, commands)), logits
        )

    def _truncated_values(
        self, model, critic, bank, state, inputs, opponent, truncated
    ):
        """Evaluate compact fixed-size batches containing only timeout lanes and padding."""
        count = len(truncated)
        size = min(count, self.inference_batch)
        padded = ((count + size - 1) // size) * size
        indices = jnp.nonzero(truncated, size=padded, fill_value=0)[0]
        length = truncated.sum()

        def batch_step(batch, result):
            rows = jax.lax.dynamic_slice(indices, (batch * size,), (size,))
            valid = batch * size + jnp.arange(size) < length
            values = self._bootstrap(
                model,
                critic,
                bank,
                jax.tree.map(lambda x: x[rows], state),
                jax.tree.map(lambda x: x[rows], inputs),
                opponent[rows],
                self.seats[rows],
            )
            return result.at[rows].add(jnp.where(valid, values, 0))

        return jax.lax.fori_loop(
            0, (length + size - 1) // size, batch_step, jnp.zeros(count, jnp.float32)
        )

    def _advance(self, model, critic, bank, eligible, carry, count):
        cfg, topology = self.config, self.topology

        def tick(carry, _):
            state, memory, inputs, key, opponent = carry
            key, sample_key, reset_key, rival_key = jax.random.split(key, 4)
            ours, theirs = (
                seat_view(inputs, self.seats),
                seat_view(inputs, 1 - self.seats),
            )
            logits, _ = self._predict(model, ours)
            logp = jax.nn.log_softmax(logits)
            commands = jax.random.categorical(sample_key, logp).astype(jnp.int32)
            enemy = self._enemy_actions(bank, theirs, opponent)
            truth = oracle(state, self.seats, enemy)
            q = self._q_values(critic, ours, truth)
            value = expectation(q, logits)
            chosen_q = jnp.take_along_axis(q, commands[:, None], -1)[:, 0]
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
                lambda: self._truncated_values(
                    model, critic, bank, following, next_inputs, opponent, truncated
                ),
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
                labels(state, self.seats, enemy),
                outcome,
                jax.vmap(material)(state)[self.rows, self.seats],
                reward_parts(outcome, phi, next_phi, terminal, cfg.gamma, cfg.shaping),
                jnp.where(truncated, next_phi, 0.0),
                ours.public_valid.sum(-1),
                chosen_q,
                truth.flags,
                jax.vmap(economy_transition)(state, following, terminal, self.seats),
            )
            return next_carry, record

        return jax.lax.scan(tick, carry, None, length=count)

    def collect(self, model, critic, bank, eligible, progress=None):
        buffer = self.buffer
        buffer.begin(seat_view(self.carry.inputs, self.seats))
        started = time.monotonic()
        wins = losses = completed = truncated = 0
        material_sum = np.zeros(3)
        reward_sums = np.zeros(3, np.float64)
        truncated_potential_sum = 0.0
        for offset in range(0, self.config.steps, self.rollout_chunk):
            count = min(self.rollout_chunk, self.config.steps - offset)
            self.carry, records = self.advance(
                model, critic, bank, eligible, self.carry, count
            )
            records = jax.device_get(records)
            buffer.write(offset, records)
            wins += np.count_nonzero(records.outcome > 0)
            losses += np.count_nonzero(records.outcome < 0)
            completed += np.count_nonzero(records.terminal)
            truncated += np.count_nonzero(records.truncated)
            material_sum += records.material.sum(axis=(0, 1))
            reward_sums += records.reward_parts.sum(axis=(0, 1), dtype=np.float64)
            truncated_potential_sum += records.truncated_potential.sum(dtype=np.float64)
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
            self.bootstrap(
                model,
                critic,
                bank,
                self.carry.state,
                self.carry.inputs,
                self.carry.opponent,
            )
        )
        buffer.advantage[:], buffer.returns[:] = trace_targets(
            buffer.reward,
            buffer.value,
            buffer.chosen_q,
            buffer.terminal,
            buffer.truncated,
            buffer.bootstrap,
            following,
            self.config.gamma,
            self.config.trace_lambda,
        )
        buffer.finish_economy()
        return buffer, {
            "economy_valid_fraction": dict(
                zip(
                    map(str, HORIZONS),
                    buffer.targets["economy_valid"].mean(axis=(0, 1)).tolist(),
                )
            ),
            "wins": int(wins),
            "losses": int(losses),
            "completed": int(completed),
            "truncated": int(truncated),
            "advantage_std": float(buffer.advantage.std()),
            "q_mean": float(buffer.chosen_q.mean()),
            "q_std": float(buffer.chosen_q.std()),
            "q_target_mean": float(buffer.returns.mean()),
            "q_target_std": float(buffer.returns.std()),
            "q_explained_variance": float(
                1
                - np.var(buffer.returns - buffer.chosen_q)
                / max(float(np.var(buffer.returns)), 1e-8)
            ),
            "mean_shaped_reward": float(buffer.reward.mean()),
            "reward_components": {
                "sum": dict(
                    zip(
                        ("outcome", "continuation", "terminal_reset"),
                        reward_sums.tolist(),
                    )
                ),
                "mean_per_transition": dict(
                    zip(
                        ("outcome", "continuation", "terminal_reset"),
                        (reward_sums / buffer.count).tolist(),
                    )
                ),
                "shaping_sum": float(reward_sums[1:].sum()),
                "total_sum": float(buffer.reward.sum(dtype=np.float64)),
                "reconciliation_error": float(
                    buffer.reward.sum(dtype=np.float64) - reward_sums.sum()
                ),
                "terminal_reset_mean_per_termination": float(reward_sums[2] / completed)
                if completed
                else None,
                "truncated_potential_mean": float(truncated_potential_sum / truncated)
                if truncated
                else None,
            },
            "mean_material": (material_sum / buffer.count).tolist(),
        }
