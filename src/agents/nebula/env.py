"""Generals-bots simulates batched games in JAX and shares observations with PyTorch through DLPack."""

import os
from functools import partial

# PyTorch and JAX share the accelerator, so JAX allocates memory as the environment needs it.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import torch
from features import empty, update
from generals import GeneralsEnv, get_observation
from generals.core.game import create_initial_state
from generals.core.grid import generate_grid


@partial(jax.jit, static_argnames=("distance", "rows", "cols", "side"))
def generate(keys, distance, rows, cols, side):
    # use the package's map generator, retaining dimensions alongside its mountain-padded states.
    def game(key):
        grid = generate_grid(key, grid_dims=(rows, cols), pad_to=side,
                             min_generals_distance=distance[0], max_generals_distance=distance[1])
        return create_initial_state(grid)
    return jax.vmap(game)(keys)


def observations(states):
    views = jax.vmap(lambda state: jax.vmap(lambda player: get_observation(state, player))(jnp.arange(2)))(states)
    # game-major order places red immediately before blue for every game.
    return jax.tree.map(lambda value: value.reshape((-1, *value.shape[2:])), views)


def tensors(memory):
    # these tensors share JAX storage. callers may read them or copy them into a rollout buffer.
    return tuple(torch.from_dlpack(value) for value in memory[:3])


class Arena:
    def __init__(self, count, config, seed, device):
        self.count, self.side = count, config.side
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("The training environment supports cpu and cuda devices")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        platform = "gpu" if self.device.type == "cuda" else "cpu"
        devices = jax.devices(platform)
        index = self.device.index or 0
        if index >= len(devices):
            raise ValueError(f"JAX device {platform}:{index} is unavailable")
        self.backend = devices[index]
        self.minimum, self.maximum = config.minimum, config.maximum
        combinations = (self.maximum - self.minimum + 1) ** 2
        self.perSize = (config.pool + combinations - 1) // combinations
        self.poolSize = self.perSize * combinations
        with jax.default_device(self.backend):
            self.key = jax.random.PRNGKey(seed)
            # neutral cities, fog and general trades implement mainstream 1v1.
            self.env = GeneralsEnv(grid_dims=(config.side, config.side), truncation=config.horizon,
                                   pool_size=self.poolSize, general_trade=True,
                                   build_castles=False, deathtouch_turn=None, perfect_info=False)
        self.advance = jax.jit(self.transition)
        self.initialize = jax.jit(lambda states, rows, cols: update(
            empty(2 * self.count, self.side), observations(states), jnp.repeat(rows, 2), jnp.repeat(cols, 2)))
        self.states = None
        self.memory = None
        self.last = None
        self.pool = None

    def reset(self, distance):
        """Refresh the map pool. Existing games keep their boards until they finish."""
        with jax.default_device(self.backend):
            self.key, poolKey = jax.random.split(self.key)
            keys = jax.random.split(poolKey, self.poolSize)
            pools, dimensions = [], []
            offset = 0
            for rows in range(self.minimum, self.maximum + 1):
                for cols in range(self.minimum, self.maximum + 1):
                    pools.append(generate(keys[offset:offset + self.perSize], tuple(distance), rows, cols, self.side))
                    dimensions.extend([(rows, cols)] * self.perSize)
                    offset += self.perSize
            pool = jax.tree.map(lambda *parts: jnp.concatenate(parts), *pools)
            self.key, shuffleKey = jax.random.split(self.key)
            order = jax.random.permutation(shuffleKey, self.poolSize)
            self.pool = jax.tree.map(lambda value: value[order], pool)
            self.dimensions = jnp.asarray(dimensions, jnp.int32)[order]
            if self.states is None:
                self.key, initialKey = jax.random.split(self.key)
                indices = jax.random.randint(initialKey, (self.count,), 0, self.poolSize)
                self.states = jax.tree.map(lambda value: value[indices], self.pool)
                self.states = self.states._replace(pool_idx=(indices + 1) % self.poolSize)
                self.rows, self.cols = self.dimensions[indices].T
                self.memory = self.initialize(self.states, self.rows, self.cols)
            jax.block_until_ready(self.pool)

    def transition(self, states, memory, rows, cols, actions, pool, dimensions):
        area = self.side ** 2
        kind, position = actions // area, actions % area
        moves = jnp.stack((kind == 8, position // self.side, position % self.side,
                           kind % 4, kind >= 4), axis=-1).astype(jnp.int32).reshape(self.count, 2, 5)
        outcome, following = jax.vmap(lambda state, action: self.env.step(state, action, pool))(states, moves)
        ended = outcome.terminated | outcome.truncated
        final = update(memory, observations(outcome.last_state), jnp.repeat(rows, 2), jnp.repeat(cols, 2))
        # auto-reset observations belong to a new game. preserve the old game's final memory for bootstrapping.
        nextRows, nextCols = dimensions[states.pool_idx % self.poolSize].T
        rows, cols = jnp.where(ended, nextRows, rows), jnp.where(ended, nextCols, cols)

        def resetMemory(current):
            fresh = self.initialize(following, rows, cols)
            players = jnp.repeat(ended, 2)
            return jax.tree.map(lambda new, old: jnp.where(
                players.reshape((-1,) + (1,) * (old.ndim - 1)), new, old), fresh, current)

        memory = jax.lax.cond(jnp.any(ended), resetMemory, lambda current: current, final)
        return following, memory, rows, cols, final, outcome.reward.reshape(-1), jnp.repeat(outcome.terminated, 2), jnp.repeat(outcome.truncated, 2)

    def read(self):
        if self.memory is None:
            raise RuntimeError("Reset the arena before reading observations")
        return tensors(self.memory)

    def step(self, actions):
        if self.states is None:
            raise RuntimeError("Reset the arena before stepping")
        if actions.shape != (2 * self.count,) or actions.device != self.device:
            raise ValueError("Expected one action per player on the environment device")
        choices = jax.dlpack.from_dlpack(actions.detach().to(dtype=torch.int32).contiguous())
        result = self.advance(self.states, self.memory, self.rows, self.cols, choices, self.pool, self.dimensions)
        self.states, self.memory, self.rows, self.cols, self.last = result[:5]
        return tuple(torch.from_dlpack(value) for value in result[5:])

    def final(self):
        if self.last is None:
            raise RuntimeError("Step the arena before reading its final observations")
        return tensors(self.last)

    def close(self):
        if self.states is not None:
            jax.block_until_ready(self.states)
        self.states = self.memory = self.last = self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
