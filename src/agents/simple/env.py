"""Run simultaneous batched 1v1 transitions; only masked observations enter the network."""

import os
from functools import partial

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from features import View
from generals import GeneralsEnv, get_observation
from generals.core.game import create_initial_state
from generals.core.grid import generate_grid


@partial(jax.jit, static_argnames=("rows", "cols", "side", "distance"))
def generate(keys, rows, cols, side, distance):
    def single(key):
        grid = generate_grid(key, grid_dims=(rows, cols), pad_to=side,
                             min_generals_distance=distance[0], max_generals_distance=distance[1])
        return create_initial_state(grid)
    return jax.vmap(single)(keys)


def observations(states, rows, cols, side):
    view = jax.vmap(lambda state: jax.vmap(lambda player: get_observation(state, player))(jnp.arange(2)))(states)
    return pack(view, rows, cols, side)


def pack(view, rows, cols, side):
    """Convert the environment's already masked observations into the network's raw fields."""
    view = jax.tree.map(lambda value: value.reshape((-1, *value.shape[2:])), view)
    inside = ((jnp.arange(side)[None, :, None] < jnp.repeat(rows, 2)[:, None, None]) &
              (jnp.arange(side)[None, None, :] < jnp.repeat(cols, 2)[:, None, None]))
    terrain = jnp.ones(view.armies.shape, jnp.int32)
    for mask, kind in ((view.mountains, 2), (view.castles, 3), (view.generals, 4),
                       (view.fog_cells, 0), (view.structures_in_fog, 5)):
        terrain = jnp.where(mask, kind, terrain)
    owner = jnp.where(view.owned_cells, 1, jnp.where(view.opponent_cells, 2, 0)).astype(jnp.int32)
    visible = inside & ~view.fog_cells & ~view.structures_in_fog
    stats = jnp.stack((view.timestep, view.owned_land_count, view.owned_army_count,
                       view.opponent_land_count, view.opponent_army_count), -1).astype(jnp.int32)
    return (jnp.where(inside, terrain, 2), jnp.where(visible, owner, 0),
            jnp.where(visible, view.armies, 0).astype(jnp.float32), stats, inside)


def select_device(name):
    """Resolve a requested CPU or GPU without silently falling back to another backend."""
    if isinstance(name, jax.Device):
        return name
    kind, _, index = name.partition(":")
    if kind not in ("cpu", "cuda", "gpu"):
        raise ValueError("Select cpu or cuda:N")
    try:
        devices = jax.devices("cpu" if kind == "cpu" else "gpu")
        return devices[int(index or 0)]
    except (RuntimeError, IndexError) as error:
        raise ValueError("Requested JAX device is unavailable") from error


class Arena:
    def __init__(self, config, device, seed=None):
        self.config = config
        self.count, self.side = config.environments, config.side
        self.backend = select_device(device)
        self.pool_size = config.pool
        with jax.default_device(self.backend):
            self.key = jax.random.PRNGKey(config.seed if seed is None else seed)
            self.engine = GeneralsEnv(grid_dims=(config.side, config.side), pool_size=self.pool_size,
                                      truncation=config.horizon, general_trade=True, build_castles=False,
                                      deathtouch_turn=None, perfect_info=False)
        self.advance = jax.jit(self.transition)
        self.read_arrays = jax.jit(lambda state, rows, cols: observations(state, rows, cols, self.side))
        self.states = self.pool = self.last = self.current = None

    def prepare(self, maps=None):
        """Refill the map pool; existing games retain their boards, dimensions and clocks."""
        maps = self.config if maps is None else maps
        maps.validate()
        if (maps.side, maps.pool, maps.environments, maps.horizon) != (
                self.side, self.pool_size, self.count, self.config.horizon):
            raise ValueError("Map refresh cannot change batch shape, pool size or episode limit")
        combinations = (maps.maximum - maps.minimum + 1)**2
        per_size, extra = divmod(self.pool_size, combinations)
        with jax.default_device(self.backend):
            self.key, pool_key = jax.random.split(self.key)
            keys = jax.random.split(pool_key, self.pool_size)
            parts, dimensions, offset = [], [], 0
            for rows in range(maps.minimum, maps.maximum + 1):
                for cols in range(maps.minimum, maps.maximum + 1):
                    size = per_size + (len(parts) < extra)
                    parts.append(generate(keys[offset:offset + size], rows, cols, self.side,
                                          tuple(maps.distance)))
                    dimensions.extend([(rows, cols)] * size)
                    offset += size
            self.key, order_key = jax.random.split(self.key)
            order = jax.random.permutation(order_key, self.pool_size)
            self.pool = jax.tree.map(lambda *values: jnp.concatenate(values)[order], *parts)
            self.dimensions = jnp.asarray(dimensions, jnp.int32)[order]
            if self.states is None:
                self.key, initial_key = jax.random.split(self.key)
                indices = jax.random.randint(initial_key, (self.count,), 0, self.pool_size)
                self.states = jax.tree.map(lambda value: value[indices], self.pool)
                self.states = self.states._replace(pool_idx=(indices + 1) % self.pool_size)
                self.rows, self.cols = self.dimensions[indices].T
            jax.block_until_ready(self.pool)

    def transition(self, states, rows, cols, actions, pool, dimensions):
        area = self.side**2
        kind, cell = actions // area, actions % area
        moves = jnp.stack((actions == 8 * area, cell // self.side, cell % self.side,
                           kind % 4, kind >= 4), -1).astype(jnp.int32).reshape(self.count, 2, 5)
        outcome, following = jax.vmap(lambda state, action: self.engine.step(state, action, pool))(states, moves)
        ended = outcome.terminated | outcome.truncated
        next_rows, next_cols = dimensions[states.pool_idx % self.pool_size].T
        rows, cols = jnp.where(ended, next_rows, rows), jnp.where(ended, next_cols, cols)
        # step already computed observations after auto-reset; reuse them instead of observing twice.
        current = pack(outcome.observation, rows, cols, self.side)
        return (following, rows, cols, current, outcome.reward.reshape(-1),
                jnp.repeat(outcome.terminated, 2), jnp.repeat(outcome.truncated, 2), outcome.last_state)

    def start(self, indices):
        """Start an explicit batch of pool maps, used to evaluate both colors on identical boards."""
        if self.pool is None or len(indices) != self.count or any(not 0 <= index < self.pool_size for index in indices):
            raise ValueError("Expected one prepared map index per environment")
        with jax.default_device(self.backend):
            chosen = jnp.asarray(indices, jnp.int32)
            self.states = jax.tree.map(lambda value: value[chosen], self.pool)
            self.states = self.states._replace(pool_idx=(chosen + 1) % self.pool_size)
            self.rows, self.cols = self.dimensions[chosen].T
            self.last = self.current = None

    def read(self):
        if self.states is None:
            raise RuntimeError("Prepare the arena first")
        # JAX's process default may differ from the device selected for this arena.
        if self.current is None:
            with jax.default_device(self.backend):
                self.current = self.read_arrays(self.states, self.rows, self.cols)
        return View(*self.current)

    def step(self, actions):
        if actions.shape != (2 * self.count,):
            raise ValueError("Expected one action per player")
        with jax.default_device(self.backend):
            result = self.advance(self.states, self.rows, self.cols, actions.astype(jnp.int32), self.pool, self.dimensions)
        self.last = (result[7], self.rows, self.cols)
        self.states, self.rows, self.cols, self.current = result[:4]
        return result[4:7]

    def final(self):
        if self.last is None:
            raise RuntimeError("Step the arena first")
        # only time-limit bootstrapping needs the old game's final view during collection.
        with jax.default_device(self.backend):
            return View(*self.read_arrays(*self.last))

    def close(self):
        if self.states is not None:
            jax.block_until_ready(self.states)
        self.states = self.pool = self.last = self.current = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
