"""Player-visible facts and bounded history shared by rollout and evaluation.

Unknown cells are not neutral. Last-seen ownership/army are timestamped facts,
not predictions. No shortest path, future income, threat score or hidden enemy
view enters the policy. Eight-neighbor sight and four-neighbor transport remain
separate relations. A finite history is an approximation, not a sufficient belief.
"""

from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
VISIBLE, OWN, ENEMY, NEUTRAL, MOUNTAIN, CITY, GENERAL, FOG, OBSTACLE, ARMY = range(10)
SPATIAL, TEMPORAL, GLOBAL = 28, 13, 11
PUBLIC_HISTORY = 512


class Graph(NamedTuple):
    neighbors: jax.Array
    valid: jax.Array
    offsets: jax.Array
    positions: jax.Array


@lru_cache(maxsize=16)
def graph(height, width):
    positions = np.stack(
        np.unravel_index(np.arange(height * width), (height, width)), -1
    )
    targets = positions[:, None] + np.asarray(OFFSETS)[None]
    valid = (
        (targets >= 0).all(-1) & (targets[..., 0] < height) & (targets[..., 1] < width)
    )
    neighbors = np.where(
        valid,
        targets[..., 0] * width + targets[..., 1],
        np.arange(height * width)[:, None],
    )
    return Graph(
        jnp.asarray(neighbors),
        jnp.asarray(valid),
        jnp.asarray(OFFSETS, jnp.float32),
        jnp.asarray(positions, jnp.float32),
    )


class Observation(NamedTuple):
    cells: jax.Array  # N x 10, current masked view only
    stats: jax.Array  # 11 public scalars, including rule configuration
    board: jax.Array  # False only for padding


class Memory(NamedTuple):
    last: jax.Array
    seen_at: jax.Array
    exposed: jax.Array
    recent: jax.Array
    stats: jax.Array
    valid: jax.Array
    enemy_army: jax.Array
    enemy_at: jax.Array
    public: jax.Array
    public_valid: jax.Array


class Inputs(NamedTuple):
    cells: jax.Array
    recent: jax.Array
    stats: jax.Array
    time_valid: jax.Array
    board: jax.Array
    movement: jax.Array
    legal: jax.Array
    public: jax.Array
    public_valid: jax.Array


def empty(area, history):
    if history < 1:
        raise ValueError("history must be positive")
    return Memory(
        jnp.zeros((area, 7)),
        jnp.full((area,), -1, jnp.int32),
        jnp.zeros(area, bool),
        jnp.zeros((history, area, TEMPORAL)),
        jnp.zeros((history, GLOBAL)),
        jnp.zeros(history, bool),
        jnp.zeros(area),
        jnp.full(area, -1, jnp.int32),
        jnp.zeros((PUBLIC_HISTORY, GLOBAL)),
        jnp.zeros(PUBLIC_HISTORY, bool),
    )


def log_army(value):
    return jnp.log1p(jnp.maximum(value, 0)) / jnp.log(256.0)


def observe(
    armies, owners, mountains, cities, generals, tick, player, rules, board, topology
):
    """Mask a simulator state BEFORE feature encoding; rules=(regen, old_order, trade).

    This adapter consumes truth only to reproduce the environment's observation.
    It never computes opponent visibility or a hidden-state estimate for the actor.
    """
    armies, owners = armies.reshape(-1), owners.reshape(-1)
    mountains, cities, generals, board = [
        x.reshape(-1) for x in (mountains, cities, generals, board)
    ]
    own = (owners == player) & board
    adjacent = topology.valid & board[topology.neighbors]
    visible = (own | jnp.any(own[topology.neighbors] & adjacent, -1)) & board
    fog = ~visible & board
    enemy = (owners == 1 - player) & visible
    fields = [
        visible,
        own,
        enemy,
        (owners == -1) & visible & ~mountains,
        mountains & visible,
        cities & visible,
        generals & visible,
        fog & ~(mountains | cities),
        fog & (mountains | cities),
        jnp.where(visible, log_army(armies), 0.0),
    ]
    area = jnp.maximum(board.sum(), 1)
    stats = jnp.array(
        [
            tick / 512.0,
            (tick % 50) / 50.0,
            tick % 2,
            log_army(jnp.sum(jnp.where(own, armies, 0))),
            log_army(jnp.sum(jnp.where((owners == 1 - player) & board, armies, 0))),
            own.sum() / area,
            ((owners == 1 - player) & board).sum() / area,
            player,
            *rules,
        ],
        jnp.float32,
    )
    return Observation(jnp.stack(fields, -1).astype(jnp.float32), stats, board)


def encode(obs, memory, previous_action, topology):
    """Append one observation exactly once. Reset by supplying empty() at episode start.

    Previous action is the player's own command, not the hidden opponent action.
    Exposure counts are known lower bounds away from own land; on own land the
    count is exact because an enemy sight source within one cell is also visible.
    """
    x, board = obs.cells, obs.board
    tick = jnp.rint(obs.stats[0] * 512).astype(jnp.int32)
    visible = x[:, VISIBLE] > 0
    last = jnp.where(
        visible[:, None],
        x[:, jnp.array([OWN, ENEMY, NEUTRAL, MOUNTAIN, CITY, GENERAL, ARMY])],
        memory.last,
    )
    seen_at = jnp.where(visible, tick, memory.seen_at)
    seen = seen_at >= 0
    age = jnp.where(seen, log_army(tick - seen_at), 0.0)
    enemy_seen = x[:, ENEMY] > 0
    enemy_army = jnp.where(enemy_seen, x[:, ARMY], memory.enemy_army)
    enemy_at = jnp.where(enemy_seen, tick, memory.enemy_at)
    edges = topology.valid & board[:, None] & board[topology.neighbors]
    own_cover = x[:, OWN] + jnp.sum(x[topology.neighbors, OWN] * edges, -1)
    enemy_cover = x[:, ENEMY] + jnp.sum(x[topology.neighbors, ENEMY] * edges, -1)
    exposed = memory.exposed | (enemy_cover > 0)
    known_mountain = seen & (last[:, 3] > 0)
    movement = (
        edges[:, :4]
        & ~known_mountain[:, None]
        & ~known_mountain[topology.neighbors[:, :4]]
    )
    # Sources we own reveal every adjacent target, so this mask uses no hidden terrain.
    movable = (x[:, OWN] > 0) & (x[:, ARMY] >= log_army(2) - 1e-6)
    legal = jnp.concatenate(
        (
            jnp.repeat((movement & movable[:, None])[..., None], 2, -1).reshape(-1),
            jnp.ones(1, bool),
        )
    )
    area = len(board)
    source, remainder = previous_action // 8, previous_action % 8
    source = jnp.minimum(source, area - 1)
    target = topology.neighbors[source, remainder // 2]
    moving = previous_action < area * 8
    marks = jnp.stack(
        (
            (jnp.arange(area) == source) & moving,
            (jnp.arange(area) == target) & moving,
            (jnp.arange(area) == source) & moving & (remainder % 2 == 1),
        ),
        -1,
    )
    recent = jnp.concatenate(
        (memory.recent[1:], jnp.concatenate((x, marks), -1)[None]), 0
    )
    stats = jnp.concatenate((memory.stats[1:], obs.stats[None]), 0)
    valid = jnp.concatenate((memory.valid[1:], jnp.ones(1, bool)))
    public = jnp.concatenate((memory.public[1:], obs.stats[None]), 0)
    public_valid = jnp.concatenate((memory.public_valid[1:], jnp.ones(1, bool)))
    extra = jnp.stack(
        (
            seen,
            age,
            own_cover / 9.0,
            enemy_cover / 9.0,
            exposed,
            x[:, OWN] > 0,
            known_mountain,
            (x[:, OBSTACLE] > 0) & ~seen,
            enemy_at >= 0,
            enemy_army,
            jnp.where(enemy_at >= 0, log_army(tick - enemy_at), 0.0),
        ),
        -1,
    )
    spatial = jnp.concatenate((x, last, extra), -1) * board[:, None]
    result = Inputs(
        spatial, recent, stats, valid, board, movement, legal, public, public_valid
    )
    return result, Memory(
        last,
        seen_at,
        exposed,
        recent,
        stats,
        valid,
        enemy_army,
        enemy_at,
        public,
        public_valid,
    )


def action(index, width, area):
    cell, rest = index // 8, index % 8
    cell = jnp.minimum(cell, area - 1)
    return jnp.array(
        [index == area * 8, cell // width, cell % width, rest // 2, rest % 2], jnp.int32
    )
