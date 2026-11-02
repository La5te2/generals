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
from memory import Archive, append, empty as empty_archive, last_row

OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
VISIBLE, OWN, ENEMY, NEUTRAL, MOUNTAIN, CITY, GENERAL, FOG, OBSTACLE, ARMY = range(10)
SPATIAL, GLOBAL = 28, 11


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


class Inputs(NamedTuple):
    cells: jax.Array
    board: jax.Array
    movement: jax.Array
    legal: jax.Array
    archive: Archive

    @property
    def stats(self):
        return last_row(self.archive.public, self.archive.cursor)


class Memory(NamedTuple):
    last: jax.Array
    seen_at: jax.Array
    exposed: jax.Array
    enemy_army: jax.Array
    enemy_at: jax.Array
    occupied: jax.Array
    inputs: Inputs


def empty(area):
    return Memory(
        jnp.zeros((area, 7)),
        jnp.full((area,), -1, jnp.int32),
        jnp.zeros(area, bool),
        jnp.zeros(area),
        jnp.full(area, -1, jnp.int32),
        jnp.zeros(area, bool),
        Inputs(jnp.zeros((area, SPATIAL)), jnp.zeros(area, bool),
               jnp.zeros((area, 4), bool), jnp.zeros(area * 8 + 1, bool), empty_archive(area)),
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
    # Occupied cells can only change player ownership during a live 1v1 game.
    occupied = (memory.occupied | (x[:, OWN] > 0) | enemy_seen) & board
    prior_tick = jnp.rint(memory.inputs.stats[0] * 512).astype(jnp.int32)
    consecutive = (memory.inputs.archive.count > 0) & (tick == prior_tick + 1)
    prior_own = memory.inputs.cells[:, OWN] > 0
    own = x[:, OWN] > 0
    events = (
        occupied.astype(jnp.uint8)
        | ((visible & (memory.seen_at < 0)).astype(jnp.uint8) << 1)
        | ((consecutive & own & ~prior_own).astype(jnp.uint8) << 2)
        | ((consecutive & prior_own & ~own).astype(jnp.uint8) << 3)
        | ((board & consecutive).astype(jnp.uint8) << 4)
        | (board.astype(jnp.uint8) << 5)
    )
    archive = append(memory.inputs.archive, x, events, previous_action, obs.stats)
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
    result = Inputs(spatial, board, movement, legal, archive)
    return result, Memory(
        last,
        seen_at,
        exposed,
        enemy_army,
        enemy_at,
        occupied,
        result,
    )


def action(index, width, area):
    cell, rest = index // 8, index % 8
    cell = jnp.minimum(cell, area - 1)
    return jnp.array(
        [index == area * 8, cell // width, cell % width, rest // 2, rest % 2], jnp.int32
    )
