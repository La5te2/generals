"""Preserve recent observations and dated facts without predicting hidden state or scoring actions."""

from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class View(NamedTuple):
    terrain: jax.Array
    owner: jax.Array
    army: jax.Array
    stats: jax.Array
    inside: jax.Array


class Inputs(NamedTuple):
    board: jax.Array       # [batch, time, terrain/owner/army, row, column], newest first
    scalars: jax.Array     # [batch, time, tick/own land/own army/enemy land/enemy army/previous action]
    valid: jax.Array       # distinguishes an unavailable historical observation from a zero-valued one
    memory: jax.Array      # dated visible-cell, enemy-sight and enemy-army records
    inside: jax.Array
    legal: jax.Array
    timeline: jax.Array    # raw public tick/land/army records, newest first; tick -1 means unavailable


CONTRACT = {
    "board": ("terrain", "owner", "army"),
    "scalars": ("tick", "own_land", "own_army", "enemy_land", "enemy_army", "previous_action"),
    "memory": ("terrain", "owner", "army", "seen_at", "exposed_at", "enemy_army", "enemy_at"),
    "directions": ("up", "down", "left", "right"),
    "timeline": ("tick", "own_land", "own_army", "enemy_land", "enemy_army"),
}


def adjacent(value):
    padded = jnp.pad(value, [(0, 0)] * (value.ndim - 2) + [(1, 1), (1, 1)])
    return jnp.stack((padded[..., :-2, 1:-1], padded[..., 2:, 1:-1],
                      padded[..., 1:-1, :-2], padded[..., 1:-1, 2:]), axis=-3)


def coverage(value):
    """Count sight sources within one row and one column; terrain does not block sight."""
    return jax.lax.reduce_window(value.astype(jnp.float32), 0., jax.lax.add,
                                 (1,) * (value.ndim - 2) + (3, 3), (1,) * value.ndim, "SAME")


def encode_action(action, side):
    if action == 8 * side * side:
        return (1, 0, 0, 0, 0)
    if not 0 <= action < 8 * side * side:
        raise ValueError("Action index outside the policy head")
    kind, cell = divmod(action, side * side)
    return (0, cell // side, cell % side, kind % 4, kind // 4)


class Encoder(NamedTuple):
    """Immutable observation history; update returns new history and network inputs."""

    board: jax.Array
    scalars: jax.Array
    valid: jax.Array
    memory: jax.Array
    tick: jax.Array
    action: jax.Array
    timeline: jax.Array

    @staticmethod
    def empty(count, side, history=8, temporal=512):
        memory = jnp.zeros((count, 7, side, side)).at[:, jnp.array([3, 4, 6])].set(-1)
        return Encoder(jnp.zeros((count, history + 1, 3, side, side)),
                       jnp.zeros((count, history + 1, 6)), jnp.zeros((count, history + 1), jnp.bool_),
                       memory, jnp.full(count, -1, jnp.int32), jnp.full(count, -1, jnp.int32),
                       jnp.zeros((count, temporal, 5)).at[:, :, 0].set(-1))

    def submitted(self, action):
        # this records a submitted command, not a claim that the server executed it.
        return self._replace(action=action)

    def update(self, view, reset):
        terrain, owner, army, stats, inside = view
        ticks = stats[:, 0].astype(jnp.int32)
        reset = reset | (self.tick < 0) | (ticks < self.tick)
        changed = reset | (ticks != self.tick)
        board = jnp.where(reset[:, None, None, None, None], 0, self.board)
        scalars = jnp.where(reset[:, None, None], 0, self.scalars)
        valid = self.valid & ~reset[:, None]
        timeline = jnp.where(reset[:, None, None], 0, self.timeline)
        timeline = timeline.at[:, :, 0].set(jnp.where(reset[:, None], -1, timeline[:, :, 0]))
        timeline = jnp.concatenate((stats.astype(jnp.float32)[:, None], timeline[:, :-1]), axis=1)
        timeline = jnp.where(changed[:, None, None], timeline, self.timeline)
        memory = jnp.where(reset[:, None, None, None], 0, self.memory)
        dates = jnp.array([3, 4, 6])
        memory = memory.at[:, dates].set(jnp.where(reset[:, None, None, None], -1, memory[:, dates]))
        visible = inside & (terrain != 0) & (terrain != 5)
        owner, army = jnp.where(visible, owner, 0), jnp.where(visible, army, 0)
        current = jnp.stack((terrain, owner, army), axis=1).astype(jnp.float32) * inside[:, None]
        previous = jnp.where(reset, -1, self.action)
        scalar = jnp.concatenate((stats.astype(jnp.float32), previous[:, None].astype(jnp.float32)), axis=1)
        board = jnp.concatenate((current[:, None], board[:, :-1]), axis=1)
        scalars = jnp.concatenate((scalar[:, None], scalars[:, :-1]), axis=1)
        valid = jnp.concatenate((jnp.ones_like(valid[:, :1]), valid[:, :-1]), axis=1)
        board = jnp.where(changed[:, None, None, None, None], board, self.board)
        scalars = jnp.where(changed[:, None, None], scalars, self.scalars)
        valid = jnp.where(changed[:, None], valid, self.valid)
        memory = memory.at[:, :3].set(jnp.where(visible[:, None], current, memory[:, :3]))
        memory = memory.at[:, 3].set(jnp.where(visible, ticks[:, None, None], memory[:, 3]))
        exposed = (coverage((owner == 2) & visible) > 0) & inside
        memory = memory.at[:, 4].set(jnp.where(exposed, ticks[:, None, None], memory[:, 4]))
        enemy = (owner == 2) & visible
        memory = memory.at[:, 5].set(jnp.where(enemy, army, memory[:, 5]))
        memory = memory.at[:, 6].set(jnp.where(enemy, ticks[:, None, None], memory[:, 6]))
        memory = jnp.where(changed[:, None, None, None], memory, self.memory)
        # unknown obstacles can be cities, so only known impossibilities are excluded.
        mountain = ((memory[:, 0] == 2) & (memory[:, 3] >= 0)) | (terrain == 2)
        moves = (inside & (owner == 1) & (army >= 2))[:, None] & adjacent(inside & ~mountain)
        legal = jnp.concatenate((jnp.tile(moves, (1, 2, 1, 1)).reshape(len(ticks), -1),
                                 jnp.ones((len(ticks), 1), jnp.bool_)), axis=1)
        encoder = Encoder(board, scalars, valid, memory, ticks, self.action, timeline)
        return encoder, Inputs(board, scalars, valid, memory, inside, legal, timeline)


def spatial(value, code):
    """Select a spatial permutation without compiling a separate rollout for each orientation."""
    side = value.shape[-1]
    cells = jnp.asarray(mappings(side)[0, :, :side * side]) % (side * side)
    return jnp.take(value.reshape((*value.shape[:-2], side * side)), cells[code], axis=-1).reshape(value.shape)


def directions(code):
    vectors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    order = [0] * 4
    for old, (row, col) in enumerate(vectors):
        if code >= 4:
            col = -col
        for rotation in range(code % 4):
            row, col = -col, row
        order[vectors.index((row, col))] = old
    return order


@lru_cache(maxsize=8)
def mappings(side):
    """Host constants become compiled gathers; no per-step construction or sorting."""
    area = side * side
    actions = np.arange(8 * area + 1, dtype=np.int32)
    grid = actions[:-1].reshape(8, side, side)
    restored = []
    for code in range(8):
        order = directions(code)
        oriented = np.rot90(np.flip(grid, -1) if code >= 4 else grid, code % 4, (-2, -1))
        restored.append(np.concatenate((oriented[order + [d + 4 for d in order]].ravel(), actions[-1:])))
    restored = np.stack(restored)
    transformed = np.empty_like(restored)
    np.put_along_axis(transformed, restored, np.broadcast_to(actions, restored.shape), axis=1)
    return np.stack((restored, transformed))


def restore(action, side, code):
    return jnp.asarray(mappings(side)[0])[code, action]


def transform(inputs, code):
    board, scalars, valid, memory, inside, legal, timeline = inputs
    restored, transformed = jnp.asarray(mappings(board.shape[-1]))[:, code]
    previous = scalars[:, :, 5].astype(jnp.int32)
    scalars = scalars.at[:, :, 5].set(jnp.where(previous >= 0, transformed[jnp.maximum(previous, 0)], -1))
    return Inputs(spatial(board, code), scalars, valid, spatial(memory, code), spatial(inside, code),
                  legal[:, restored], timeline)
