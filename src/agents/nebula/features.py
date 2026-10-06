"""Batch observation memory on the environment's device, matching features.hpp for deployment."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Memory(NamedTuple):
    board: jax.Array
    legal: jax.Array
    history: jax.Array
    age: jax.Array
    tick: jax.Array


def empty(count, side):
    return Memory(jnp.zeros((count, 38, side, side), jnp.float32),
                  jnp.zeros((count, 9, side, side), jnp.bool_),
                  jnp.zeros((count, 2, 512), jnp.float32),
                  jnp.zeros((count, side, side), jnp.float32),
                  jnp.full((count,), -1, jnp.int32))


def sight(cells):
    # each owned cell reveals its surrounding three-by-three square.
    return jax.lax.reduce_window(cells, False, jax.lax.bitwise_or, (1, 3, 3), (1, 1, 1), "SAME")


def update(memory, view, rows, cols):
    side = memory.board.shape[-1]
    row = jnp.arange(side)[None, :, None]
    col = jnp.arange(side)[None, None, :]
    inside = (row < rows[:, None, None]) & (col < cols[:, None, None])
    own, enemy = view.owned_cells & inside, view.opponent_cells & inside
    visible = ~(view.fog_cells | view.structures_in_fog) & inside
    army = jnp.where(inside, view.armies, 0).astype(jnp.float32)
    ours, theirs = jnp.where(own, army, 0), jnp.where(enemy, army, 0)
    elapsed = jnp.where(memory.tick < 0, 1, view.timestep - memory.tick)[:, None, None]
    age = jnp.where(theirs > 0, 0, memory.age + elapsed)
    old = memory.board
    board = jnp.zeros_like(old)
    board = board.at[:, 0].set(army)
    board = board.at[:, 1].set(ours)
    board = board.at[:, 2].set(theirs)
    board = board.at[:, 3].set(jnp.where(~(own | enemy) & visible, army, 0))
    board = board.at[:, 4].set(jnp.maximum(old[:, 4], sight(own)))
    board = board.at[:, 5].set(jnp.maximum(old[:, 5], sight(enemy)))
    board = board.at[:, 6].set(jnp.maximum(old[:, 6], view.generals & inside))
    board = board.at[:, 7].set(jnp.maximum(old[:, 7], view.castles & inside))
    # padded cells use the same remembered boundary markers as the C++ encoder.
    mountains = jnp.where(inside, jnp.maximum(old[:, 8], view.mountains), board[:, 4])
    board = board.at[:, 8].set(mountains)
    board = board.at[:, 9].set(~(own | enemy | view.mountains) & visible)
    board = board.at[:, 10].set(own)
    board = board.at[:, 11].set(enemy)
    board = board.at[:, 12].set(view.fog_cells & inside)
    board = board.at[:, 13].set(jnp.where(inside, view.structures_in_fog, 1 - mountains))
    for channel, value in enumerate((view.timestep, view.timestep % 50 / 50,
                                     view.owned_land_count, view.owned_army_count,
                                     view.opponent_land_count, view.opponent_army_count), start=14):
        board = board.at[:, channel].set(value[:, None, None])
    board = board.at[:, 20].set(jnp.where(theirs > 0, theirs, old[:, 20]))
    board = board.at[:, 21].set(jnp.log1p(age) / 5)
    board = board.at[:, 22].set(col / (side - 1))
    board = board.at[:, 23].set(row / (side - 1))
    board = board.at[:, 24].set(ours - old[:, 1])
    board = board.at[:, 25:31].set(old[:, 24:30])
    board = board.at[:, 31].set(theirs - old[:, 2])
    board = board.at[:, 32:38].set(old[:, 31:37])

    # action channels contain four full moves, four half moves, and a pass at every cell.
    target = jnp.pad(inside & ~view.mountains, ((0, 0), (1, 1), (1, 1)))
    adjacent = jnp.stack((target[:, :-2, 1:-1], target[:, 2:, 1:-1],
                          target[:, 1:-1, :-2], target[:, 1:-1, 2:]), axis=1)
    moves = adjacent & (own & (army >= 2))[:, None]
    legal = jnp.concatenate((moves, moves, jnp.ones_like(moves[:, :1])), axis=1)
    scores = jnp.stack((view.opponent_army_count, view.opponent_land_count), axis=1)
    history = jnp.concatenate((memory.history[:, :, 1:], scores[:, :, None]), axis=2)
    return Memory(board, legal, history, age, view.timestep)
