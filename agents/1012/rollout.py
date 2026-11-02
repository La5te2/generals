"""Lossless host-side PPO storage with shared temporal frames.

Only a selected minibatch crosses to the accelerator. Binary features are packed
into bits; nonbinary features retain float32 precision. History windows are
gathered from one timeline per player and masked at episode boundaries.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from features import Inputs

BOOLEAN = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 21, 22, 23, 24)
REAL = (9, 16, 18, 19, 20)


class Samples(NamedTuple):
    inputs: object
    actions: jax.Array
    old_logp: jax.Array
    advantage: jax.Array
    returns: jax.Array
    targets: dict


def bits(value, start=0):
    shifts = jnp.arange(start, start + value.shape[-1], dtype=jnp.uint32)
    return jnp.sum(value.astype(jnp.uint32) << shifts, axis=-1, dtype=jnp.uint32)


def pack(inputs):
    """Use all 32 bits: spatial facts, board, edges, legal moves, action marks."""
    flags = bits(inputs.cells[..., jnp.array(BOOLEAN)])
    flags |= inputs.board.astype(jnp.uint32) << 20
    flags |= bits(inputs.movement, 21)
    legal = inputs.legal[..., :-1].reshape((*inputs.board.shape, 4, 2))
    flags |= bits(legal[..., 0], 25)
    flags |= bits(inputs.recent[..., -1, :, 10:13], 29)
    return flags, inputs.cells[..., jnp.array(REAL)]


def unpack_bits(flags, start, count):
    return ((flags[..., None] >> jnp.arange(start, start + count, dtype=jnp.uint32)) & 1).astype(bool)


class PackedBatch(NamedTuple):
    flags: object
    values: object
    recent_flags: object
    recent_army: object
    stats: object
    valid: object
    actions: object
    old_logp: object
    advantage: object
    returns: object
    targets: dict

    def unpack(self):
        binary = unpack_bits(self.flags, 0, len(BOOLEAN)).astype(jnp.float32)
        columns = {k: binary[..., i] for i, k in enumerate(BOOLEAN)}
        columns.update({k: self.values[..., i] for i, k in enumerate(REAL)})
        cells = jnp.stack([columns[i] for i in range(25)], -1)
        recent = jnp.concatenate((
            unpack_bits(self.recent_flags, 0, 9).astype(jnp.float32),
            self.recent_army[..., None],
            unpack_bits(self.recent_flags, 29, 3).astype(jnp.float32),
        ), -1)
        recent = jnp.where(self.valid[..., None, None], recent, 0)
        stats = jnp.where(self.valid[..., None], self.stats, 0)
        movement = unpack_bits(self.flags, 21, 4)
        legal = jnp.repeat(unpack_bits(self.flags, 25, 4), 2, -1)
        legal = jnp.concatenate((legal.reshape((len(cells), -1)), jnp.ones((len(cells), 1), bool)), -1)
        inputs = Inputs(cells, recent, stats, self.valid, unpack_bits(self.flags, 20, 1)[..., 0], movement, legal)
        return Samples(inputs, self.actions, self.old_logp, self.advantage, self.returns, self.targets)


class Record(NamedTuple):
    flags: object
    values: object
    stats: object
    history_length: object
    actions: object
    old_logp: object
    value: object
    bootstrap: object
    reward: object
    terminal: object
    truncated: object
    movable: object
    targets: dict


class Rollout:
    """Reusable host buffer. read() materializes only the requested compact batch."""

    def __init__(self, steps, environments, area, history):
        self.steps, self.lanes, self.area, self.history = steps, 2 * environments, area, history
        shape = (steps, self.lanes)
        frames = (steps + history - 1, self.lanes)
        self.flags = np.empty((*frames, area), np.uint32)
        self.values = np.empty((*frames, area, len(REAL)), np.float32)
        self.stats = np.empty((*frames, 11), np.float32)
        self.history_length = np.empty(shape, np.int32)
        self.actions = np.empty(shape, np.int32)
        self.old_logp = np.empty(shape, np.float32)
        self.value = np.empty(shape, np.float32)
        self.bootstrap = np.empty(shape, np.float32)
        self.reward = np.empty(shape, np.float32)
        self.terminal = np.empty(shape, bool)
        self.truncated = np.empty(shape, bool)
        self.movable = np.empty(shape, bool)
        self.advantage = np.empty(shape, np.float32)
        self.returns = np.empty(shape, np.float32)
        self.targets = {"general": np.empty(shape, np.int32),
                        "owner": np.empty((*shape, area), np.uint8),
                        "army": np.empty((*shape, area), np.float32)}

    @property
    def nbytes(self):
        return sum(x.nbytes for x in vars(self).values() if isinstance(x, np.ndarray)) + sum(x.nbytes for x in self.targets.values())

    @property
    def count(self):
        return self.steps * self.lanes

    def begin(self, inputs):
        # Keep the history preceding the first recorded observation of this rollout.
        recent = np.asarray(inputs.recent).reshape((self.lanes, self.history, self.area, 13))[:, :-1].transpose(1, 0, 2, 3)
        flags = np.sum(recent[..., :9].astype(np.uint32) << np.arange(9, dtype=np.uint32), -1, dtype=np.uint32)
        flags |= np.sum(recent[..., 10:13].astype(np.uint32) << np.arange(29, 32, dtype=np.uint32), -1, dtype=np.uint32)
        self.flags[:self.history - 1] = flags
        self.values[:self.history - 1] = 0
        self.values[:self.history - 1, ..., 0] = recent[..., 9]
        self.stats[:self.history - 1] = np.asarray(inputs.stats).reshape((self.lanes, self.history, 11))[:, :-1].transpose(1, 0, 2)

    def write(self, offset, record):
        count = len(record.actions)
        rows = slice(offset, offset + count)
        frames = slice(offset + self.history - 1, offset + self.history - 1 + count)
        for name in ("flags", "values", "stats"):
            value = getattr(record, name)
            getattr(self, name)[frames] = value.reshape((count, self.lanes, *value.shape[3:]))
        for name in ("history_length", "actions", "old_logp", "value", "bootstrap", "reward", "movable"):
            getattr(self, name)[rows] = getattr(record, name).reshape((count, self.lanes))
        for name in ("terminal", "truncated"):
            getattr(self, name)[rows] = np.repeat(getattr(record, name), 2, axis=-1)
        for name, value in record.targets.items():
            self.targets[name][rows] = value.reshape((count, self.lanes, *value.shape[3:]))

    def read(self, indices):
        times, lanes = np.divmod(np.asarray(indices), self.lanes)
        frames = times + self.history - 1
        window = frames[:, None] - np.arange(self.history - 1, -1, -1)
        valid = np.arange(self.history)[None] >= self.history - self.history_length[times, lanes, None]
        return PackedBatch(
            self.flags[frames, lanes], self.values[frames, lanes],
            self.flags[window, lanes[:, None]], self.values[window, lanes[:, None], :, 0],
            self.stats[window, lanes[:, None]], valid,
            self.actions[times, lanes], self.old_logp[times, lanes], self.advantage[times, lanes], self.returns[times, lanes],
            {k: v[times, lanes] for k, v in self.targets.items()},
        )
