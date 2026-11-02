"""Shared rollout timelines and indexed reconstruction of observation windows.

Current persistent features are stored once per transition. Packed observation
frames, own commands and public scores share a chronological archive across
samples. Optimization shares one device timeline and reconstructs microbatches.
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from critic import Oracle
from features import SPATIAL, Inputs
from memory import Archive, CAPACITY, last_row
from economy import CHANNELS, HORIZONS, targets as economy_targets
from gathering import empty as empty_gathering

BOOLEAN = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 21, 22, 23, 24)
REAL = (16, 18, 19, 20, 25, 26, 27)


class Samples(NamedTuple):
    inputs: object
    actions: jax.Array
    old_logp: jax.Array
    advantage: jax.Array
    returns: jax.Array
    targets: dict
    oracle: Oracle


def bits(value, start=0):
    shifts = jnp.arange(start, start + value.shape[-1], dtype=jnp.uint32)
    return jnp.sum(value.astype(jnp.uint32) << shifts, axis=-1, dtype=jnp.uint32)


def pack(inputs):
    flags = bits(inputs.cells[..., jnp.array(BOOLEAN)])
    flags |= inputs.board.astype(jnp.uint32) << 20
    flags |= bits(inputs.movement, 21)
    legal = inputs.legal[..., :-1].reshape((*inputs.board.shape, 4, 2))
    flags |= bits(legal[..., 0], 25)
    return flags, inputs.cells[..., jnp.array(REAL)]


def unpack_bits(flags, start, count):
    return ((flags[..., None] >> jnp.arange(start, start + count, dtype=jnp.uint32)) & 1).astype(bool)


class PackedBatch(NamedTuple):
    flags: object
    values: object
    archive: "Archive | WindowIndex"
    actions: object
    old_logp: object
    advantage: object
    returns: object
    targets: dict
    oracle_flags: object

    def unpack(self, timeline=None):
        archive = self.archive.materialize(timeline) if isinstance(self.archive, WindowIndex) else self.archive
        binary = unpack_bits(self.flags, 0, len(BOOLEAN)).astype(jnp.float32)
        columns = {k: binary[..., i] for i, k in enumerate(BOOLEAN)}
        columns.update({k: self.values[..., i] for i, k in enumerate(REAL)})
        columns[9] = last_row(archive.army, archive.cursor)
        cells = jnp.stack([columns[i] for i in range(SPATIAL)], -1)
        movement = unpack_bits(self.flags, 21, 4)
        legal = jnp.repeat(unpack_bits(self.flags, 25, 4), 2, -1)
        legal = jnp.concatenate((legal.reshape((len(cells), -1)), jnp.ones((len(cells), 1), bool)), -1)
        inputs = Inputs(cells, unpack_bits(self.flags, 20, 1)[..., 0], movement, legal, archive)
        return Samples(inputs, self.actions, self.old_logp, self.advantage, self.returns,
                       self.targets, Oracle(self.oracle_flags, self.targets["army"], self.targets["enemy_action"]))


class Record(NamedTuple):
    flags: object
    values: object
    stats: object
    actions: object
    old_logp: object
    value: object
    bootstrap: object
    reward: object
    terminal: object
    truncated: object
    movable: object
    targets: dict
    outcome: object
    material: object
    reward_parts: object
    truncated_potential: object
    chosen_q: object
    oracle_flags: object
    economy_flow: object
    archive_flags: object
    archive_army: object
    previous_command: object
    archive_count: object


class Timeline(NamedTuple):
    """One device copy of the immutable rollout observation timeline."""

    flags: object
    army: object
    commands: object
    public: object


class WindowIndex(NamedTuple):
    """Sample indices reconstructed inside each device microbatch."""

    times: object
    lanes: object
    count: object

    def materialize(self, timeline):
        if timeline is None:
            raise ValueError("Indexed batches require their shared rollout timeline")
        window = self.times[:, None] + jnp.arange(CAPACITY)[None]
        rows = self.lanes[:, None]
        return Archive(timeline.flags[window, rows], timeline.army[window, rows],
                       timeline.commands[window, rows], timeline.public[window, rows],
                       self.count, jnp.zeros_like(self.count))


class Rollout:
    """Keep one shared archive timeline, including the rollout-start prefix."""

    archive_prefix = CAPACITY - 1

    def __init__(self, steps, environments, area):
        self.steps, self.lanes, self.area = steps, environments, area
        shape = (steps, environments)
        frames = (steps + CAPACITY - 1, environments)
        self.flags = np.empty((*shape, area), np.uint32)
        self.values = np.empty((*shape, area, len(REAL)), np.float32)
        self.archive_flags = np.empty((*frames, area), np.uint16)
        self.army = np.empty((*frames, area), np.float32)
        self.commands = np.empty(frames, np.int32)
        self.public = np.empty((*frames, 11), np.float32)
        self.archive_count = np.empty(shape, np.int32)
        self.oracle_flags = np.empty((*shape, area), np.uint8)
        self.actions = np.empty(shape, np.int32)
        for name in ("old_logp", "value", "bootstrap", "reward", "advantage", "returns", "chosen_q"):
            setattr(self, name, np.empty(shape, np.float32))
        for name in ("terminal", "truncated", "movable"):
            setattr(self, name, np.empty(shape, bool))
        self.economy_flow = np.empty((*shape, CHANNELS), np.float32)
        self.targets = {
            "general": np.empty(shape, np.int32),
            "owner": np.empty((*shape, area), np.uint8),
            "army": np.empty((*shape, area), np.float32),
            "enemy_action": np.empty(shape, np.int32),
            "economy": np.zeros((*shape, len(HORIZONS), CHANNELS), np.float32),
            "economy_valid": np.zeros((*shape, len(HORIZONS)), bool),
        }
        self.gathering = empty_gathering(1)
        self.gathering_index = np.zeros(self.count, np.int32)

    @property
    def nbytes(self):
        return sum(x.nbytes for x in vars(self).values() if isinstance(x, np.ndarray)) + sum(
            x.nbytes for x in self.targets.values()) + sum(x.nbytes for x in self.gathering.values())

    @property
    def count(self):
        return self.steps * self.lanes

    def begin(self, inputs):
        archive = jax.device_get(inputs.archive)
        indices = (np.asarray(archive.cursor)[:, None] + np.arange(CAPACITY)) % CAPACITY
        rows = np.arange(self.lanes)[:, None]
        for target, source in (("archive_flags", "flags"), ("army", "army"),
                               ("commands", "commands"), ("public", "public")):
            timeline = np.asarray(getattr(archive, source))[rows, indices][:, :-1]
            getattr(self, target)[:CAPACITY - 1] = np.swapaxes(timeline, 0, 1)
        self.gathering = empty_gathering(1)
        self.gathering_index.fill(0)

    def write(self, offset, record):
        count = len(record.actions)
        rows = slice(offset, offset + count)
        frames = slice(offset + CAPACITY - 1, offset + CAPACITY - 1 + count)
        for name in ("flags", "values", "actions", "old_logp", "value", "bootstrap", "reward",
                     "terminal", "truncated", "movable", "chosen_q", "oracle_flags",
                     "economy_flow", "archive_count"):
            getattr(self, name)[rows] = getattr(record, name)
        for target, source in (("archive_flags", "archive_flags"), ("army", "archive_army"),
                               ("commands", "previous_command"), ("public", "stats")):
            getattr(self, target)[frames] = getattr(record, source)
        for name, value in record.targets.items():
            self.targets[name][rows] = value

    def finish_economy(self):
        values, valid = economy_targets(self.economy_flow, self.terminal, self.truncated)
        self.targets["economy"][:] = values
        self.targets["economy_valid"][:] = valid

    def device_timeline(self):
        """Upload shared rows once after collection, before all optimization passes."""
        return jax.device_put(Timeline(self.archive_flags, self.army, self.commands, self.public))

    def read(self, indices, indexed=False):
        times, lanes = np.divmod(np.asarray(indices), self.lanes)
        if indexed:
            archive = WindowIndex(times.astype(np.int32), lanes.astype(np.int32), self.archive_count[times, lanes])
        else:
            window = times[:, None] + np.arange(CAPACITY)[None]
            rows = lanes[:, None]
            archive = Archive(self.archive_flags[window, rows], self.army[window, rows],
                              self.commands[window, rows], self.public[window, rows],
                              self.archive_count[times, lanes], np.zeros(len(times), np.int32))
        targets = {k: v[times, lanes] for k, v in self.targets.items()}
        targets.update({"gathering_" + k: v[self.gathering_index[indices]] for k, v in self.gathering.items()})
        return PackedBatch(self.flags[times, lanes], self.values[times, lanes], archive,
                           self.actions[times, lanes], self.old_logp[times, lanes],
                           self.advantage[times, lanes], self.returns[times, lanes],
                           targets, self.oracle_flags[times, lanes])
