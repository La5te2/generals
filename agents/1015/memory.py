"""Lossless observation archive and aligned rolling interval queries.

Packed player-visible facts, float32 armies, own commands and public scores share
one timeline. An extra boundary frame supports differences at the window start.
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp

WINDOW, CAPACITY, CHANNELS = 512, 513, 48
FRAME_CHANNELS = 24


class Archive(NamedTuple):
    flags: object
    army: object
    commands: object
    public: object
    count: object
    cursor: object


class Reading(NamedTuple):
    spatial: object
    public: object
    valid: object
    counts: object
    immediate: object
    flags: object
    army: object
    commands: object


def empty(area):
    return Archive(jnp.zeros((CAPACITY, area), jnp.uint16),
                   jnp.zeros((CAPACITY, area), jnp.float32),
                   jnp.full(CAPACITY, area * 8, jnp.int32),
                   jnp.zeros((CAPACITY, 11), jnp.float32),
                   jnp.array(0, jnp.int32), jnp.array(0, jnp.int32))


def last_row(array, cursor):
    """Read the latest row with arbitrary leading batch dimensions."""
    index = ((cursor - 1) % CAPACITY)[..., None, None]
    return jnp.take_along_axis(array, index, axis=-2)[..., 0, :]


def last_command(archive):
    index = ((archive.cursor - 1) % CAPACITY)[..., None]
    return jnp.take_along_axis(archive.commands, index, axis=-1)[..., 0]


def append(archive, cells, events, command, public):
    bits = jnp.sum(cells[:, :9].astype(jnp.uint16) << jnp.arange(9, dtype=jnp.uint16), -1, dtype=jnp.uint16)
    flags = bits | (events.astype(jnp.uint16) << 9)
    i = archive.cursor
    return Archive(archive.flags.at[i].set(flags), archive.army.at[i].set(cells[:, 9]),
                   archive.commands.at[i].set(command), archive.public.at[i].set(public),
                   jnp.minimum(archive.count + 1, CAPACITY), (i + 1) % CAPACITY)


def ordered(archive):
    """Return chronological rows with explicit validity for left padding."""
    index = (archive.cursor + jnp.arange(CAPACITY)) % CAPACITY
    valid = jnp.arange(CAPACITY) >= CAPACITY - archive.count
    return (archive.flags[index], archive.army[index], archive.commands[index],
            archive.public[index], valid)


def frames(reading, topology, start, length):
    """Decode one ordered interval, including its predecessor for valid changes.

    Channels contain observation (10), source/target/half (3), own/enemy
    change and overlap (3), archive events (6), command-valid and pass (2).
    The command in row s precedes observation s. Padding is explicitly masked.
    """
    flags = jax.lax.dynamic_slice_in_dim(reading.flags, start, length + 1)
    logs = jax.lax.dynamic_slice_in_dim(reading.army, start, length + 1)
    commands = jax.lax.dynamic_slice_in_dim(reading.commands, start, length)
    valid = jax.lax.dynamic_slice_in_dim(reading.valid, start, length)
    now, before = flags[1:], flags[:-1]
    events = ((now[..., None] >> jnp.arange(9, 15, dtype=jnp.uint16)) & 1).astype(jnp.float32)
    real = valid[:, None] & (events[..., 5] > 0)
    observation = ((now[..., None] >> jnp.arange(9, dtype=jnp.uint16)) & 1).astype(jnp.float32)
    observation = jnp.concatenate((observation, logs[1:, :, None]), -1)
    area = now.shape[-1]
    command_valid = valid & (events[..., 4] > 0).any(-1)
    moving = command_valid & (commands >= 0) & (commands < area * 8)
    source = jnp.clip(commands // 8, 0, area - 1)
    target = topology.neighbors[source, (commands % 8) // 2]
    src = (jnp.arange(area)[None] == source[:, None]) & moving[:, None]
    dst = (jnp.arange(area)[None] == target[:, None]) & moving[:, None]
    half = src & (commands[:, None] % 2 == 1)
    overlap = ((now & 1) != 0) & ((before & 1) != 0) & (events[..., 4] > 0)
    change = (jnp.expm1(logs[1:] * jnp.log(256.)) - jnp.expm1(logs[:-1] * jnp.log(256.))) / 256.
    own_pair = overlap & ((now & 2) != 0) & ((before & 2) != 0)
    enemy_pair = overlap & ((now & 4) != 0) & ((before & 4) != 0)
    details = jnp.stack((src, dst, half, change * own_pair, change * enemy_pair, overlap), -1)
    command = jnp.stack((command_valid, command_valid & (commands == area * 8)), -1)
    command = jnp.broadcast_to(command[:, None], (*now.shape, 2))
    return jnp.concatenate((observation, details, events, command), -1) * real[..., None]


def read(archive, topology, period):
    """Read relative intervals inside a bounded inference or training microbatch."""
    flags, logs, commands, public, recorded = ordered(archive)
    flags = jnp.where(recorded[:, None], flags, 0)
    logs = jnp.where(recorded[:, None], logs, 0)
    commands = jnp.where(recorded, commands, flags.shape[-1] * 8)
    prior_flags, prior_logs, prior_public = flags[:-1], logs[:-1], public[:-1]
    flags, logs, commands, public, valid = flags[1:], logs[1:], commands[1:], public[1:], recorded[1:]
    area, segments = flags.shape[-1], WINDOW // period
    events = flags >> 9
    real = ((events & 32) != 0) & valid[:, None]
    own = ((flags & 2) != 0) & real
    enemy = ((events & 1) != 0) & ~own & real
    neutral = ((flags & 8) != 0) & real
    visible = ((flags & 1) != 0) & real
    army = jnp.maximum(jnp.rint(jnp.expm1(logs * jnp.log(256.))), 0)
    pos = jnp.broadcast_to((jnp.arange(WINDOW) % period + 1)[:, None], own.shape)

    def split(x):
        return x.reshape(segments, period, area)

    def total(x):
        return split(x.astype(jnp.float32)).sum(1)

    def last(mask):
        return split(jnp.where(mask, pos, 0)).max(1).astype(jnp.float32)

    def group(mask):
        active, amounts = split(mask), split(army)
        count = active.sum(1).astype(jnp.float32)
        first_at = jnp.where(count > 0, jnp.where(active, split(pos), period + 1).min(1), 0)
        last_at = last(mask)
        first_value = jnp.take_along_axis(amounts, jnp.maximum(first_at.astype(jnp.int32) - 1, 0)[:, None], 1)[:, 0]
        last_value = jnp.take_along_axis(amounts, jnp.maximum(last_at.astype(jnp.int32) - 1, 0)[:, None], 1)[:, 0]
        peak = jnp.where(active, amounts, 0).max(1)
        peak_at = jnp.where(active & (amounts == peak[:, None]), split(pos), 0).max(1)
        return [count, total(army * mask), peak, jnp.where(count > 0, first_value, 0),
                jnp.where(count > 0, last_value, 0), first_at, last_at, peak_at]

    out = [total(m) for m in (own, enemy, neutral, real & ~(own | enemy | neutral))]
    for owner in (own, enemy, neutral):
        out.extend(group(visible & owner))
    moving = (commands < area * 8) & valid & ((events & 16) != 0).any(-1)
    source = jnp.minimum(commands // 8, area - 1)
    target = topology.neighbors[source, (commands % 8) // 2]
    src = (jnp.arange(area)[None] == source[:, None]) & moving[:, None]
    dst = (jnp.arange(area)[None] == target[:, None]) & moving[:, None]
    half = src & (commands[:, None] % 2 == 1)
    out.extend([total(visible), total(src), total(dst), total(half), last(src), last(dst)])
    out.extend(total(((events & bit) != 0) & real) for bit in (4, 8, 2))
    ticks = jnp.rint(public[:, 0] * 512).astype(jnp.int32)
    ordinary = ((events & 16) != 0) & real
    producer = (flags & ((1 << 5) | (1 << 6))) != 0
    income = (ticks[:, None] % 50 == 0).astype(jnp.float32) + (ticks[:, None] % 2 == 0) * producer
    out.append(total(income * own * ordinary))
    observed_enemy = ((flags & 4) != 0) & real
    cover = observed_enemy
    for direction in range(8):
        neighbor = topology.neighbors[:, direction]
        cover = cover | (observed_enemy[:, neighbor] & topology.valid[None, :, direction])
    exposed = own & cover
    peak = jnp.where(split(exposed), split(army), 0).max(1)
    peak_at = jnp.where(split(exposed) & (split(army) == peak[:, None]), split(pos), 0).max(1)
    out.extend([total(exposed), total(exposed * army), peak, last(exposed), peak_at])
    consecutive = valid & recorded[:-1] & (ticks == jnp.rint(prior_public[:, 0] * 512).astype(jnp.int32) + 1)
    overlap = visible & ((prior_flags & 1) != 0) & consecutive[:, None]
    change = (jnp.expm1(logs * jnp.log(256.)) - jnp.expm1(prior_logs * jnp.log(256.))) / 256.
    own_pair = overlap & own & ((prior_flags & 2) != 0)
    enemy_pair = overlap & observed_enemy & ((prior_flags & 4) != 0)
    out.extend([total(change * own_pair), total(change * enemy_pair), total(own_pair), total(enemy_pair), total(overlap)])
    summary = jnp.stack(out, -1).astype(jnp.float32)
    counts = valid.reshape(segments, period).sum(-1)
    duration = jnp.maximum(counts[:, None], 1)
    normalized = [summary[..., i] / duration for i in range(CHANNELS)]
    for k in (4, 12, 20):
        for i in range(1, 5):
            value = summary[..., k + i] / jnp.maximum(summary[..., k], 1) if i == 1 else summary[..., k + i]
            normalized[k + i] = jnp.log1p(value) / jnp.log(256.)
        for i in (5, 6, 7):
            normalized[k + i] = summary[..., k + i] / period
    for i in (32, 33, 41, 42):
        normalized[i] = summary[..., i] / period
    for i in (37, 39, 40):
        normalized[i] = jnp.log1p(summary[..., i]) / jnp.log(256.)
    for i in (43, 44):
        normalized[i] = summary[..., i] / jnp.maximum(summary[..., i + 2], 1)
    immediate = jnp.stack((src[-1], dst[-1], half[-1], change[-1] * own_pair[-1],
                           change[-1] * enemy_pair[-1], overlap[-1]), -1).astype(jnp.float32)
    return Reading(jnp.stack(normalized, -1), public * valid[:, None], valid, counts, immediate,
                   jnp.concatenate((prior_flags[:1], flags)),
                   jnp.concatenate((prior_logs[:1], logs)), commands)
