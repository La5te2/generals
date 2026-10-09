"""Pack rollout history and store native Equinox policies and resumable training state."""

import json
from dataclasses import asdict, dataclass
from functools import partial
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from config import Config
from features import CONTRACT, Inputs
from model import GenFormer


class Frame(NamedTuple):
    cells: jax.Array
    armies: jax.Array
    scalars: jax.Array
    valid: jax.Array
    padding: jax.Array
    terrain: jax.Array
    memory: jax.Array
    inside: jax.Array
    legal: jax.Array
    stats: jax.Array
    reset: jax.Array


def pack(inputs, reset):
    board, scalars, valid, memory, inside, legal, timeline = inputs
    cells = board[:, 0, 0].astype(jnp.uint8) | (board[:, 0, 1].astype(jnp.uint8) << 3)
    terrain = memory[:, 0].astype(jnp.uint8) | (memory[:, 1].astype(jnp.uint8) << 3)
    count, side = inside.shape[:2]
    moves = legal[:, :4 * side**2].reshape(count, 4, side, side).astype(jnp.uint8)
    bits = moves[:, 0] | (moves[:, 1] << 1) | (moves[:, 2] << 2) | (moves[:, 3] << 3)
    return Frame(cells, board[:, 0, 2], scalars[:, 0], valid, scalars[:, -1],
                 terrain, memory[:, 2:], inside, bits, timeline[:, 0], reset)


class Rollout(NamedTuple):
    cells: jax.Array
    armies: jax.Array
    scalars: jax.Array
    valid: jax.Array
    padding: jax.Array
    terrain: jax.Array
    memory: jax.Array
    inside: jax.Array
    legal: jax.Array
    timeline: jax.Array
    starts: jax.Array

    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def empty(initial, steps):
        """Allocate one rollout buffer, including the history that predates its first step."""
        packed = initial.board[:, 1:, 0].astype(jnp.uint8) | (initial.board[:, 1:, 1].astype(jnp.uint8) << 3)
        def prefix(value):
            value = jnp.flip(value, 1).swapaxes(0, 1)
            return jnp.pad(value, ((0, steps), *((0, 0),) * (value.ndim - 1)))
        frame = pack(initial, jnp.zeros(len(initial.board), jnp.bool_))
        def allocate(value):
            return jnp.zeros((steps, *value.shape), value.dtype)
        return Rollout(prefix(packed), prefix(initial.board[:, 1:, 2]), prefix(initial.scalars[:, 1:]),
                       allocate(frame.valid), allocate(frame.padding), allocate(frame.terrain),
                       allocate(frame.memory), allocate(frame.inside), allocate(frame.legal),
                       prefix(initial.timeline[:, 1:]), allocate(frame.reset).astype(jnp.int32))

    @partial(jax.jit, donate_argnums=(0,))
    def write(self, frames, offset):
        """Consume this buffer and return its updated owner; no complete rollout is copied."""
        history = self.valid.shape[2] - 1
        temporal = len(self.timeline) - len(self.inside) + 1
        def write(buffer, values, prefix=0):
            return jax.lax.dynamic_update_slice_in_dim(buffer, values, offset + prefix, axis=0)
        def start(previous, data):
            step, reset = data
            current = jnp.where((step > 0) & reset, step, previous)
            return current, current
        previous = jnp.where(offset > 0, self.starts[jnp.maximum(offset - 1, 0)], 1 - temporal)
        starts = jax.lax.scan(start, previous, (offset + jnp.arange(len(frames.cells)), frames.reset))[1]
        return Rollout(write(self.cells, frames.cells, history), write(self.armies, frames.armies, history),
                       write(self.scalars, frames.scalars, history), write(self.valid, frames.valid),
                       write(self.padding, frames.padding), write(self.terrain, frames.terrain),
                       write(self.memory, frames.memory), write(self.inside, frames.inside),
                       write(self.legal, frames.legal), write(self.timeline, frames.stats, temporal - 1),
                       write(self.starts, starts))

    @property
    def nbytes(self):
        return sum(value.size * value.dtype.itemsize for value in self)

    @jax.jit
    def read(self, ids):
        count, side = self.inside.shape[1:3]
        history = self.valid.shape[2] - 1
        temporal = len(self.timeline) - len(self.inside) + 1
        step, player = ids // count, ids % count
        recent = step[:, None] + history - jnp.arange(history + 1)
        cells = self.cells[recent, player[:, None]]
        valid = self.valid[step, player]
        board = jnp.stack(((cells & 7).astype(jnp.float32), (cells >> 3).astype(jnp.float32),
                           self.armies[recent, player[:, None]]), axis=2)
        board = jnp.where(valid[:, :, None, None, None], board, 0)
        scalars = jnp.where(valid[:, :, None], self.scalars[recent, player[:, None]], self.padding[step, player][:, None])
        terrain = self.terrain[step, player]
        memory = jnp.concatenate(((terrain & 7).astype(jnp.float32)[:, None],
                                  (terrain >> 3).astype(jnp.float32)[:, None], self.memory[step, player]), axis=1)
        moves = self.legal[step, player][:, None]
        bits = jnp.arange(4)[None, :, None, None]
        moves = jnp.tile(((moves >> bits) & 1).astype(jnp.bool_), (1, 2, 1, 1)).reshape(len(ids), 8 * side**2)
        legal = jnp.concatenate((moves, jnp.ones((len(ids), 1), jnp.bool_)), axis=1)
        times = step[:, None] - jnp.arange(temporal)
        timeline = self.timeline[times + temporal - 1, player[:, None]]
        missing = times < self.starts[step, player][:, None]
        timeline = jnp.where(missing[:, :, None], 0, timeline)
        timeline = timeline.at[:, :, 0].set(jnp.where(missing, -1, timeline[:, :, 0]))
        return Inputs(board, scalars, valid, memory, self.inside[step, player], legal, timeline)


@dataclass
class Progress:
    iteration: int = 0
    samples: int = 0
    evaluations: int = 0
    stage: int = 0
    selected: str = "ema"


def write(path, header, values):
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write((json.dumps(header) + "\n").encode("utf-8"))
            eqx.tree_serialise_leaves(stream, values)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def metadata(config):
    return {"config": asdict(config), "contract": CONTRACT, "inputs": Inputs._fields}


def save(path, model, ema, optimizer_state, key, config, progress):
    write(path, dict(metadata(config), progress=asdict(progress), selected=progress.selected, resumable=True),
          (model, ema, optimizer_state, key))


def export(path, model, config):
    write(path, dict(metadata(config), selected="model", resumable=False), model)


def load(path, backend, weights="selected"):
    if weights not in ("selected", "model", "ema"):
        raise ValueError("Select selected, model or ema weights")
    if path.suffix != ".eqx":
        raise ValueError("Expected a native .eqx checkpoint or policy")
    with jax.default_device(backend), path.open("rb") as stream:
        header = json.loads(stream.readline())
        settings = header["config"]
        settings["distance"] = tuple(settings["distance"])
        settings["curriculum"] = tuple(tuple(item) for item in settings["curriculum"])
        config = Config(**settings).validate()
        if header["contract"] != json.loads(json.dumps(CONTRACT)) or tuple(header["inputs"]) != Inputs._fields:
            raise ValueError("Checkpoint input contract does not match GenFormer")
        template = GenFormer(config, jax.random.PRNGKey(0))
        if header["resumable"]:
            from ppo import optimizer
            state = optimizer(config).init(eqx.filter(template, eqx.is_inexact_array))
            model, ema, state, key = eqx.tree_deserialise_leaves(
                stream, (template, template, state, jax.random.PRNGKey(0)))
            header.update(model=model, ema=ema, optimizer=state, random=key)
            selected = header["selected"] if weights == "selected" else weights
            model = header[selected]
        else:
            if weights == "ema":
                raise ValueError("Exported policies contain only their selected network")
            model = eqx.tree_deserialise_leaves(stream, template)
            header["model"] = model
    return header, config, model
