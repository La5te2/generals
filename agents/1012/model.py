"""A shared GNN/attention actor-critic with learned temporal and spatial relations.

Transport messages follow known non-mountain edges; sight messages follow eight
neighbors. Global attention is unrestricted across real cells. Auxiliary heads
learn hidden-state marginals from the same legal representation, not an oracle.
"""

from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from features import GLOBAL, SPATIAL, TEMPORAL


@dataclass(frozen=True)
class Config:
    width: int = 128
    depth: int = 6
    heads: int = 4
    history: int = 8

    def __post_init__(self):
        if (
            min(self.width, self.depth, self.heads, self.history) < 1
            or self.width % self.heads
        ):
            raise ValueError(
                "positive dimensions and width divisible by heads required"
            )


def linear(layer, x):
    return x @ layer.weight.T + (layer.bias if layer.bias is not None else 0.0)


def norm(layer, x):
    return jax.vmap(layer)(x)


class Temporal(eqx.Module):
    projection: eqx.nn.Linear
    positions: jax.Array
    query: jax.Array

    def __init__(self, features, config, key):
        a, b, c = jax.random.split(key, 3)
        self.projection = eqx.nn.Linear(features, config.width, key=a)
        self.positions = jax.random.normal(b, (config.history, config.width)) * 0.02
        self.query = jax.random.normal(c, (config.width,)) / config.width**0.5

    def __call__(self, x, valid):
        h = jax.nn.gelu(linear(self.projection, x) + self.positions)
        weights = jax.nn.softmax(jnp.where(valid, h @ self.query, -1e9), -1)
        return jnp.sum(weights[..., None] * h, -2)


class Block(eqx.Module):
    message: tuple
    gate: tuple
    merge: eqx.nn.Linear
    qkv: eqx.nn.Linear | None
    relation: eqx.nn.Linear | None
    attention_out: eqx.nn.Linear | None
    feed_in: eqx.nn.Linear
    feed_out: eqx.nn.Linear
    norms: tuple
    heads: int = eqx.field(static=True)

    def __init__(self, config, attention, key):
        keys = iter(jax.random.split(key, 12))
        d = config.width
        self.message = tuple(
            eqx.nn.Linear(3 * d + 2, d, key=next(keys)) for _ in range(2)
        )
        self.gate = tuple(eqx.nn.Linear(3 * d + 2, d, key=next(keys)) for _ in range(2))
        self.merge = eqx.nn.Linear(2 * d, d, key=next(keys))
        self.qkv = eqx.nn.Linear(d, 3 * d, key=next(keys)) if attention else None
        self.relation = (
            eqx.nn.Linear(3, config.heads, use_bias=False, key=next(keys))
            if attention
            else None
        )
        self.attention_out = eqx.nn.Linear(d, d, key=next(keys)) if attention else None
        self.feed_in = eqx.nn.Linear(d, 4 * d, key=next(keys))
        self.feed_out = eqx.nn.Linear(4 * d, d, key=next(keys))
        self.norms = tuple(eqx.nn.LayerNorm(d) for _ in range(3))
        self.heads = config.heads

    def __call__(self, h, inputs, graph):
        z = norm(self.norms[0], h)
        outputs = []
        for relation, count in enumerate((4, 8)):
            target = z[graph.neighbors[:, :count]]
            source = jnp.broadcast_to(z[:, None], target.shape)
            offsets = jnp.broadcast_to(
                graph.offsets[None, :count], (*target.shape[:-1], 2)
            )
            edge = jnp.concatenate((source, target, source * target, offsets), -1)
            message = jax.nn.gelu(linear(self.message[relation], edge))
            gate = jax.nn.sigmoid(linear(self.gate[relation], edge))
            valid = (
                inputs.movement
                if relation == 0
                else graph.valid & inputs.board[:, None] & inputs.board[graph.neighbors]
            )
            outputs.append(jnp.sum(message * gate * valid[..., None], 1) / count)
        h = h + linear(self.merge, jnp.concatenate(outputs, -1))
        if self.qkv is not None:
            z = norm(self.norms[1], h)
            n, d = z.shape
            q, k, v = jnp.moveaxis(
                linear(self.qkv, z).reshape(n, 3, self.heads, d // self.heads), 1, 0
            )
            delta = graph.positions[:, None] - graph.positions[None, :]
            delta = jnp.sign(delta) * jnp.log1p(jnp.abs(delta)) / 4.0
            geometry = jnp.concatenate(
                (delta, jnp.abs(delta).sum(-1, keepdims=True)), -1
            )
            bias = linear(self.relation, geometry).transpose(2, 0, 1)
            # Content-dependent q/k scores retain all cross-board interactions.
            score = jnp.einsum("ihd,jhd->hij", q, k) / (d // self.heads) ** 0.5 + bias
            weights = jax.nn.softmax(
                jnp.where(inputs.board[None, None, :], score, -1e9), -1
            )
            h = h + linear(
                self.attention_out, jnp.einsum("hij,jhd->ihd", weights, v).reshape(n, d)
            )
        h = h + linear(
            self.feed_out, jax.nn.gelu(linear(self.feed_in, norm(self.norms[2], h)))
        )
        return h * inputs.board[:, None]


class Output(NamedTuple):
    logits: jax.Array
    value: jax.Array
    general: jax.Array
    owner: jax.Array
    army: jax.Array


class Network(eqx.Module):
    spatial: eqx.nn.Linear
    temporal: Temporal
    global_history: Temporal
    input_merge: eqx.nn.Linear
    blocks: tuple
    pool: eqx.nn.Linear
    action_hidden: eqx.nn.Linear
    action_out: eqx.nn.Linear
    pass_head: eqx.nn.Linear
    value_head: eqx.nn.Linear
    general_head: eqx.nn.Linear
    owner_head: eqx.nn.Linear
    army_head: eqx.nn.Linear
    final_norm: eqx.nn.LayerNorm
    config: Config = eqx.field(static=True)

    def __init__(self, config=None, *, key):
        config = config or Config()
        self.config = config
        keys = iter(jax.random.split(key, 14 + config.depth))
        d = config.width
        self.spatial = eqx.nn.Linear(SPATIAL, d, key=next(keys))
        self.temporal = Temporal(TEMPORAL, config, next(keys))
        self.global_history = Temporal(GLOBAL, config, next(keys))
        self.input_merge = eqx.nn.Linear(3 * d, d, key=next(keys))
        self.blocks = tuple(
            Block(config, i % 2 == 1, next(keys)) for i in range(config.depth)
        )
        self.pool = eqx.nn.Linear(d, 1, key=next(keys))
        self.action_hidden = eqx.nn.Linear(4 * d + 2, d, key=next(keys))
        self.action_out = eqx.nn.Linear(d, 2, key=next(keys))
        self.pass_head = eqx.nn.Linear(d, 1, key=next(keys))
        self.value_head = eqx.nn.Linear(d, 1, key=next(keys))
        self.general_head = eqx.nn.Linear(d, 1, key=next(keys))
        self.owner_head = eqx.nn.Linear(d, 3, key=next(keys))
        self.army_head = eqx.nn.Linear(d, 1, key=next(keys))
        self.final_norm = eqx.nn.LayerNorm(d)

    def __call__(self, inputs, graph):
        history = self.temporal(inputs.recent.transpose(1, 0, 2), inputs.time_valid)
        public = self.global_history(inputs.stats, inputs.time_valid)
        h = linear(
            self.input_merge,
            jnp.concatenate(
                (
                    linear(self.spatial, inputs.cells),
                    history,
                    jnp.broadcast_to(public, history.shape),
                ),
                -1,
            ),
        )
        h = h * inputs.board[:, None]
        for block in self.blocks:
            h = block(h, inputs, graph)
        h = norm(self.final_norm, h)
        weight = jax.nn.softmax(
            jnp.where(inputs.board, linear(self.pool, h)[:, 0], -1e9)
        )
        context = jnp.sum(h * weight[:, None], 0)
        target = h[graph.neighbors[:, :4]]
        source = jnp.broadcast_to(h[:, None], target.shape)
        global_cell = jnp.broadcast_to(context, target.shape)
        offsets = jnp.broadcast_to(graph.offsets[None, :4], (*target.shape[:-1], 2))
        edge = jnp.concatenate(
            (source, target, source * target, global_cell, offsets), -1
        )
        moves = linear(
            self.action_out, jax.nn.gelu(linear(self.action_hidden, edge))
        ).reshape(-1)
        logits = jnp.concatenate((moves, self.pass_head(context)))
        return Output(
            jnp.where(inputs.legal, logits, -1e9),
            self.value_head(context)[0],
            jnp.where(inputs.board, linear(self.general_head, h)[:, 0], -1e9),
            linear(self.owner_head, h),
            linear(self.army_head, h)[:, 0],
        )
