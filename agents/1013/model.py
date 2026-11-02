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

COMPUTE_DTYPE = jnp.float32


def precision(bf16=False):
    """Select arithmetic before tracing; parameters and normalization stay float32."""
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = jnp.bfloat16 if bf16 else jnp.float32


def product(x, weight):
    return (x.astype(COMPUTE_DTYPE) @ weight.T.astype(COMPUTE_DTYPE)).astype(
        jnp.float32
    )


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
    return product(x, layer.weight) + (layer.bias if layer.bias is not None else 0.0)


def edge_linear(layer, source, neighbors, offsets, context=None):
    """Factor shared source/target projections before gathering, preserving weights."""
    return edge_weights(layer.weight, layer.bias, source, neighbors, offsets, context)


def edge_weights(weight, bias, source, neighbors, offsets, context=None):
    d = source.shape[-1]
    result = product(source, weight[:, :d])[:, None]
    result = result + product(source, weight[:, d : 2 * d])[neighbors]
    result = result + product(
        source[:, None] * source[neighbors], weight[:, 2 * d : 3 * d]
    )
    if context is not None:
        result = result + product(context, weight[:, 3 * d : 4 * d])
    return result + product(offsets, weight[:, -2:]) + bias


def production_evidence(inputs):
    """Public production residual, with an explicit consecutive-growth validity bit.

    q=P_own+land_boundary*(L_own-L_enemy)-delta(A_own-A_enemy)
     =P_enemy+N_own-N_enemy. Neutral attrition is deliberately NOT assumed zero.
    The third entry is a visible lower bound on enemy producers. General trades
    break this identity; the network receives the residual as evidence, not truth.
    """
    now = inputs.stats[-1]
    before = inputs.stats[-2] if inputs.stats.shape[0] > 1 else now
    tick = jnp.rint(now[0] * 512).astype(jnp.int32)
    prior = jnp.rint(before[0] * 512).astype(jnp.int32)
    valid = (inputs.time_valid.sum() >= 2) & (tick == prior + 1) & (tick % 2 == 0)
    armies = jnp.expm1(jnp.stack((before[3:5], now[3:5])) * jnp.log(256.0))
    difference = (armies[1, 0] - armies[1, 1]) - (armies[0, 0] - armies[0, 1])
    producer = (inputs.cells[:, 5] + inputs.cells[:, 6]) > 0
    own = jnp.sum(producer * inputs.cells[:, 1])
    lower = jnp.sum(producer * inputs.cells[:, 2])
    area = jnp.maximum(inputs.board.sum(), 1)
    residual = own + (tick % 50 == 0) * (now[5] - now[6]) * area - difference
    return jnp.array([jnp.where(valid, residual / area, 0), valid, lower / area])


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
            neighbors, offsets = graph.neighbors[:, :count], graph.offsets[:count]
            weight = jnp.concatenate(
                (self.message[relation].weight, self.gate[relation].weight)
            )
            bias = jnp.concatenate(
                (self.message[relation].bias, self.gate[relation].bias)
            )
            message, gate = jnp.split(
                edge_weights(weight, bias, z, neighbors, offsets), 2, -1
            )
            message, gate = jax.nn.gelu(message), jax.nn.sigmoid(gate)
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
            score = jnp.einsum(
                "ihd,jhd->hij", q.astype(COMPUTE_DTYPE), k.astype(COMPUTE_DTYPE)
            ).astype(jnp.float32)
            score = score / (d // self.heads) ** 0.5 + bias
            weights = jax.nn.softmax(
                jnp.where(inputs.board[None, None, :], score, -1e9), -1
            )
            attended = jnp.einsum(
                "hij,jhd->ihd", weights.astype(COMPUTE_DTYPE), v.astype(COMPUTE_DTYPE)
            ).astype(jnp.float32)
            h = h + linear(self.attention_out, attended.reshape(n, d))
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
    enemy_action: jax.Array
    enemy_view: jax.Array


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
    belief_hidden: eqx.nn.Linear
    belief_out: eqx.nn.Linear
    enemy_move: eqx.nn.Linear
    enemy_pass: eqx.nn.Linear
    enemy_view: eqx.nn.Linear
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
        extra = jax.random.split(jax.random.fold_in(key, 1013), 5)
        self.belief_hidden = eqx.nn.Linear(d + 3, d, key=extra[0])
        self.belief_out = eqx.nn.Linear(d, d, key=extra[1])
        self.belief_out = eqx.tree_at(
            lambda x: (x.weight, x.bias),
            self.belief_out,
            (jnp.zeros_like(self.belief_out.weight), jnp.zeros(d)),
        )
        self.enemy_move = eqx.nn.Linear(3 * d + 2, 2, key=extra[2])
        self.enemy_pass = eqx.nn.Linear(d, 1, key=extra[3])
        self.enemy_view = eqx.nn.Linear(d, 10, key=extra[4])

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
        evidence = jnp.broadcast_to(production_evidence(inputs), (len(h), 3))
        belief = jax.nn.gelu(
            linear(self.belief_hidden, jnp.concatenate((h, evidence), -1))
        )
        h = h + linear(self.belief_out, belief)
        weight = jax.nn.softmax(
            jnp.where(inputs.board, linear(self.pool, h)[:, 0], -1e9)
        )
        context = jnp.sum(h * weight[:, None], 0)
        moves = linear(
            self.action_out,
            jax.nn.gelu(
                edge_linear(
                    self.action_hidden,
                    h,
                    graph.neighbors[:, :4],
                    graph.offsets[:4],
                    context,
                )
            ),
        ).reshape(-1)
        logits = jnp.concatenate((moves, self.pass_head(context)))
        # This decoder cannot use the opponent's true action mask or observation.
        enemy_moves = edge_linear(
            self.enemy_move, belief, graph.neighbors[:, :4], graph.offsets[:4]
        )
        possible = (
            graph.valid[:, :4]
            & inputs.board[:, None]
            & inputs.board[graph.neighbors[:, :4]]
        )
        enemy_moves = jnp.where(possible[..., None], enemy_moves, -1e9).reshape(-1)
        enemy_context = jnp.sum(belief * weight[:, None], 0)
        return Output(
            jnp.where(inputs.legal, logits, -1e9),
            self.value_head(context)[0],
            jnp.where(inputs.board, linear(self.general_head, h)[:, 0], -1e9),
            linear(self.owner_head, h),
            linear(self.army_head, h)[:, 0],
            jnp.concatenate((enemy_moves, self.enemy_pass(enemy_context))),
            linear(self.enemy_view, belief),
        )
