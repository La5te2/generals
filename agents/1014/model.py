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
import numpy as np
from features import GLOBAL, PUBLIC_HISTORY, SPATIAL, TEMPORAL
from economy import CHANNELS, HORIZONS

COMPUTE_DTYPE = jnp.float32
# Integer bins: zero and small garrisons are exact; the last bin is an open tail.
ARMY_BOUNDARIES = np.concatenate(
    (np.arange(1, 17), np.rint(np.geomspace(17, 65536, 47)))
).astype(np.int32)
ARMY_BINS = len(ARMY_BOUNDARIES) + 1


def army_classes(log_army):
    boundaries = jnp.log1p(jnp.asarray(ARMY_BOUNDARIES, jnp.float32) - 0.5) / jnp.log(
        256.0
    )
    return jnp.searchsorted(boundaries, log_army, side="right")


def history_changes(inputs):
    """Comparable observed changes; growth and exchanges remain in the residual."""
    frames, valid = inputs.recent, inputs.time_valid
    army = jnp.expm1(frames[..., 9] * jnp.log(256.0)) / 256.0
    ticks = jnp.rint(inputs.stats[:, 0] * 512).astype(jnp.int32)
    consecutive = valid[1:] & valid[:-1] & (ticks[1:] == ticks[:-1] + 1)
    overlap = (frames[1:, :, 0] > 0) & (frames[:-1, :, 0] > 0) & consecutive[:, None]
    same_owner = (frames[1:, :, 1:3] > 0) & (frames[:-1, :, 1:3] > 0)
    delta = (army[1:] - army[:-1])[..., None] * same_owner * overlap[..., None]
    delta = (
        jnp.concatenate((jnp.zeros_like(delta[:1]), delta), 0)
        if len(frames) > 1
        else jnp.zeros((*army.shape, 2))
    )
    overlap = jnp.concatenate((jnp.zeros_like(frames[:1, :, 0]), overlap), 0)
    return delta, overlap


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
    depth: int = 8
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


class HistoryRelations(eqx.Module):
    """Mix short spatial histories before compression; attend to ordered score chunks."""

    frame: eqx.nn.Linear
    neighbors: eqx.nn.Linear
    short_out: eqx.nn.Linear
    public_in: eqx.nn.Linear
    public_hidden: eqx.nn.Linear
    positions: jax.Array
    query: eqx.nn.Linear
    key: eqx.nn.Linear
    value: eqx.nn.Linear
    public_out: eqx.nn.Linear
    heads: int = eqx.field(static=True)

    def __init__(self, config, key):
        keys = iter(jax.random.split(key, 10))
        d, small = config.width, 32
        self.frame = eqx.nn.Linear(TEMPORAL + 3, small, key=next(keys))
        self.neighbors = eqx.nn.Linear(3 * small, small, key=next(keys))
        self.short_out = eqx.nn.Linear(config.history * small, d, key=next(keys))
        self.public_in = eqx.nn.Linear(64 * (GLOBAL + 1), d, key=next(keys))
        self.public_hidden = eqx.nn.Linear(d, d, key=next(keys))
        self.positions = jax.random.normal(next(keys), (PUBLIC_HISTORY // 64, d)) * 0.02
        self.query = eqx.nn.Linear(d, d, key=next(keys))
        self.key = eqx.nn.Linear(d, d, key=next(keys))
        self.value = eqx.nn.Linear(d, d, key=next(keys))
        self.public_out = eqx.nn.Linear(d, d, key=next(keys))
        self.heads = config.heads

    def __call__(self, h, inputs, graph):
        frames, valid = inputs.recent, inputs.time_valid
        delta, overlap = history_changes(inputs)
        x = jnp.concatenate((frames, delta, overlap[..., None]), -1)
        z = jax.nn.gelu(linear(self.frame, x))
        edges = graph.valid & inputs.board[:, None] & inputs.board[graph.neighbors]
        # Geometric neighbors exchange frame-local facts before temporal compression.
        adjacent = z[:, graph.neighbors] * edges[None, ..., None]
        local = jnp.concatenate(
            (z, adjacent[:, :, :4].sum(2) / 4, adjacent.sum(2) / 8), -1
        )
        z = jax.nn.gelu(linear(self.neighbors, local)) * valid[:, None, None]
        short = linear(self.short_out, z.transpose(1, 0, 2).reshape(len(h), -1))
        public = jnp.concatenate(
            (
                inputs.public * inputs.public_valid[:, None],
                inputs.public_valid[:, None],
            ),
            -1,
        )
        tokens = jax.nn.gelu(linear(self.public_in, public.reshape(8, -1)))
        tokens = linear(self.public_hidden, tokens) + self.positions
        mask = inputs.public_valid.reshape(8, 64).any(-1)
        d = h.shape[-1]
        q = linear(self.query, h).reshape(len(h), self.heads, -1)
        k = linear(self.key, tokens).reshape(8, self.heads, -1)
        v = linear(self.value, tokens).reshape(8, self.heads, -1)
        scores = jnp.einsum("nhd,thd->hnt", q, k) / (d // self.heads) ** 0.5
        weights = jax.nn.softmax(jnp.where(mask[None, None], scores, -1e9), -1)
        context = jnp.einsum("hnt,thd->nhd", weights, v).reshape(len(h), d)
        return h + short + linear(self.public_out, context)


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
    economy: jax.Array


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
    state_head: eqx.nn.Linear
    final_norm: eqx.nn.LayerNorm
    belief_hidden: eqx.nn.Linear
    belief_out: eqx.nn.Linear
    enemy_move: eqx.nn.Linear
    enemy_pass: eqx.nn.Linear
    config: Config = eqx.field(static=True)
    history_relations: HistoryRelations
    economy_hidden: eqx.nn.Linear
    economy_out: eqx.nn.Linear

    def __init__(self, config=None, *, key):
        config = config or Config()
        self.config = config
        keys = iter(jax.random.split(key, 14 + config.depth))
        d = config.width
        self.spatial = eqx.nn.Linear(SPATIAL, d, key=next(keys))
        self.temporal = Temporal(TEMPORAL, config, next(keys))
        self.global_history = Temporal(GLOBAL, config, next(keys))
        self.history_relations = HistoryRelations(config, jax.random.fold_in(key, 1014))
        self.input_merge = eqx.nn.Linear(3 * d, d, key=next(keys))
        self.blocks = tuple(
            Block(config, i % 2 == 1, next(keys)) for i in range(config.depth)
        )
        self.pool = eqx.nn.Linear(d, 1, key=next(keys))
        self.action_hidden = eqx.nn.Linear(4 * d + 2, d, key=next(keys))
        self.action_out = eqx.nn.Linear(d, 2, key=next(keys))
        self.pass_head = eqx.nn.Linear(d, 1, key=next(keys))
        self.value_head = eqx.nn.Linear(d, 1, key=next(keys))
        self.state_head = eqx.nn.Linear(d, 4 + 3 * ARMY_BINS, key=next(keys))
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
        economy_keys = jax.random.split(jax.random.fold_in(key, 1015), 2)
        self.economy_hidden = eqx.nn.Linear(2 * d + 2, d, key=economy_keys[0])
        self.economy_out = eqx.nn.Linear(
            d, len(HORIZONS) * CHANNELS, key=economy_keys[1]
        )

    def __call__(self, inputs, graph, economy_action=None, checkpoint="full"):
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
        history_layer = (
            eqx.filter_checkpoint(self.history_relations)
            if checkpoint == "full"
            else self.history_relations
        )
        h = history_layer(h, inputs, graph) * inputs.board[:, None]
        for block in self.blocks:
            layer = eqx.filter_checkpoint(block) if checkpoint != "none" else block
            h = layer(h, inputs, graph)
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
        action_features = jax.nn.gelu(
            edge_linear(
                self.action_hidden,
                h,
                graph.neighbors[:, :4],
                graph.offsets[:4],
                context,
            )
        )
        moves = linear(self.action_out, action_features).reshape(-1)
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
        state = linear(self.state_head, h)
        economic = jnp.zeros((len(HORIZONS), CHANNELS))
        if economy_action is not None:
            moving = economy_action < len(h) * 8
            pair = jnp.minimum(economy_action // 2, len(h) * 4 - 1)
            selected = action_features.reshape(-1, h.shape[-1])[pair] * moving
            mode = jax.nn.one_hot(economy_action % 2, 2) * moving
            economic = self.economy_out(
                jax.nn.gelu(
                    linear(
                        self.economy_hidden, jnp.concatenate((context, selected, mode))
                    )
                )
            ).reshape(len(HORIZONS), CHANNELS)
        return Output(
            jnp.where(inputs.legal, logits, -1e9),
            self.value_head(context)[0],
            jnp.where(inputs.board, state[:, 0], -1e9),
            state[:, 1:4],
            state[:, 4:].reshape(len(h), 3, ARMY_BINS),
            jnp.concatenate((enemy_moves, self.enemy_pass(enemy_context))),
            economic,
        )
