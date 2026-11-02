"""Training-only centralized action values and Expected-SARSA trace targets.

The actor never receives Oracle. A deterministic frozen opponent's command is
part of the transition context; no sampled learner action enters the encoder.
All action values share one small graph trunk and one source/target decoder.
"""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from features import SPATIAL
from memory import read as read_history
from model import Block, Config, HistoryEncoder, edge_linear, linear, norm


class Oracle(NamedTuple):
    flags: object
    army: object
    enemy_action: object


class Critic(eqx.Module):
    spatial: eqx.nn.Linear
    history: HistoryEncoder
    blocks: tuple
    normalization: eqx.nn.LayerNorm
    pool: eqx.nn.Linear
    action_hidden: eqx.nn.Linear
    action_out: eqx.nn.Linear
    pass_head: eqx.nn.Linear

    def __init__(self, period, key, recent=32):
        keys = iter(jax.random.split(key, 11))
        d = 64
        self.spatial = eqx.nn.Linear(SPATIAL + 16, d, key=next(keys))
        self.history = HistoryEncoder(d, period, recent, next(keys), small=16)
        config = Config(width=d, depth=3, heads=4, period=period, recent=recent)
        self.blocks = tuple(Block(config, i == 1, next(keys)) for i in range(3))
        self.normalization = eqx.nn.LayerNorm(d)
        self.pool = eqx.nn.Linear(d, 1, key=next(keys))
        self.action_hidden = eqx.nn.Linear(4 * d + 2, d, key=next(keys))
        self.action_out = eqx.nn.Linear(d, 2, key=next(keys))
        self.pass_head = eqx.nn.Linear(d, 1, key=next(keys))
        self.action_out = eqx.tree_at(
            lambda x: (x.weight, x.bias),
            self.action_out,
            (jnp.zeros_like(self.action_out.weight), jnp.zeros(2)),
        )
        self.pass_head = eqx.tree_at(
            lambda x: (x.weight, x.bias),
            self.pass_head,
            (jnp.zeros_like(self.pass_head.weight), jnp.zeros(1)),
        )

    def __call__(self, inputs, oracle, graph, checkpoint="full"):
        n = len(inputs.board)
        facts = ((oracle.flags[:, None] >> jnp.arange(6)) & 1).astype(jnp.float32)
        command = oracle.enemy_action
        source = jnp.minimum(command // 8, n - 1)
        target = graph.neighbors[source, (command % 8) // 2]
        moving = command < 8 * n
        marks = jnp.stack(
            (
                (jnp.arange(n) == source) & moving,
                (jnp.arange(n) == target) & moving,
                (jnp.arange(n) == source) & moving & (command % 2 == 1),
            ),
            -1,
        )
        reading = read_history(inputs.archive, graph, self.history.period)
        x = jnp.concatenate((inputs.cells, facts, oracle.army[:, None], marks, reading.immediate), -1)
        history_layer = eqx.filter_checkpoint(self.history) if checkpoint == "full" else self.history
        h = jax.nn.gelu(linear(self.spatial, x) + history_layer(reading, graph, inputs.board)) * inputs.board[:, None]
        # The centralized trunk can use actual transport edges. Policy masks stay legal.
        mountain = facts[:, 3] > 0
        movement = (
            graph.valid[:, :4]
            & inputs.board[:, None]
            & inputs.board[graph.neighbors[:, :4]]
            & ~mountain[:, None]
            & ~mountain[graph.neighbors[:, :4]]
        )
        central = inputs._replace(movement=movement)
        for block in self.blocks:
            layer = eqx.filter_checkpoint(block) if checkpoint != "none" else block
            h = layer(h, central, graph)
        h = norm(self.normalization, h)
        weights = jax.nn.softmax(
            jnp.where(inputs.board, linear(self.pool, h)[:, 0], -1e9)
        )
        context = jnp.sum(h * weights[:, None], 0)
        hidden = jax.nn.gelu(
            edge_linear(
                self.action_hidden,
                h,
                graph.neighbors[:, :4],
                graph.offsets[:4],
                context,
            )
        )
        moves = linear(self.action_out, hidden).reshape(-1)
        return jnp.concatenate((moves, self.pass_head(context)))


def expectation(q, logits):
    return jnp.sum(jax.nn.softmax(logits, -1) * q, -1)


def trace_targets(
    reward,
    value,
    chosen_q,
    terminal,
    truncated,
    bootstrap,
    following,
    gamma=1.0,
    lam=0.95,
):
    """Stop traces on reset; bootstrap time limits from the pre-reset observation."""
    advantage = np.empty_like(reward, dtype=np.float32)
    target = np.empty_like(reward, dtype=np.float32)
    accumulated = np.zeros(reward.shape[1:], np.float32)
    for t in range(len(reward) - 1, -1, -1):
        next_value = np.where(truncated[t], bootstrap[t], following)
        residual = reward[t] + gamma * ~terminal[t] * next_value - chosen_q[t]
        accumulated = (
            residual + gamma * lam * ~(terminal[t] | truncated[t]) * accumulated
        )
        target[t] = chosen_q[t] + accumulated
        advantage[t] = target[t] - value[t]
        following = value[t]
    return advantage, target


def make_update(tx, microbatch, checkpoint="full"):
    """Independent microbatched critic regression; actor weights stay untouched."""

    @eqx.filter_jit(donate="all-except-first")
    def update(batch_and_graph, critic, state):
        batch, graph, timeline = batch_and_graph
        size = len(batch.actions)
        chunks = jax.tree.map(
            lambda x: x.reshape((size // microbatch, microbatch, *x.shape[1:])), batch
        )
        zeros = jax.tree.map(jnp.zeros_like, eqx.filter(critic, eqx.is_inexact_array))

        def loss_fn(model, packed):
            data = packed.unpack(timeline)
            q = jax.vmap(lambda x, truth: model(x, truth, graph, checkpoint))(
                data.inputs, data.oracle
            )
            selected = jnp.take_along_axis(q, data.actions[:, None], -1)[:, 0]
            return 0.5 * jnp.mean((selected - data.returns) ** 2)

        def accumulate(carry, part):
            loss, gradients = carry
            value, grad = eqx.filter_value_and_grad(loss_fn)(critic, part)
            return (
                loss + value * (microbatch / size),
                jax.tree.map(lambda a, b: a + b * (microbatch / size), gradients, grad),
            ), None

        (loss, grads), _ = jax.lax.scan(accumulate, (jnp.array(0.0), zeros), chunks)
        magnitude = optax.tree.norm(grads)
        updates, state = tx.update(
            grads, state, eqx.filter(critic, eqx.is_inexact_array)
        )
        return eqx.apply_updates(critic, updates), state, loss, magnitude

    return update
