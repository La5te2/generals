"""Observation-only fitted potential and terminal-correct reward components."""

import jax.numpy as jnp
from pathlib import Path
from potential import empty, evaluate, load_parameters, observe, update

WEIGHTS, POTENTIAL_SHA256 = load_parameters(Path(__file__).with_name("potential.json"))


def material(state):
    """Return land, army and producing-cell counts for both players."""
    land = state.ownership.sum(axis=(-2, -1))
    army = (state.ownership * state.armies).sum(axis=(-2, -1))
    production = (state.ownership * (state.castles | state.generals)).sum(axis=(-2, -1))
    return jnp.stack((land, army, production), -1).astype(jnp.float32)


def potential(state, player, memory, topology):
    """Consume the next learner observation and cache its fitted potential."""
    height, width = state.armies.shape
    vector, memory = update(observe(state, player, topology), memory, topology, height, width)
    return memory._replace(value=evaluate(vector, jnp.asarray(WEIGHTS)))


def shape_reward(reward, before, after, terminal, gamma=1.0, scale=1.0):
    """Only actual termination has zero potential; truncations keep continuation."""
    return reward + scale * (gamma * jnp.where(terminal, 0.0, after) - before)


def reward_parts(reward, before, after, terminal, gamma=1.0, scale=1.0):
    """Disjoint outcome, continuing-potential, and terminal-reset contributions."""
    continuation = scale * jnp.where(terminal, 0.0, gamma * after - before)
    terminal_reset = -scale * jnp.where(terminal, before, 0.0)
    return jnp.stack((reward, continuation, terminal_reset), -1)
