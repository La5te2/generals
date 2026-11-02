"""Antisymmetric material potentials and terminal-correct potential shaping.

Production includes owned cities and generals, avoiding the zero-city ratio.
The simulator supplies production counts for training rewards only. Public
production differences constrain inference but hidden neutral attrition can
prevent exact identification (literature/mathematics.md, section 2).
"""

import jax.numpy as jnp


def material(state):
    """Return land, army and producing-cell counts for both players."""
    land = state.ownership.sum(axis=(-2, -1))
    army = (state.ownership * state.armies).sum(axis=(-2, -1))
    production = (state.ownership * (state.castles | state.generals)).sum(axis=(-2, -1))
    return jnp.stack((land, army, production), -1).astype(jnp.float32)


def potential(state, maximum=4.0):
    counts = jnp.maximum(material(state), 1.0)
    ratio = jnp.log(counts[0] / counts[1]) / jnp.log(maximum)
    value = jnp.dot(jnp.clip(ratio, -1, 1), jnp.array([0.3, 0.3, 0.4]))
    return jnp.array([value, -value])


def shape_reward(reward, before, after, terminal, gamma=1.0, scale=1.0):
    """Only actual termination has zero potential; truncations keep continuation."""
    return reward + scale * (gamma * jnp.where(terminal, 0.0, after) - before)
