"""Realized economic trajectories used as action-conditioned auxiliary labels."""

import jax.numpy as jnp
import numpy as np
import optax
from rewards import material

HORIZONS = (25, 50, 200)
CHANNELS = 6


def transition(before, after, terminal, seat):
    """Production, army change and land change, each ordered learner then opponent.

    Settlement uses post-move ownership. The decisive transition contributes zero
    to all six channels so confiscated territory and armies stay out of the target.
    Completed trajectories then have an absorbing zero continuation.
    """
    old, new = material(before), material(after)
    income = (after.time % 2 == 0) * new[:, 2]
    income = income + (after.time % 50 == 0) * new[:, 0]
    players = jnp.array([seat, 1 - seat])
    values = jnp.concatenate(
        (income[players], (new - old)[players, 1], (new - old)[players, 0])
    )
    return jnp.where(terminal, 0.0, values)


def targets(flows, terminal, truncated, horizons=HORIZONS):
    """Host-side finite-window sums with terminal absorption and censoring masks.

    A timeout supplies its observed transition, then ends the available future.
    Rollout boundaries censor longer windows. Real termination completes a window
    through absorbing zeros. Prefix sums share work across prediction horizons.
    """
    steps, lanes = terminal.shape
    rows = np.arange(steps)[:, None]
    columns = np.arange(lanes)[None, :]
    ended = terminal | truncated
    boundary = np.minimum.accumulate(np.where(ended, rows, steps)[::-1], axis=0)[::-1]
    terminal_boundary = np.concatenate((terminal, np.zeros((1, lanes), bool)), 0)
    reaches_terminal = terminal_boundary[boundary, columns]
    prefix = np.concatenate(
        (np.zeros((1, lanes, CHANNELS)), np.cumsum(flows, axis=0, dtype=np.float64)), 0
    )
    output = np.zeros((*terminal.shape, len(horizons), CHANNELS), np.float32)
    valid = np.zeros((*terminal.shape, len(horizons)), bool)
    for index, length in enumerate(horizons):
        stop = np.minimum(np.minimum(rows + length, boundary + 1), steps)
        known = (rows + length <= steps) & (rows + length <= boundary + 1)
        known |= reaches_terminal & (boundary < rows + length)
        total = prefix[stop, columns] - prefix[:-1]
        encoded = np.sign(total) * np.log1p(np.abs(total)) / np.log(256.0)
        output[..., index, :] = np.where(known[..., None], encoded, 0.0)
        valid[..., index] = known
    return output, valid


def prediction_loss(predicted, target, valid):
    """One equally weighted scalar per sample, averaging its observed horizons."""
    error = optax.huber_loss(predicted, target, delta=1.0).mean(-1)
    return (error * valid).sum(-1) / jnp.maximum(valid.sum(-1), 1)
