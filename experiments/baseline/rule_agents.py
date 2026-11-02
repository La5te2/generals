"""Rule-agent opponents for the frozen evaluator.

`evaluate.Evaluator.matches` requires the opponent to expose `config.history` and to
be callable as `(Inputs, topology) -> object with .logits`. The vendored agents in
the pinned generals-bots dependency expect a `generals.core.observation.Observation`, so this module
rebuilds one from the 1014 legal Inputs and converts the returned command into the
1014 flat action index.

Verified 2026-11-01: rebuilt observations match `views()` exactly for armies, owned,
mountain, castle, fog and obstacle masks; agents emit legal indices once the general
accumulates army.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from generals.agents import ExpanderAgent, HunterAgent, RandomAgent  # noqa: E402
from generals.core.observation import Observation  # noqa: E402

from features import ARMY, CITY, ENEMY, FOG, GENERAL, MOUNTAIN, NEUTRAL, OBSTACLE, OWN  # noqa: E402
from model import Config  # noqa: E402


class Choice(NamedTuple):
    """Minimal stand-in for the network output the evaluator reads."""

    logits: jax.Array


def observation_from_inputs(inputs, side):
    """Rebuild the vendored environment's Observation from the 1014 legal Inputs."""
    cells = inputs.cells
    army = jnp.rint(jnp.expm1(cells[:, ARMY] * jnp.log(256.0))).astype(jnp.int32)
    owned = cells[:, OWN] > 0
    opponent = cells[:, ENEMY] > 0
    return Observation(
        armies=army.reshape(side, side),
        generals=(cells[:, GENERAL] > 0).reshape(side, side),
        castles=(cells[:, CITY] > 0).reshape(side, side),
        mountains=(cells[:, MOUNTAIN] > 0).reshape(side, side),
        neutral_cells=(cells[:, NEUTRAL] > 0).reshape(side, side),
        owned_cells=owned.reshape(side, side),
        opponent_cells=opponent.reshape(side, side),
        fog_cells=(cells[:, FOG] > 0).reshape(side, side),
        structures_in_fog=(cells[:, OBSTACLE] > 0).reshape(side, side),
        owned_land_count=owned.sum(),
        owned_army_count=jnp.where(owned, army, 0).sum(),
        opponent_land_count=opponent.sum(),
        opponent_army_count=jnp.where(opponent, army, 0).sum(),
        timestep=jnp.rint(inputs.stats[-1, 0] * 512).astype(jnp.int32),
    )


class Rule:
    """Callable opponent carrying the `config.history` attribute the evaluator checks."""

    def __init__(self, agent, side, history=8):
        self.agent = agent
        self.side = side
        self.area = side * side
        self.config = Config(history=history)
        self.illegal = 0

    def __call__(self, inputs, topology):
        del topology
        observation = observation_from_inputs(inputs, self.side)
        tick = jnp.rint(inputs.stats[-1, 0] * 512).astype(jnp.int32)
        # The evaluator exposes no game index, so derive a per-lane stream from content.
        salt = (
            jnp.rint(inputs.stats[-1, 3] * 1e6).astype(jnp.int32) * 31
            + jnp.rint(inputs.stats[-1, 4] * 1e6).astype(jnp.int32) * 17
            + jnp.rint(jnp.sum(inputs.board)).astype(jnp.int32)
        )
        key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(7), tick), salt)
        command = self.agent.act(observation, key)
        index = jnp.where(
            command[0] == 1,
            self.area * 8,
            (command[1] * self.side + command[2]) * 8 + command[3] * 2 + command[4],
        )
        legal = inputs.legal[jnp.clip(index, 0, self.area * 8)]
        index = jnp.where(legal, index, self.area * 8)
        return Choice(jnp.full(self.area * 8 + 1, -1e9).at[index].set(0.0))


def build(name, side, history=8):
    agents = {
        "random": RandomAgent,
        "expander": ExpanderAgent,
        "hunter": HunterAgent,
    }
    if name not in agents:
        raise ValueError(f"unknown rule agent: {name}")
    return Rule(agents[name](), side, history)
