"""One fixed random-vs-random trajectory; inspect pre-fallback actions for 200 ticks."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agents/1014"))

from rule_agents import build, observation_from_inputs

import jax
import jax.numpy as jnp
import numpy as np
from environment import Run, encode_players, initial_maps, views
from features import action, empty, graph
from generals.core.game import step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve existing diagnostic results")
    seed, ticks, history = 930001, 200, 8
    config = Run()
    topology = graph(config.side, config.side)
    rule = build("random", config.side, history)
    area = config.side**2
    state = initial_maps(config, jax.random.split(jax.random.PRNGKey(seed), 1))
    memory = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (1, 2, *x.shape)), empty(area, history)
    )
    inputs, memory = encode_players(
        views(state, topology), memory, jnp.full((1, 2), area * 8), topology
    )

    def inspect(inputs):
        # Match Rule.__call__ exactly, but retain the index before fallback.
        tick = jnp.rint(inputs.stats[-1, 0] * 512).astype(jnp.int32)
        salt = (
            jnp.rint(inputs.stats[-1, 3] * 1e6).astype(jnp.int32) * 31
            + jnp.rint(inputs.stats[-1, 4] * 1e6).astype(jnp.int32) * 17
            + jnp.rint(jnp.sum(inputs.board)).astype(jnp.int32)
        )
        key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(7), tick), salt)
        command = rule.agent.act(observation_from_inputs(inputs, config.side), key)
        index = jnp.where(
            command[0] == 1, area * 8,
            (command[1] * config.side + command[2]) * 8 + command[3] * 2 + command[4],
        )
        legal = inputs.legal[jnp.clip(index, 0, area * 8)]
        actual = jnp.argmax(rule(inputs, topology).logits)
        expected = jnp.where(legal, index, area * 8)
        actual_legal = inputs.legal[actual]
        return actual, ~legal, ~actual_legal, actual != expected, command[0] == 0

    def advance(carry, _):
        state, memory, inputs, active = carry
        commands, fallback, illegal, mismatch, attempted = jax.vmap(jax.vmap(inspect))(inputs)
        live = jnp.broadcast_to(active[:, None], commands.shape)
        counts = jnp.stack([
            live.sum(), (fallback & live).sum(), (illegal & live).sum(),
            (mismatch & live).sum(), (attempted & live).sum(),
        ])
        commands = jnp.where(live, commands, area * 8)
        moves = jax.vmap(jax.vmap(lambda a: action(a, config.side, area)))(commands)
        following, info = jax.vmap(lambda s, a: step(s, a, general_trade=True))(state, moves)
        inputs, memory = encode_players(views(following, topology), memory, commands, topology)
        return (following, memory, inputs, active & ~info.is_done), counts

    final, counts = jax.jit(lambda c: jax.lax.scan(advance, c, None, length=ticks))(
        (state, memory, inputs, jnp.ones(1, bool))
    )
    total, fallback, illegal, mismatch, attempted = map(int, np.asarray(counts).sum(0))
    rate = fallback / max(total, 1)
    result = {
        "purpose": "random_self_play_fallback_check",
        "seed": seed, "requested_ticks": ticks, "final_tick": int(final[0].time[0]),
        "side": config.side, "history": history, "active_action_samples": total,
        "attempted_moves": attempted, "fallback_count": fallback, "fallback_rate": rate,
        "post_fallback_illegal_count": illegal, "adapter_action_mismatches": mismatch,
        "threshold": 0.05,
        "passed": rate <= 0.05 and illegal == 0 and mismatch == 0 and total == ticks * 2,
        "note": "Diagnostic only, no strength conclusion. Terminal trajectories fail the full-200-tick check.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit("adapter diagnostic failed; do not start the panel")


if __name__ == "__main__":
    main()
