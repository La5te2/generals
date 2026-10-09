"""Compare frozen policies on held-out maps, playing each map once from each color."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from env import Arena, select_device
from features import Encoder
from ppo import predict
from storage import load


class Evaluator:
    def __init__(self, config, device, games=None, seed=None):
        self.games = config.games if games is None else games
        self.seed = config.evaluation_seed if seed is None else seed
        if self.games < 2 or self.games % 2:
            raise ValueError("Evaluation requires an even game count for color pairs")
        shapes = (config.maximum - config.minimum + 1)**2
        lanes = max(2, min(config.environments, 32, self.games) // 2 * 2)
        self.config = replace(config, environments=lanes, pool=max(shapes, self.games // 2)).validate()
        self.arena = Arena(self.config, device, self.seed)
        self.infer = eqx.filter_jit(predict)
        self.encode = jax.jit(lambda encoder, view, reset: encoder.update(view, reset))

    def __enter__(self):
        print(json.dumps({"phase": "evaluation_maps", "distance": self.config.distance}), flush=True)
        self.arena.prepare()
        return self

    def __exit__(self, *exception):
        self.arena.close()

    def matches(self, model, opponent=None):
        arena, config = self.arena, self.config
        kind = "random" if opponent is None else "baseline" if isinstance(opponent, str) and opponent == "baseline" else "network"
        if kind == "baseline":
            from baseline import Baseline
            opponent = None
        wins = losses = draws = cutoffs = 0
        begun = time.perf_counter()
        bf16 = config.bf16 and arena.backend.platform == "gpu"
        with jax.default_device(arena.backend):
            key = jax.random.PRNGKey(self.seed)
            for offset in range(0, self.games // 2, arena.count // 2):
                pairs = min(arena.count // 2, self.games // 2 - offset)
                indices = [offset + pair for pair in range(pairs) for seat in range(2)]
                arena.start(indices + [0] * (arena.count - len(indices)))
                active = jnp.arange(arena.count) < 2 * pairs
                seats = jnp.arange(arena.count) % 2
                bases = 2 * jnp.arange(arena.count)
                ours, theirs = bases + seats, bases + 1 - seats
                count = 2 * arena.count
                encoder = Encoder.empty(count, config.side, config.history, config.temporal)
                reset = jnp.ones(count, jnp.bool_)
                benchmark = Baseline(count, config.side, arena.backend) if kind == "baseline" else None
                ticks = 0
                print(json.dumps({"phase": "evaluation", "event": "start", "opponent": kind,
                                  "games": 2 * pairs, "horizon": config.horizon}), flush=True)
                while bool(active.any()):
                    view = arena.read()
                    encoder, inputs = self.encode(encoder, view, reset)
                    moves = jnp.full(count, 8 * config.side**2, jnp.int32)
                    ours_inputs = jax.tree.map(lambda value, ids=ours: value[ids], inputs)
                    moves = moves.at[ours].set(self.infer(model, ours_inputs, config.microbatch, bf16)[0].argmax(1))
                    if opponent is not None:
                        their_inputs = jax.tree.map(lambda value, ids=theirs: value[ids], inputs)
                        moves = moves.at[theirs].set(self.infer(opponent, their_inputs, config.microbatch, bf16)[0].argmax(1))
                    if benchmark:
                        moves = moves.at[theirs].set(benchmark.act(view, reset)[theirs])
                    elif opponent is None:
                        possible = inputs.legal[theirs, :-1]
                        legal = jnp.concatenate((possible, ~possible.any(1, keepdims=True)), axis=1)
                        key, choice = jax.random.split(key)
                        moves = moves.at[theirs].set(jax.random.categorical(choice, jnp.where(legal, 0., -jnp.inf)))
                    moves = jnp.where(jnp.repeat(active, 2), moves, 8 * config.side**2)
                    encoder = encoder.submitted(moves)
                    if benchmark:
                        benchmark.submitted(moves)
                    reward, terminal, truncated = arena.step(moves)
                    reset = terminal | truncated
                    ended = reset[::2] & active
                    scores = reward[ours]
                    wins += int(((scores > 0) & ended).sum())
                    losses += int(((scores < 0) & ended).sum())
                    cutoffs += int((truncated[ours] & ended).sum())
                    draws += int(((scores == 0) & terminal[ours] & ended).sum())
                    active = active & ~ended
                    ticks += 1
                    if ticks % 256 == 0 or not bool(active.any()):
                        print(json.dumps({"phase": "evaluation", "tick": ticks, "active": int(active.sum()),
                                          "wins": wins, "losses": losses, "truncated": cutoffs,
                                          "seconds": round(time.perf_counter() - begun, 2)}), flush=True)
                    if ticks > config.horizon:
                        raise RuntimeError("Evaluation environment did not end at its configured horizon")
        return {"games": self.games, "wins": wins, "losses": losses, "draws": draws, "truncated": cutoffs,
                "score": (wins - losses) / self.games, "seed": self.seed, "horizon": config.horizon,
                "distance": config.distance, "opponent": kind}


def evaluate(path, opponent_path=None, games=32, seed=200044, device="cpu", horizon=None, weights="selected", distance=None, baseline=False):
    backend = select_device(device)
    config, model = load(path, backend, weights)[1:]
    config = replace(config, horizon=config.horizon if horizon is None else horizon).validate()
    if distance is not None:
        config = replace(config, distance=tuple(distance)).validate()
    opponent = "baseline" if baseline else None
    if opponent_path is not None:
        other_config, opponent = load(opponent_path, backend)[1:]
        if (config.side, config.history, config.temporal) != (other_config.side, other_config.history, other_config.temporal):
            raise ValueError("Evaluation checkpoints must use the same side and history lengths")
    with Evaluator(config, backend, games, seed) as evaluator:
        result = evaluator.matches(model, opponent)
    result["opponent"] = str(opponent_path) if opponent_path else "baseline" if baseline else "random"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    rivals = parser.add_mutually_exclusive_group()
    rivals.add_argument("--opponent", type=Path)
    rivals.add_argument("--baseline", action="store_true")
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--seed", type=int, default=200044)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--weights", choices=("selected", "model", "ema"), default="selected")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--distance", type=int, nargs=2, metavar=("MIN", "MAX"))
    args = parser.parse_args()
    if args.games < 2 or args.games % 2:
        parser.error("games must be positive color pairs")
    print(json.dumps(evaluate(args.checkpoint, args.opponent, args.games, args.seed, args.device,
                              args.horizon, args.weights, args.distance, args.baseline)))


if __name__ == "__main__":
    main()
