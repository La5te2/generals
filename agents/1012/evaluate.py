"""Frozen-policy evaluation on held-out maps with paired starting positions.

Evaluation has its own maps, histories and random streams. It never updates
weights, training rollouts or optimizer state. Timeouts remain unresolved.
"""

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple

import baseline
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from features import (
    CITY,
    ENEMY,
    GENERAL,
    MOUNTAIN,
    OBSTACLE,
    OWN,
    VISIBLE,
    action,
    empty,
    graph,
)
from generals.core.game import step
from ppo import Run, choose, encode_players, initial_maps, views
from train import load, save


def emit(record):
    print(json.dumps(record), flush=True)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def map_hashes(states):
    """Identify initial maps, treating swapped player labels as the same map."""
    leaves = [np.asarray(x) for x in (states.mountains, states.castles, states.armies, states.ownership)]
    return [
        hashlib.sha256(b"".join(x[i].tobytes() for x in leaves[:-1]) +
                       b"".join(sorted(x.tobytes() for x in leaves[-1][i]))).hexdigest()
        for i in range(len(leaves[0]))
    ]


def baseline_inputs(states, observation, side):
    """Adapt only the baseline player's masked view to its tile/owner convention."""
    x = observation.cells
    visible = x[..., VISIBLE] > 0
    terrain = jnp.where(visible, 1, jnp.where(x[..., OBSTACLE] > 0, 5, 0))
    terrain = jnp.where(x[..., MOUNTAIN] > 0, 2, terrain)
    terrain = jnp.where(x[..., CITY] > 0, 3, terrain)
    terrain = jnp.where(x[..., GENERAL] > 0, 4, terrain)
    owner = jnp.where(x[..., OWN] > 0, 1, jnp.where(x[..., ENEMY] > 0, 2, 0))
    army = jnp.where(visible, states.armies.reshape(visible.shape), 0)
    shape = (-1, side, side)
    return (
        terrain.reshape(shape), owner.reshape(shape), army.reshape(shape),
        observation.board.reshape(shape), states.time,
        jnp.zeros(len(visible), bool),
    )


def baseline_action(index, area):
    """Direction-major baseline action -> cell-major policy action; pass unchanged."""
    return jnp.where(index == 8 * area, 8 * area,
                     index % area * 8 + (index // area % 4) * 2 + index // (4 * area))


class Match(NamedTuple):
    state: object
    memory: object
    inputs: object
    baseline: object
    active: jax.Array
    outcome: jax.Array  # -1 loss, 0 draw, 1 win, 2 timeout, 3 unfinished
    ticks: jax.Array
    passes: jax.Array
    optional_passes: jax.Array
    movable: jax.Array
    invalid: jax.Array


def summarize(records):
    """Use independent maps as statistical units; two colors form one pair.

    Hoeffding bounds are conservative, distribution-free fixed-look intervals.
    They do not certify improvement after repeated checkpoint selection.
    Net win rate counts +1/win, -1/loss and 0/unresolved or draw.
    """
    outcomes = {name: sum(r["outcome"] == name for r in records)
                for name in ("win", "loss", "draw", "truncated")}
    pairs = {}
    for record in records:
        pairs.setdefault(record["map"], []).append(record)
    if not records or any(len(p) != 2 or {r["seat"] for r in p} != {0, 1} for p in pairs.values()):
        raise ValueError("Evaluation requires complete map/color pairs")
    games, count = len(records), len(pairs)
    win_rate = outcomes["win"] / games
    net = (outcomes["win"] - outcomes["loss"]) / games
    radius = math.sqrt(math.log(40) / (2 * count))
    ticks = sum(r["ticks"] for r in records)
    movable = sum(r["movable"] for r in records)
    return {
        "games": games, "map_pairs": count,
        "wins": outcomes["win"], "losses": outcomes["loss"],
        "draws": outcomes["draw"], "truncated": outcomes["truncated"],
        "win_rate": win_rate, "net_win_rate": net,
        "win_rate_ci95": [max(0., win_rate - radius), min(1., win_rate + radius)],
        "net_win_rate_ci95": [max(-1., net - 2 * radius), min(1., net + 2 * radius)],
        "interval": "map-paired Hoeffding, fixed-look; unresolved games score zero",
        "mean_ticks": ticks / games,
        "pass_fraction": sum(r["passes"] for r in records) / max(ticks, 1),
        "optional_pass_fraction": sum(r["optional_passes"] for r in records) / movable if movable else None,
        "movable_samples": movable,
    }


def paired_change(current, reference):
    """Compare the same map/color pairs; the interval covers a single comparison."""
    first, second = current["records"], reference["records"]
    settings = ("seed", "side", "distance", "horizon", "selection", "opponent_sha256")
    if len(first) != len(second) or any(current[k] != reference[k] for k in settings):
        raise ValueError("Initial evaluation settings or opponent changed")
    for a, b in zip(first, second):
        if any(a[k] != b[k] for k in ("map", "seat", "map_sha256")):
            raise ValueError("Initial evaluation map/color pairs changed")
    change = current["net_win_rate"] - reference["net_win_rate"]
    radius = 4 * math.sqrt(math.log(40) / (2 * current["map_pairs"]))
    return {"net_win_rate": change, "net_win_rate_ci95": [max(-2., change - radius), min(2., change + radius)],
            "win_rate": current["win_rate"] - reference["win_rate"],
            "truncated_fraction": (current["truncated"] - reference["truncated"]) / current["games"]}


class Evaluator:
    def __init__(self, config, history, games=128, lanes=16, seed=1001012, greedy=False, excluded=None):
        if games < 2 or games % 2 or lanes < 2 or lanes % 2:
            raise ValueError("games and evaluation lanes must be positive even counts")
        self.config, self.history = config, history
        self.games, self.lanes = games, min(lanes, games)
        self.seed, self.greedy = seed, greedy
        self.topology = graph(config.side, config.side)
        emit({"phase": "evaluation_maps", "games": games, "seed": seed,
              "side": config.side, "distance": config.distance})
        # A separate PRNG domain prevents accidental reuse of training generation keys.
        key = jax.random.fold_in(jax.random.PRNGKey(seed), 0x4556414C)
        keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(games // 2))
        self.pool = initial_maps(config, keys)
        self.hashes = map_hashes(self.pool)
        if excluded is not None and set(self.hashes) & set(map_hashes(excluded)):
            raise ValueError("Evaluation map overlaps training pool; choose another evaluation seed")
        self.start = eqx.filter_jit(self._start)
        self.advance = eqx.filter_jit(self._advance)

    def _start(self, indices):
        states = jax.tree.map(lambda x: x[jnp.asarray(indices) // 2], self.pool)
        memory = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (len(indices), 2, *x.shape)),
            empty(self.config.side**2, self.history),
        )
        previous = jnp.full((len(indices), 2), self.config.side**2 * 8)
        inputs, memory = encode_players(views(states, self.topology), memory, previous, self.topology)
        zeros = jnp.zeros(len(indices), jnp.int32)
        return Match(states, memory, inputs, baseline.empty(len(indices), self.config.side**2),
                     jnp.ones(len(indices), bool), jnp.full(len(indices), 3), zeros, zeros,
                     zeros, zeros, zeros)

    def _advance(self, model, opponent, match, indices):
        side, topology = self.config.side, self.topology
        area, count = side**2, len(indices)
        seats, rows = indices % 2, jnp.arange(count)
        keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(self.seed), i))(indices)
        predict = jax.vmap(model, in_axes=(0, None))
        rival = jax.vmap(opponent, in_axes=(0, None)) if opponent is not None else None

        def select(logits, key):
            return jnp.argmax(logits, -1) if self.greedy else jax.vmap(jax.random.categorical)(key, logits)

        def tick(current):
            state, memory, inputs, bot = current[:4]
            ours = jax.tree.map(lambda x: x[rows, seats], inputs)
            theirs = jax.tree.map(lambda x: x[rows, 1 - seats], inputs)
            tick_keys = jax.vmap(jax.random.fold_in)(keys, state.time)
            our_keys = jax.vmap(lambda k: jax.random.fold_in(k, 0))(tick_keys)
            their_keys = jax.vmap(lambda k: jax.random.fold_in(k, 1))(tick_keys)
            our_action = select(predict(ours, topology).logits, our_keys)
            if rival is None:
                obs = jax.tree.map(lambda x: x[rows, 1 - seats], views(state, topology))
                bot, their_action = baseline.decisions(bot, *baseline_inputs(state, obs, side))
                their_action = baseline_action(their_action, area)
            else:
                their_action = select(rival(theirs, topology).logits, their_keys)
            commands = jnp.full((count, 2), area * 8, jnp.int32)
            commands = commands.at[rows, seats].set(our_action).at[rows, 1 - seats].set(their_action)
            commands = jnp.where(current.active[:, None], commands, area * 8)
            legal = jnp.take_along_axis(inputs.legal, jnp.clip(commands, 0, area * 8)[..., None], -1)[..., 0]
            valid = legal & (commands >= 0) & (commands <= area * 8)
            moves = jax.vmap(jax.vmap(lambda a: action(a, side, area)))(commands)
            following, info = jax.vmap(lambda s, a: step(s, a, general_trade=True))(state, moves)
            terminal = info.is_done
            timeout = (following.time >= self.config.horizon) & ~terminal
            ended = current.active & (terminal | timeout)
            outcome = jnp.where(timeout, 2, jnp.where(info.winner == seats, 1,
                                jnp.where(info.winner == 1 - seats, -1, 0)))
            next_inputs, next_memory = encode_players(views(following, topology), memory, commands, topology)
            movable = ours.legal[:, :-1].any(-1) & current.active
            passed = (our_action == area * 8) & current.active
            return Match(
                choose(current.active, following, state),
                choose(current.active, next_memory, memory),
                choose(current.active, next_inputs, inputs), bot,
                current.active & ~ended, jnp.where(ended, outcome, current.outcome),
                current.ticks + current.active.astype(jnp.int32),
                current.passes + passed.astype(jnp.int32),
                current.optional_passes + (passed & movable).astype(jnp.int32),
                current.movable + movable.astype(jnp.int32),
                current.invalid + (current.active & ~valid.all(-1)).astype(jnp.int32),
            )

        # Short compiled chunks bound progress-report latency without recording trajectories.
        return jax.lax.while_loop(lambda item: (item[0] < 32) & item[1].active.any(),
                                  lambda item: (item[0] + 1, tick(item[1])), (0, match))[1]

    def matches(self, model, opponent, name):
        if model.config.history != self.history or (opponent is not None and opponent.config.history != self.history):
            raise ValueError("Evaluation policies must have matching history lengths")
        started, records = time.monotonic(), []
        emit({"phase": "evaluation", "event": "start", "opponent": name, "games": self.games,
              "lanes": self.lanes, "horizon": self.config.horizon})
        for offset in range(0, self.games, self.lanes):
            ids = np.arange(offset, min(offset + self.lanes, self.games), dtype=np.int32)
            match = self.start(ids)
            while bool(match.active.any()):
                match = self.advance(model, opponent, match, jnp.asarray(ids))
                if bool(match.invalid.any()):
                    raise RuntimeError("Evaluation player submitted an action outside its legal mask")
                emit({"phase": "evaluation", "opponent": name, "finished_games": len(records),
                      "tick": int(match.ticks.max()), "active": int(match.active.sum()),
                      "seconds": round(time.monotonic() - started, 2)})
            names = {-1: "loss", 0: "draw", 1: "win", 2: "truncated"}
            values = [np.asarray(x) for x in (match.outcome, match.ticks, match.passes,
                                             match.optional_passes, match.movable)]
            for i, index in enumerate(ids):
                records.append({"game": int(index), "map": int(index // 2), "seat": int(index % 2),
                                "map_sha256": self.hashes[index // 2], "outcome": names[int(values[0][i])],
                                **{key: int(value[i]) for key, value in zip(
                                    ("ticks", "passes", "optional_passes", "movable"), values[1:])}})
        return {"opponent": name, "seed": self.seed, "side": self.config.side,
                "distance": self.config.distance, "horizon": self.config.horizon,
                "selection": "argmax" if self.greedy else "categorical",
                **summarize(records), "seconds": time.monotonic() - started, "records": records}


class EvaluationSuite:
    """Own immutable references and evaluation artifacts, independently of PPO."""
    def __init__(self, output, original, initial, config, initialization, games, lanes, seed, excluded):
        self.output = Path(output)
        self.references = self.output / "references"
        metadata = {"phase": "ppo_reference", "run": asdict(config), **initialization}
        save(self.references / "original.zip", original, dict(metadata, reference="original", init_pass_scale=1.0))
        save(self.references / "initial.zip", initial, dict(metadata, reference="initial"))
        self.evaluator = Evaluator(config, initial.config.history, games, lanes, seed, excluded=excluded)
        self.baseline_hash = file_hash(Path(baseline.__file__))

    def run(self, model, iteration, checkpoint, history=2):
        rivals = [("original", self.references / "original.zip"),
                  ("initial", self.references / "initial.zip")]
        snapshots = sorted((self.output / "snapshots").glob("iteration-*.zip"))
        older = [p for p in snapshots if 0 < int(p.stem.split("-")[-1]) < iteration]
        if history and older:
            # Even spacing retains early references alongside recent versions.
            ids = (np.array([len(older) - 1]) if history == 1 else
                   np.linspace(0, len(older) - 1, min(history, len(older)), dtype=int))
            rivals.extend((older[i].stem, older[i]) for i in ids)
        rivals.append(("baseline", None))
        checkpoint_hash = file_hash(checkpoint)
        for name, path in rivals:
            opponent = load(path)[0] if path else None
            result = self.evaluator.matches(model, opponent, name)
            result.update(iteration=iteration, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                          opponent_sha256=file_hash(path) if path else self.baseline_hash)
            references = sorted((self.output / "evaluation").glob(f"iteration-*/{name}.json"))
            if references:
                with references[0].open(encoding="utf-8") as stream:
                    reference = json.load(stream)
                result["comparison_iteration"] = reference["iteration"]
                result["change_from_first_evaluation"] = paired_change(result, reference)
            folder = self.output / "evaluation" / f"iteration-{iteration:08d}"
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / f"{name}.json").open("w", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
            summary = {k: v for k, v in result.items() if k != "records"}
            with (self.output / "evaluation.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(summary) + "\n")
            emit({"phase": "evaluation", "event": "result", **summary})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    opponent = parser.add_mutually_exclusive_group(required=True)
    opponent.add_argument("--opponent", type=Path)
    opponent.add_argument("--baseline", action="store_true")
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--lanes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1001012)
    parser.add_argument("--side", type=int, default=Run.side)
    parser.add_argument("--distance", type=int, default=Run.distance)
    parser.add_argument("--horizon", type=int, default=Run.horizon)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new result file to preserve earlier evaluations")
    if args.side < 4 or not 1 <= args.distance <= 2 * (args.side - 1) or args.horizon < 1:
        parser.error("Require side >= 4, feasible distance and positive horizon")
    model, _ = load(args.checkpoint)
    rival = load(args.opponent)[0] if args.opponent else None
    config = Run(side=args.side, distance=args.distance, horizon=args.horizon)
    evaluator = Evaluator(config, model.config.history, args.games, args.lanes, args.seed, args.greedy)
    result = evaluator.matches(model, rival, str(args.opponent) if args.opponent else "baseline")
    result.update(checkpoint_sha256=file_hash(args.checkpoint),
                  opponent_sha256=file_hash(args.opponent or Path(baseline.__file__)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    emit({k: v for k, v in result.items() if k != "records"})


if __name__ == "__main__":
    main()
