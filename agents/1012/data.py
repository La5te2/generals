"""Bounded archive/parquet conversion and causal replay batches.

Whole-map hashes determine splits (both seats and duplicate games stay together).
Only player-view fields enter Inputs. Ground truth remains in separate targets.
Files are streamed, never extracted with archive-supplied filesystem paths.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from collections import Counter
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from features import Observation, empty, encode, graph, observe

ROOT = Path(__file__).resolve().parent


def records(path):
    path = Path(path)
    if path.is_dir():
        for child in sorted(path.rglob("*.gioreplay")):
            yield child.name, json.loads(child.read_text(encoding="utf-8"))
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
            for row in batch.to_pylist():
                yield row["id"], row
    else:
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                if member.isfile() and member.name.endswith((".gioreplay", ".json")):
                    if member.size > 16 * 1024 * 1024:
                        raise ValueError(f"Oversized replay: {member.name}")
                    yield member.name, json.load(archive.extractfile(member))


def map_key(replay):
    fields = ("mapWidth", "mapHeight", "cities", "cityArmies", "generals", "mountains")
    # Canonical geometry independent of recorded list ordering and player seat.
    canonical = [
        replay[fields[0]],
        replay[fields[1]],
        sorted(zip(replay["cities"], replay["cityArmies"])),
        sorted(replay["generals"]),
        sorted(replay["mountains"]),
    ]
    return hashlib.sha256(
        json.dumps(canonical, separators=(",", ":")).encode()
    ).hexdigest()


def split_for(replay):
    value = int(map_key(replay)[:8], 16) % 100
    return "train" if value < 85 else "validation" if value < 95 else "test"


class Simulator:
    def __init__(self):
        node = shutil.which("node")
        if node is None:
            raise RuntimeError(
                "Node.js is required only for historical replay conversion"
            )
        self.process = subprocess.Popen(
            [node, str(ROOT / "replay.js")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )

    def replay(self, value, limit):
        self.process.stdin.write(
            json.dumps({"replay": value, "limit": limit}, ensure_ascii=True) + "\n"
        )
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Replay subprocess exited")
        result = json.loads(line)
        if "error" in result:
            raise ValueError(result["error"])
        return result

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=10)
        self.process.stdout.close()


@eqx.filter_jit
def encode_chunk(obs, actions, memory, previous, topology):
    def one(carry, row):
        memory, previous = carry
        observation, action = row
        x, memory = encode(observation, memory, previous, topology)
        return (memory, action), x

    return jax.lax.scan(one, (memory, previous), (obs, actions))


def batches(path, history, chunk=32, min_stars=0):
    """Generate bounded chunks; history persists across chunks, never across games."""
    with np.load(path, allow_pickle=False) as file:
        data = dict(file)
    h, w = data["shape"].tolist()
    topology = graph(h, w)
    for player in range(2):
        if data["stars"][player] < min_stars:
            continue
        memory, previous = empty(h * w, history), jnp.int32(h * w * 8)
        for start in range(0, len(data["actions"]), chunk):
            stop = min(start + chunk, len(data["actions"]))
            actions = data["actions"][start:stop, player]
            obs = Observation(
                jnp.asarray(data["cells"][start:stop, player]),
                jnp.asarray(data["stats"][start:stop, player]),
                jnp.ones((stop - start, h * w), bool),
            )
            (memory, previous), x = encode_chunk(
                obs,
                jnp.asarray(np.where(actions >= 0, actions, h * w * 8)),
                memory,
                previous,
                topology,
            )
            legal = np.asarray(x.legal)
            usable = (actions >= 0) & legal[
                np.arange(len(actions)), np.maximum(actions, 0)
            ]
            targets = {
                "action": jnp.asarray(np.maximum(actions, 0)),
                "weight": jnp.asarray(usable, jnp.float32),
                "general": jnp.asarray(data["generals"][start:stop, 1 - player]),
                "owner": jnp.asarray(
                    np.where(
                        data["owners"][start:stop] == player,
                        1,
                        np.where(data["owners"][start:stop] == 1 - player, 2, 0),
                    )
                ),
                "army": jnp.log1p(jnp.asarray(data["armies"][start:stop], jnp.float32))
                / jnp.log(256.0),
            }
            yield x, targets, topology


def convert(inputs, output, limit, max_ticks):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    counts, accepted, seen = Counter(), 0, set()
    manifest_path = output / "manifest.jsonl"
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            seen.add(json.loads(line)["id"])
    simulator = Simulator()
    make_views = eqx.filter_jit(
        jax.vmap(
            lambda a, o, m, c, g, t, p, r, b, top: observe(
                a, o, m, c, g, t, p, r, b, top
            ),
            in_axes=(0, 0, None, 0, 0, 0, None, None, None, None),
        )
    )
    try:
        with manifest_path.open("a", encoding="utf-8") as manifest:
            for source in inputs:
                for name, replay in records(source):
                    identity = str(replay.get("id", name))
                    if identity in seen:
                        counts["duplicate"] += 1
                        continue
                    try:
                        h, w = int(replay["mapHeight"]), int(replay["mapWidth"])
                        if not (2 <= h <= 32 and 2 <= w <= 32):
                            raise ValueError("Map size outside 2..32")
                        result = simulator.replay(replay, max_ticks)
                        if result["invalid"]:
                            raise ValueError(
                                f"{result['invalid']} invalid recorded actions"
                            )
                        if result["winner"] < 0:
                            raise ValueError("No confirmed terminal result")
                        frames = result["frames"]
                        n, area = len(frames), h * w
                        armies = np.array([f["armies"] for f in frames], np.int32)
                        owners = np.array([f["owners"] for f in frames], np.int8)
                        generals = np.array([f["generals"] for f in frames], np.int32)
                        city = np.zeros((n, area), bool)
                        gen = np.zeros_like(city)
                        for i, frame in enumerate(frames):
                            city[i, frame["cities"]] = True
                            gen[i, [g for g in frame["generals"] if g >= 0]] = True
                        mountain = np.zeros(area, bool)
                        mountain[replay["mountains"]] = True
                        rules = (replay["version"] < 6, True, False)
                        views = []
                        # Fixed chunks avoid recompiling observation generation for
                        # every different game length. Only board shapes specialize.
                        for p in range(2):
                            chunks = []
                            for start in range(0, n, 64):
                                indices = np.minimum(
                                    np.arange(start, start + 64), n - 1
                                )
                                view = make_views(
                                    jnp.array(armies[indices]),
                                    jnp.array(owners[indices]),
                                    jnp.array(mountain),
                                    jnp.array(city[indices]),
                                    jnp.array(gen[indices]),
                                    jnp.array(indices),
                                    p,
                                    rules,
                                    jnp.ones(area, bool),
                                    graph(h, w),
                                )
                                chunks.append(
                                    jax.tree.map(
                                        lambda x, start=start, n=n: np.asarray(x)[
                                            : min(64, n - start)
                                        ],
                                        view,
                                    )
                                )
                            views.append(
                                jax.tree.map(lambda *xs: np.concatenate(xs), *chunks)
                            )
                        actions = np.array([f["actions"] for f in frames], np.int32)
                        actions[actions == -1] = area * 8
                        arrays = {
                            "cells": np.stack([np.asarray(v.cells) for v in views], 1),
                            "stats": np.stack([np.asarray(v.stats) for v in views], 1),
                            "actions": actions,
                            "owners": owners,
                            "armies": armies,
                            "generals": generals,
                            "shape": np.array([h, w]),
                            "stars": np.asarray(replay.get("stars") or [0, 0]),
                            "version": np.array(replay["version"]),
                            "winner": np.array(result["winner"]),
                        }
                        split = split_for(replay)
                        folder = output / split
                        folder.mkdir(exist_ok=True)
                        filename = (
                            hashlib.sha256(identity.encode()).hexdigest()[:24] + ".npz"
                        )
                        np.savez_compressed(folder / filename, **arrays)
                        entry = {
                            "id": identity,
                            "source": str(source),
                            "file": f"{split}/{filename}",
                            "version": replay["version"],
                            "map": map_key(replay),
                            "ticks": n,
                        }
                        manifest.write(json.dumps(entry) + "\n")
                        manifest.flush()
                        seen.add(identity)
                        accepted += 1
                        counts[split] += 1
                        print(
                            json.dumps(
                                {
                                    "accepted": accepted,
                                    "id": identity,
                                    "ticks": n,
                                    "split": split,
                                }
                            ),
                            flush=True,
                        )
                    except (ValueError, KeyError, IndexError) as error:
                        counts[f"rejected:{error}"] += 1
                    if limit and accepted >= limit:
                        return counts
    finally:
        simulator.close()
    return counts


def inspect(inputs):
    report = {}
    previous_ids = set()
    for source in inputs:
        versions, players, stars = Counter(), Counter(), Counter()
        total, afks, too_long = 0, 0, 0
        identities = set()
        for _, replay in records(source):
            total += 1
            identities.add(str(replay.get("id")))
            versions[str(replay.get("version"))] += 1
            players[str(len(replay.get("generals", [])))] += 1
            for star in replay.get("stars") or []:
                stars[str(int(star) // 10 * 10)] += 1
            afks += bool(replay.get("afks"))
            moves = replay.get("moves") or []
            if moves:
                last = (
                    moves[-1]["turn"] if isinstance(moves[-1], dict) else moves[-1][4]
                )
                too_long += last >= 4096
        report[str(source)] = {
            "games": total,
            "versions": versions,
            "players": players,
            "stars_by_decade": stars,
            "has_afk": afks,
            "longer_than_4096": too_long,
            "duplicate_ids_with_earlier_sources": len(identities & previous_ids),
        }
        previous_ids.update(identities)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-ticks", type=int, default=4096)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.limit < 0 or args.max_ticks < 1:
        parser.error("Nonnegative limit and positive max-ticks required")
    report = (
        inspect(args.inputs)
        if args.inspect
        else convert(args.inputs, args.output, args.limit, args.max_ticks)
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
