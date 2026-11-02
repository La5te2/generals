"""Recount archived baseline games and pairing checks using only the standard library."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parents[2] / "docs/research/2026-11-defense-baseline"
INITIAL = {
    "P1": ("P1-33-vs-39.json", "iteration-00000039"),
    "P2": ("P2-33-vs-ppo20.json", "ppo20"),
    "P3": ("P3-33-vs-random.json", "random"),
    "P4": ("P4-33-vs-expander.json", "expander"),
    "P5": ("P5-33-vs-hunter.json", "hunter"),
}
OUTCOMES = {"win": "wins", "loss": "losses", "draw": "draws", "truncated": "truncated"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tally(records):
    require(bool(records), "empty records")
    require(all(r["outcome"] in OUTCOMES for r in records), "unknown outcome")
    maps = {}
    for r in records:
        maps.setdefault(r["map"], []).append(r)
    require(all(len(g) == 2 and {r["seat"] for r in g} == {0, 1}
                and len({r["map_sha256"] for r in g}) == 1 for g in maps.values()),
            "incomplete map/seat pairing")
    count = Counter(r["outcome"] for r in records)
    games = len(records)
    rate = count["win"] / games
    radius = math.sqrt(math.log(40) / (2 * len(maps)))
    return {"games": games, "map_pairs": len(maps),
            **{plural: count[one] for one, plural in OUTCOMES.items()},
            "win_rate": rate, "win_rate_ci95": [max(0, rate-radius), min(1, rate+radius)]}


def check_result(result, games, seed):
    for key, value in {"games": games, "seed": seed, "side": 24, "distance": 17,
                       "horizon": 1024, "selection": "argmax", "opponent_selection": "argmax"}.items():
        require(result[key] == value, f"unexpected {key}")
    records = result["records"]
    require(len(records) == games and sorted(r["game"] for r in records) == list(range(games)),
            "missing or duplicate game IDs")
    require(all(r["map"] == r["game"] // 2 and r["seat"] == r["game"] % 2 for r in records),
            "unexpected map/seat IDs")
    require(all(0 < r["ticks"] <= 1024 for r in records), "invalid terminal tick")
    calculated = tally(records)
    for key, expected in calculated.items():
        if key == "win_rate_ci95":
            require(len(result[key]) == 2 and all(math.isclose(a, b, abs_tol=1e-12)
                    for a, b in zip(result[key], expected)), "inconsistent confidence interval")
        else:
            require(result[key] == expected, f"summary disagrees with records: {key}")
    maps = [r["map_sha256"] for r in sorted(records, key=lambda r: r["game"]) if r["seat"] == 0]
    require(len(set(maps)) == games // 2, "duplicate physical map hash")
    return maps, calculated


def paired_outcomes(first, second):
    key = lambda r: (r["map_sha256"], r["seat"])
    a = {key(r): r["outcome"] == "win" for r in first}
    b = {key(r): r["outcome"] == "win" for r in second}
    require(a.keys() == b.keys() and len(a) == len(first) == len(second), "unmatched paired comparisons")
    counts = Counter((a[k], b[k]) for k in a)
    left, right = counts[(True, False)], counts[(False, True)]
    n = left + right
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(left, right)+1)) / 2**n) if n else 1.0
    return {"both_win": counts[(True, True)], "only_33_win": left,
            "only_39_win": right, "both_nonwin": counts[(False, False)],
            "mcnemar_exact_two_sided_p": p}


def verify(archive=ARCHIVE):
    read = lambda name: json.loads((archive / name).read_text())
    manifest = read("manifest.json")
    for entry in manifest["files"]:
        actual = hashlib.sha256((archive / entry["path"]).read_bytes()).hexdigest()
        require(actual == entry["sha256"], f"archive hash mismatch: {entry['path']}")
    model_hash = lambda name: manifest["models"][name]["sha256"]
    results, summaries, initial_maps = {}, {}, None
    for pair, (filename, opponent) in INITIAL.items():
        result = read("data/baseline/initial/" + filename)
        maps, summaries[pair] = check_result(result, 200, 930001)
        require(result["checkpoint_sha256"] == model_hash("iteration-00000033"), "wrong learner")
        if opponent in manifest["models"]:
            require(result["opponent_sha256"] == model_hash(opponent), "wrong opponent checkpoint")
        else:
            require(result["agent"] == opponent, "wrong rule opponent")
        require(initial_maps is None or maps == initial_maps, "initial panels used different maps")
        initial_maps = maps
        results[pair] = result
    second = read("data/baseline/extension/P2-batch2-33-vs-ppo20.json")
    new_maps, summaries["P2_batch2"] = check_result(second, 400, 930002)
    require(not set(initial_maps) & set(new_maps), "extension maps overlap the first batch")
    require(second["checkpoint_sha256"] == model_hash("iteration-00000033")
            and second["opponent_sha256"] == model_hash("ppo20"), "wrong extension checkpoints")
    p6 = read("data/baseline/extension/P6-39-vs-ppo20.json")
    p6_maps, summaries["P6"] = check_result(p6, 200, 930001)
    require(p6_maps == initial_maps, "P6 does not use the shared 100 maps")
    require(p6["checkpoint_sha256"] == model_hash("iteration-00000039")
            and p6["opponent_sha256"] == model_hash("ppo20"), "wrong P6 checkpoints")
    merged = []
    for batch, result in enumerate((results["P2"], second)):
        merged += [{**r, "map": f"{batch}:{r['map']}"} for r in result["records"]]
    summaries["P2_combined"] = tally(merged)
    return {"passed": True, "archived_files_checked": len(manifest["files"]),
            "formal_games_recounted": sum(r["games"] for key, r in summaries.items()
                                          if key != "P2_combined"), "pairs": summaries,
            "shared_map_33_vs_39": paired_outcomes(results["P2"]["records"], p6["records"]),
            "caveats": [
                "Combined P2 interval is nominal fixed-look Hoeffding; expansion followed observation of batch 1, with no optional-stopping adjustment.",
                "P6 only covers the first 100 maps. Equal observed win counts and p=1 do not establish equivalence or a causal saturation mechanism.",
                "Four-direction excerpts are file-hash checked; their underlying training and probe fits are not rerun by this verifier.",
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = verify(args.archive)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(f"PASS: {report['formal_games_recounted']} games recounted; {report['archived_files_checked']} evidence files hash-checked")
        for name, r in report["pairs"].items():
            lo, hi = r["win_rate_ci95"]
            print(f"{name:12} {r['wins']:3}/{r['games']:<3} {r['win_rate']:.2%}  nominal 95% [{lo:.2%}, {hi:.2%}]")
        print("Shared-map comparison:", report["shared_map_33_vs_39"])
        for caveat in report["caveats"]:
            print("NOTE:", caveat)


if __name__ == "__main__":
    main()
