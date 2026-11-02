"""Evaluate one named baseline pair with unchanged agents/1014 code. Never trains."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "docs/research/2026-11-defense-baseline"
PAIRS = {
    "P1": ("iteration-00000033", "iteration-00000039"),
    "P2": ("iteration-00000033", "ppo20"),
    "P3": ("iteration-00000033", "random"),
    "P4": ("iteration-00000033", "expander"),
    "P5": ("iteration-00000033", "hunter"),
    "P6": ("iteration-00000039", "ppo20"),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True, choices=PAIRS)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        help="Optional new learner; records its actual hash instead of claiming historical parity")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--lanes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=930001)
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.games < 2 or args.games % 2 or args.lanes < 2 or args.lanes % 2:
        parser.error("games and lanes must be positive even counts")
    if args.horizon < 1 or not 0 <= args.seed < 2**32:
        parser.error("require a positive horizon and a uint32 seed")
    if args.output.exists():
        parser.error("output exists; choose a new path to preserve prior results")

    manifest = json.loads((ARCHIVE / "manifest.json").read_text())
    for name, expected in manifest["agent_source_sha256"].items():
        if sha256(ROOT / name) != expected:
            parser.error(f"agent source differs from the archived version: {name}; use this PR's source version")
    learner_id, opponent_id = PAIRS[args.pair]

    def snapshot(name):
        entry = manifest["models"][name]
        path = args.models_dir / entry["filename"]
        if not path.is_file():
            parser.error(f"missing {path}; obtain the existing snapshot from project maintainers")
        if sha256(path) != entry["sha256"]:
            parser.error(f"snapshot SHA256 mismatch: {path}")
        return path

    learner_path = args.checkpoint if args.checkpoint else snapshot(learner_id)
    if not learner_path.is_file():
        parser.error(f"learner checkpoint not found: {learner_path}")
    opponent_path = snapshot(opponent_id) if opponent_id in manifest["models"] else None

    # Import the repository's existing model/evaluator only after CLI validation.
    sys.path.insert(0, str(ROOT / "agents/1014"))
    import jax
    from environment import Run
    from evaluate import Evaluator
    from train import load
    from rule_agents import build

    if jax.default_backend() != "cpu":
        parser.error("this baseline protocol uses CPU/fp32; set JAX_PLATFORMS=cpu")
    before = {str(p): sha256(p) for p in (learner_path, opponent_path) if p is not None}
    learner = load(learner_path)[0]
    opponent = load(opponent_path)[0] if opponent_path else build(opponent_id, 24, learner.config.history)
    evaluator = Evaluator(Run(side=24, distance=17, horizon=args.horizon),
                          learner.config.history, args.games, args.lanes, args.seed, True)
    if len(set(evaluator.hashes)) != args.games // 2:
        raise ValueError("duplicate generated map identities")
    result = evaluator.matches(learner, opponent, opponent_id)
    if any(sha256(Path(p)) != expected for p, expected in before.items()):
        raise ValueError("snapshot changed during evaluation")
    for name, expected in manifest["agent_source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"agent source changed during evaluation: {name}")
    result.update(pair=args.pair, checkpoint_sha256=before[str(learner_path)],
                  custom_learner=args.checkpoint is not None,
                  agent_source_sha256=manifest["agent_source_sha256"])
    if opponent_path:
        result["opponent_sha256"] = before[str(opponent_path)]
    else:
        result["agent"] = opponent_id
        result["rule_adapter_sha256"] = sha256(Path(__file__).with_name("rule_agents.py"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("pair", "wins", "losses", "draws", "truncated", "win_rate")}))


if __name__ == "__main__":
    main()
