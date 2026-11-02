"""Fit a tiny real replay batch with the default network, not a toy classifier.

This detects disconnected gradients and incompatible labels. It does not
measure held-out generalization, policy strength or information sufficiency.
"""

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from data import batches
from model import Network
from train import make_update, optimizer, save, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/pilot/train"))
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("runs/smoke"))
    args = parser.parse_args()
    if args.updates < 1 or args.batch < 1:
        parser.error("Positive batch and updates required")
    model = Network(key=jax.random.PRNGKey(1012))
    selected = None
    for path in sorted(args.data.glob("*.npz")):
        for x, y, topology in batches(path, model.config.history, 32):
            ids = np.flatnonzero(
                np.asarray((y["weight"] > 0) & (y["action"] < x.legal.shape[-1] - 1))
            )
            if len(ids) >= args.batch:
                selected = (
                    path,
                    jax.tree.map(lambda a, ids=ids: a[ids[: args.batch]], x),
                    jax.tree.map(lambda a, ids=ids: a[ids[: args.batch]], y),
                    topology,
                )
                break
        if selected:
            break
    if selected is None:
        raise ValueError("No real replay chunk contains enough usable move labels")
    path, x, y, topology = selected
    tx = optimizer(3e-4)
    state = tx.init(eqx.filter(model, eqx.is_inexact_array))
    update = make_update(tx, 0.02)

    def measure(model):
        row = np.asarray(validate(model, x, y, topology))
        return {
            "action_nll": float(row[0] / row[3]),
            "accuracy": float(row[1] / row[3]),
        }

    start = time.monotonic()
    initial = measure(model)
    print(
        json.dumps(
            {
                "phase": "fit",
                "initial": initial,
                "file": str(path),
                "shape": list(x.cells.shape),
            }
        ),
        flush=True,
    )
    for index in range(args.updates):
        model, state, loss, _, gradient = update(model, state, x, y, topology)
        if not bool(jnp.isfinite(loss) & jnp.isfinite(gradient)):
            raise FloatingPointError("Nonfinite fit")
        if (index + 1) % 4 == 0:
            print(json.dumps({"updates": index + 1, **measure(model)}), flush=True)
    final = measure(model)
    result = {
        "kind": "training_batch_fit_only",
        "parameters": sum(a.size for a in jax.tree.leaves(model)),
        "examples": args.batch,
        "updates": args.updates,
        "initial": initial,
        "final": final,
        "seconds": time.monotonic() - start,
        "source": str(path),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    save(args.output / "policy.zip", model, result)
    print(json.dumps(result), flush=True)
    if final["action_nll"] >= initial["action_nll"]:
        raise AssertionError("Training batch did not improve")


if __name__ == "__main__":
    main()
