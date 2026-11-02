"""Supervised policy learning and shared checkpoint/update utilities.

Human actions supervise the actor; recorded outcomes are not current-policy
values. Validation uses held-out complete maps. PPO starts a fresh optimizer
and estimates its own value targets from on-policy rollouts.
"""

import argparse
import hashlib
import io
import json
import os
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from data import batches
from model import Config, Network


def save(path, model, metadata, optimizer_state=None):
    """Atomically persist array leaves and explicit JSON metadata, without pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            "metadata.json",
            json.dumps(dict(format=1, config=asdict(model.config), **metadata)),
        )
        stream = io.BytesIO()
        eqx.tree_serialise_leaves(stream, model)
        archive.writestr("model.eqx", stream.getvalue())
        if optimizer_state is not None:
            stream = io.BytesIO()
            eqx.tree_serialise_leaves(stream, optimizer_state)
            archive.writestr("optimizer.eqx", stream.getvalue())
    os.replace(temp, path)


def load(path):
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        if metadata["format"] != 1:
            raise ValueError("Unsupported checkpoint schema")
        model = Network(Config(**metadata["config"]), key=jax.random.PRNGKey(0))
        model = eqx.tree_deserialise_leaves(
            io.BytesIO(archive.read("model.eqx")), model
        )
    return model, metadata


def restore_optimizer(path, template):
    """Restore training state against the exact optimizer structure in use."""
    with zipfile.ZipFile(path) as archive:
        return eqx.tree_deserialise_leaves(
            io.BytesIO(archive.read("optimizer.eqx")), template
        )


def optimizer(rate):
    return optax.chain(optax.clip_by_global_norm(0.5), optax.adam(rate))


def auxiliary(output, inputs, targets):
    """Marginal prediction losses. Targets never enter a model forward call."""
    hidden = inputs.board & (inputs.cells[..., 0] == 0)
    owner = optax.softmax_cross_entropy_with_integer_labels(
        output.owner, targets["owner"]
    )
    army = optax.huber_loss(output.army, targets["army"])
    spatial = jnp.sum((owner + army) * hidden, -1) / jnp.maximum(hidden.sum(-1), 1)
    general = optax.softmax_cross_entropy_with_integer_labels(
        output.general, jnp.maximum(targets["general"], 0)
    )
    return spatial + jnp.where(targets["general"] >= 0, general, 0.0)


def supervised_loss(model, x, targets, topology, auxiliary_weight):
    output = jax.vmap(model, in_axes=(0, None))(x, topology)
    weight = targets["weight"]
    denominator = jnp.maximum(weight.sum(), 1.0)
    ce = optax.softmax_cross_entropy_with_integer_labels(
        output.logits, targets["action"]
    )
    aux = auxiliary(output, x, targets)
    loss = jnp.sum((ce + auxiliary_weight * aux) * weight) / denominator
    correct = (jnp.argmax(output.logits, -1) == targets["action"]) * weight
    moving = targets["action"] != output.logits.shape[-1] - 1
    # Accumulate numerators so short batches and pass-heavy games cannot distort metrics.
    return loss, jnp.array(
        [
            jnp.sum(ce * weight),
            correct.sum(),
            jnp.sum(aux * weight),
            weight.sum(),
            jnp.sum(correct * moving),
            jnp.sum(weight * moving),
            jnp.sum(correct * ~moving),
            jnp.sum(weight * ~moving),
        ]
    )


def make_update(tx, auxiliary_weight):
    @eqx.filter_jit
    def update(model, state, x, targets, topology):
        (loss, metrics), gradients = eqx.filter_value_and_grad(
            supervised_loss, has_aux=True
        )(model, x, targets, topology, auxiliary_weight)
        changes, state = tx.update(
            gradients, state, eqx.filter(model, eqx.is_inexact_array)
        )
        return (
            eqx.apply_updates(model, changes),
            state,
            loss,
            metrics,
            optax.tree.norm(gradients),
        )

    return update


@eqx.filter_jit
def validate(model, x, targets, topology):
    return supervised_loss(model, x, targets, topology, 0.0)[1]


def report(metrics):
    metrics = np.asarray(metrics)
    total = metrics[3]
    if total == 0:
        raise ValueError(
            "No usable examples; inspect conversion report, split and star filter"
        )
    return dict(
        zip(
            ("action_nll", "accuracy", "auxiliary_loss"), (metrics[:3] / total).tolist()
        ),
        examples=int(total),
        move_accuracy=float(metrics[4] / max(metrics[5], 1)),
        pass_accuracy=float(metrics[6] / max(metrics[7], 1)),
        move_examples=int(metrics[5]),
        pass_examples=int(metrics[7]),
    )


def epoch(
    model,
    files,
    history,
    batch,
    min_stars,
    update=None,
    state=None,
    seed=0,
    progress=None,
    callback=None,
    max_batches=0,
):
    """Resume by game/chunk; reconstruct causal history within the interrupted game."""
    rng = np.random.default_rng(seed)
    files = sorted(files)
    if update is not None:
        rng.shuffle(files)
    progress = progress or {"file": 0, "chunk": 0, "batches": 0, "metrics": [0.0] * 8}
    metrics = np.array(progress["metrics"], dtype=np.float64)
    for file_index in range(progress["file"], len(files)):
        skip = progress["chunk"] if file_index == progress["file"] else 0
        for chunk_index, (x, targets, topology) in enumerate(
            batches(files[file_index], history, batch, min_stars)
        ):
            if chunk_index < skip:
                continue
            # Keep all samples, including the final short batch, at a fixed JIT shape.
            n = len(targets["action"])
            ids = np.arange(n)
            if update is not None:
                np.random.default_rng([seed, file_index, chunk_index]).shuffle(ids)
            ids = np.pad(ids, (0, batch - n), mode="edge")
            x, targets = jax.tree.map(lambda a, ids=ids: a[ids], (x, targets))
            targets["weight"] = targets["weight"] * (jnp.arange(batch) < n)
            if not bool(jnp.any(targets["weight"])):
                continue
            if update is None:
                row = validate(model, x, targets, topology)
            else:
                model, state, loss, row, gradient = update(
                    model, state, x, targets, topology
                )
                if not bool(jnp.isfinite(loss) & jnp.isfinite(gradient)):
                    raise FloatingPointError("Nonfinite supervised update")
            row = np.asarray(row)
            metrics += row
            progress = {
                "file": file_index,
                "chunk": chunk_index + 1,
                "batches": progress["batches"] + 1,
                "metrics": metrics.tolist(),
            }
            if callback is not None:
                callback(model, state, progress)
            if max_batches and progress["batches"] >= max_batches:
                return model, state, report(metrics), progress, False
        progress = dict(progress, file=file_index + 1, chunk=0)
    return model, state, report(metrics), progress, True


def dataset_signature(folder):
    """Portable identity of the prepared manifest and training/validation inventory."""
    digest = hashlib.sha256()
    manifest = folder / "manifest.jsonl"
    if manifest.exists():
        digest.update(manifest.read_bytes())
    for split in ("train", "validation"):
        for path in sorted((folder / split).glob("*.npz")):
            digest.update(f"{split}/{path.name}:{path.stat().st_size}\n".encode())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("runs/supervised"))
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--init", type=Path)
    start.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--auxiliary", type=float, default=0.05)
    parser.add_argument("--min-stars", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1012)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--max-updates",
        type=int,
        default=0,
        help="Stop at this total optimizer-step count; 0 is unlimited",
    )
    parser.add_argument(
        "--validation-batches",
        type=int,
        default=0,
        help="Bound validation for smoke checks only; 0 uses all games",
    )
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.batch < 1
        or args.learning_rate <= 0
        or args.auxiliary < 0
        or min(args.save_every, args.log_every) < 1
        or min(args.max_updates, args.validation_batches, args.min_stars, args.seed) < 0
    ):
        parser.error(
            "epochs/batch/rate must be positive; auxiliary must be nonnegative"
        )
    training = sorted((args.data / "train").glob("*.npz"))
    validation = sorted((args.data / "validation").glob("*.npz"))
    if not training or not validation:
        parser.error(
            "Both train and validation games are required; prepare more replays"
        )
    settings = {
        name: getattr(args, name)
        for name in (
            "batch",
            "learning_rate",
            "auxiliary",
            "min_stars",
            "seed",
            "validation_batches",
        )
    }
    settings["dataset"] = dataset_signature(args.data)
    metadata = {}
    if args.resume or args.init:
        model, metadata = load(args.resume or args.init)
    else:
        model = Network(key=jax.random.PRNGKey(args.seed))
    if args.resume and metadata.get("settings") != settings:
        parser.error(
            "Resume requires the same prepared data and training settings; use --init for a new run"
        )
    if not args.resume and (args.output / "checkpoint.zip").exists():
        parser.error(
            "Output already contains a checkpoint; use --resume or another --output"
        )
    tx = optimizer(args.learning_rate)
    state = tx.init(eqx.filter(model, eqx.is_inexact_array))
    if args.resume:
        state = restore_optimizer(args.resume, state)
    update = make_update(tx, args.auxiliary)
    cursor = (
        metadata["cursor"]
        if args.resume
        else {"epoch": 0, "updates": 0, "progress": None}
    )
    best = metadata.get("best") if args.resume else None
    args.output.mkdir(parents=True, exist_ok=True)

    def log(row):
        text = json.dumps(row)
        print(text, flush=True)
        with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(text + "\n")

    def checkpoint(model, state, cursor):
        save(
            args.output / "checkpoint.zip",
            model,
            {
                "phase": "supervised",
                "settings": settings,
                "cursor": cursor,
                "best": best,
            },
            state,
        )

    log(
        {
            "event": "start",
            "phase": "supervised",
            "devices": [str(d) for d in jax.devices()],
            "parameters": sum(a.size for a in jax.tree.leaves(model)),
            "train_games": len(training),
            "validation_games": len(validation),
            "settings": settings,
            "cursor": cursor,
        }
    )
    for index in range(cursor["epoch"], args.epochs):
        if args.max_updates and cursor["updates"] >= args.max_updates:
            break
        start = time.monotonic()
        previous = cursor["progress"]
        base = cursor["updates"] - (previous["batches"] if previous else 0)
        log({"event": "train_start", "epoch": index + 1})

        def progress(model, state, position, index=index, base=base, start=start):
            current = {
                "epoch": index,
                "updates": base + position["batches"],
                "progress": position,
            }
            if current["updates"] % args.save_every == 0:
                checkpoint(model, state, current)
            if position["batches"] == 1 or current["updates"] % args.log_every == 0:
                log(
                    dict(
                        phase="train",
                        epoch=index + 1,
                        updates=current["updates"],
                        game=position["file"] + 1,
                        games=len(training),
                        seconds=time.monotonic() - start,
                        **report(position["metrics"]),
                    )
                )

        model, state, train_metrics, position, complete = epoch(
            model,
            training,
            model.config.history,
            args.batch,
            args.min_stars,
            update,
            state,
            args.seed + index,
            progress=previous,
            callback=progress,
            max_batches=max(0, args.max_updates - base) if args.max_updates else 0,
        )
        cursor = {
            "epoch": index,
            "updates": base + position["batches"],
            "progress": position,
        }
        checkpoint(model, state, cursor)
        if not complete:
            log(
                {
                    "event": "paused",
                    "updates": cursor["updates"],
                    "checkpoint": str(args.output / "checkpoint.zip"),
                }
            )
            break
        log({"event": "validation_start", "epoch": index + 1})
        _, _, val, _, validated_all = epoch(
            model,
            validation,
            model.config.history,
            args.batch,
            args.min_stars,
            callback=lambda _m, _s, p: (
                log(
                    {
                        "phase": "validation",
                        "batches": p["batches"],
                        "game": p["file"] + 1,
                        "games": len(validation),
                    }
                )
                if p["batches"] % args.log_every == 0
                else None
            ),
            max_batches=args.validation_batches,
        )
        result = {
            "phase": "supervised",
            "epoch": index + 1,
            "train": train_metrics,
            "validation": val,
            "validation_complete": validated_all,
            "updates": cursor["updates"],
            "seconds": time.monotonic() - start,
        }
        log(result)
        if best is None or val["action_nll"] < best:
            best = val["action_nll"]
            save(args.output / "policy.zip", model, result)
        cursor = {"epoch": index + 1, "updates": cursor["updates"], "progress": None}
        checkpoint(model, state, cursor)


if __name__ == "__main__":
    main()
