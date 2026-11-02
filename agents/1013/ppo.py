"""PPO orchestration, immutable historical opponents, and resumable iteration saves.

Every environment contributes one learner trajectory. Evaluation and opponent
decoders never supply oracle inputs to the actor. Checkpoints include unfinished
episodes, PRNGs, optimizer moments and the exact frozen opponent bank.
"""

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from environment import Environment, Run
from model import precision
from train import auxiliary, load, opponent_loss, optimizer, restore, save


def emit(record):
    print(json.dumps(record), flush=True)


class Opponents:
    """Current rollout policy and its two predecessors; active slots are immutable."""

    def __init__(self, model, recent=3, spare=3):
        self.recent = recent
        self.capacity = recent + spare
        members = [model] * self.capacity
        arrays, static = zip(*(eqx.partition(m, eqx.is_array) for m in members))
        self.bank = eqx.combine(
            jax.tree.map(lambda *a: jnp.stack(a), *arrays), static[0]
        )
        self.versions = [0] + [None] * (self.capacity - 1)
        self.roster = [0]

    def eligible(self):
        return jnp.zeros(self.capacity, bool).at[jnp.array(self.roster)].set(True)

    def add(self, model, version, active):
        """Store the new version without overwriting any active or eligible opponent."""
        busy = set(np.asarray(active).tolist())
        free = [
            i for i in range(self.capacity) if i not in busy and i not in self.roster
        ]
        if not free:
            raise RuntimeError(
                "Opponent retirement slots exhausted; check the episode horizon"
            )
        slot = free[0]
        arrays, static = eqx.partition(self.bank, eqx.is_array)
        value = eqx.filter(model, eqx.is_array)
        self.bank = eqx.combine(
            jax.tree.map(lambda a, b: a.at[slot].set(b), arrays, value), static
        )
        self.versions[slot] = version
        self.roster = (self.roster + [slot])[-self.recent :]
        return True

    def metadata(self):
        return {
            "recent": self.recent,
            "versions": self.versions,
            "roster": self.roster,
            "sampling_versions": [self.versions[i] for i in self.roster],
        }


def objective(
    model,
    data,
    topology,
    entropy_coefficient,
    auxiliary_weight,
    opponent_weight,
    clip=0.2,
):
    output = jax.vmap(model, in_axes=(0, None))(data.inputs, topology)
    logp = jax.nn.log_softmax(output.logits)
    current = jnp.take_along_axis(logp, data.actions[:, None], -1)[:, 0]
    delta = current - data.old_logp
    ratio = jnp.exp(delta)
    policy = -jnp.mean(
        jnp.minimum(
            ratio * data.advantage, jnp.clip(ratio, 1 - clip, 1 + clip) * data.advantage
        )
    )
    value = jnp.mean((output.value - data.returns) ** 2)
    entropy = -jnp.mean(jnp.sum(jnp.exp(logp) * logp, -1))
    kl = jnp.mean((ratio - 1) - delta)
    aux = jnp.mean(auxiliary(output, data.inputs, data.targets))
    command, reconstruction = opponent_loss(output, data.inputs, data.targets)
    command, reconstruction = command.mean(), reconstruction.mean()
    agreement = (output.enemy_action.argmax(-1) == data.targets["enemy_action"]).mean()
    loss = (
        policy
        + 0.5 * value
        - entropy_coefficient * entropy
        + auxiliary_weight * aux
        + opponent_weight * (command + reconstruction)
    )
    return loss, jnp.array(
        [policy, value, entropy, kl, aux, command, reconstruction, agreement]
    )


def make_update(
    tx, entropy, auxiliary_weight=0.02, opponent_weight=0.02, microbatch=64
):
    @eqx.filter_jit(donate="all-except-first")
    def update(batch_and_graph, model, state):
        data, topology = batch_and_graph
        size = len(data.actions)
        chunk = math.gcd(size, min(size, microbatch))
        batches = jax.tree.map(lambda x: x.reshape((-1, chunk, *x.shape[1:])), data)
        zeros = jax.tree.map(jnp.zeros_like, eqx.filter(model, eqx.is_inexact_array))

        def accumulate(carry, part):
            total, gradients = carry
            (_, metrics), gradient = eqx.filter_value_and_grad(objective, has_aux=True)(
                model,
                part.unpack(),
                topology,
                entropy,
                auxiliary_weight,
                opponent_weight,
            )
            gradients = jax.tree.map(
                lambda a, b: a + b * (chunk / size), gradients, gradient
            )
            return (total + metrics * (chunk / size), gradients), None

        (metrics, gradient), _ = jax.lax.scan(
            accumulate, (jnp.zeros(8), zeros), batches
        )
        norm = optax.tree.norm(gradient)
        change, state = tx.update(
            gradient, state, eqx.filter(model, eqx.is_inexact_array)
        )
        return eqx.apply_updates(model, change), state, metrics, norm

    return update


def minibatches(data, permutation, batch):
    """One host prefetch overlaps temporal gathering with accelerator execution."""
    with ThreadPoolExecutor(max_workers=1) as worker:
        ranges = [permutation[i : i + batch] for i in range(0, len(permutation), batch)]
        future = worker.submit(data.read, ranges[0])
        for i, indices in enumerate(ranges):
            part = future.result()
            if i + 1 < len(ranges):
                future = worker.submit(data.read, ranges[i + 1])
            yield jax.device_put(part), len(indices)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    start = result.add_mutually_exclusive_group()
    start.add_argument("--init", type=Path, default=None)
    start.add_argument("--resume", type=Path)
    result.add_argument(
        "--reference",
        type=Path,
        default=Path("initial/original.zip"),
        help="Supervised model for evaluation only",
    )
    result.add_argument("--output", type=Path, default=Path("runs/ppo"))
    result.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Total iterations, including resumed iterations",
    )
    for name, default in (
        ("side", 24),
        ("distance", 17),
        ("environments", 512),
        ("pool", 4096),
        ("steps", 512),
        ("horizon", 1024),
        ("batch", 1024),
        ("microbatch", 64),
        ("inference-batch", 64),
        ("rollout-chunk", 16),
        ("epochs", 1),
        ("evaluate", 10),
        ("evaluation-games", 128),
        ("evaluation-environments", 16),
        ("seed", 1013),
    ):
        result.add_argument("--" + name, type=int, default=default)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--entropy", type=float, default=0.01)
    result.add_argument("--auxiliary", type=float, default=0.02)
    result.add_argument("--opponent-loss", type=float, default=0.02)
    result.add_argument("--shaping", type=float, default=1.0)
    result.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto")
    return result


def main():
    cli = parser()
    args = cli.parse_args()
    for name in (
        "iterations",
        "environments",
        "pool",
        "steps",
        "horizon",
        "batch",
        "microbatch",
        "inference_batch",
        "rollout_chunk",
        "epochs",
    ):
        if getattr(args, name) < 1:
            cli.error(name + " must be positive")
    if args.side < 4 or not 1 <= args.distance <= 2 * (args.side - 1):
        cli.error("Choose side >= 4 and a feasible distance")
    if args.batch % args.microbatch or args.environments * args.steps % args.batch:
        cli.error("batch must divide learner samples, and microbatch must divide batch")
    if (
        args.evaluate < 0
        or args.evaluation_games < 2
        or args.evaluation_games % 2
        or args.evaluation_environments < 2
        or args.evaluation_environments % 2
    ):
        cli.error(
            "Require a nonnegative evaluation interval and positive even evaluation counts"
        )
    if (
        not all(
            math.isfinite(x) and x >= 0
            for x in (args.entropy, args.auxiliary, args.opponent_loss, args.shaping)
        )
        or not math.isfinite(args.learning_rate)
        or not args.learning_rate > 0
    ):
        cli.error("Invalid loss or learning-rate setting")
    config = Run(
        **{
            key: getattr(args, key)
            for key in (
                "side",
                "distance",
                "environments",
                "pool",
                "steps",
                "horizon",
                "shaping",
            )
        }
    )
    # Execution knobs can change on resume; statistical training settings cannot.
    settings = {
        key: getattr(args, key)
        for key in (
            "batch",
            "epochs",
            "learning_rate",
            "entropy",
            "auxiliary",
            "opponent_loss",
            "seed",
        )
    }
    bf16 = args.precision == "bf16" or (
        args.precision == "auto" and jax.default_backend() == "gpu"
    )
    settings["precision"] = "bf16" if bf16 else "fp32"
    precision(bf16)
    source = args.resume or args.init or Path("initial/policy.zip")
    model, metadata = load(source)
    if args.resume:
        if (
            metadata.get("run") != asdict(config)
            or metadata.get("settings") != settings
        ):
            cli.error(
                "Resume requires the checkpoint's training settings; runtime batch sizes may change"
            )
        iteration0 = metadata["iteration"]
        if args.output.resolve() != args.resume.parent.resolve():
            cli.error(
                "Resume in the original output directory to preserve snapshot references"
            )
    else:
        if args.output.exists() and any(args.output.iterdir()):
            cli.error(
                "Choose a new output directory or explicitly --resume checkpoint.zip"
            )
        iteration0 = 0
    league = Opponents(model, spare=math.ceil(args.horizon / args.steps) + 1)
    if not args.resume:
        reference = load(args.reference)[0]
        save(
            args.output / "references" / "original.zip",
            reference,
            {"phase": "evaluation_reference"},
        )
        save(
            args.output / "references" / "initial.zip",
            model,
            {"phase": "evaluation_reference", "iteration": 0},
        )
        save(
            args.output / "snapshots" / "iteration-00000000.zip",
            model,
            {"phase": "ppo", "iteration": 0},
        )
        del reference
    emit(
        {
            "event": "start",
            "devices": [str(x) for x in jax.devices()],
            "run": asdict(config),
            "settings": settings,
            "parameters": sum(
                x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array))
            ),
            "learner_samples_per_iteration": args.environments * args.steps,
            "frozen_samples_used_for_ppo": 0,
            "microbatch": args.microbatch,
            "inference_batch": args.inference_batch,
            "save_every": 1,
            "startup_evaluation": False,
            "iteration": iteration0,
        }
    )
    env = Environment(
        config,
        model.config.history,
        args.seed,
        args.inference_batch,
        args.rollout_chunk,
    )
    tx = optimizer(args.learning_rate)
    state = tx.init(eqx.filter(model, eqx.is_inexact_array))
    rng = np.random.default_rng(args.seed)
    if args.resume:
        state = restore(source, "optimizer", state)
        env.carry = restore(source, "runtime", env.carry)
        league.bank = restore(source, "opponents", league.bank)
        league.versions, league.roster = (
            metadata["league"]["versions"],
            metadata["league"]["roster"],
        )
        rng.bit_generator.state = metadata["numpy_rng"]
    else:
        initial = {
            "phase": "ppo",
            "run": asdict(config),
            "settings": settings,
            "iteration": 0,
            "league": league.metadata(),
            "numpy_rng": rng.bit_generator.state,
        }
        save(
            args.output / "checkpoint.zip",
            model,
            initial,
            state,
            env.carry,
            league.bank,
        )
        save(args.output / "policy.zip", model, initial)
    update = make_update(
        tx, args.entropy, args.auxiliary, args.opponent_loss, args.microbatch
    )
    emit({"event": "buffer", "storage": "cpu", "gib": env.buffer.nbytes / 2**30})
    evaluation = None
    for iteration in range(iteration0 + 1, args.iterations + 1):
        started = time.monotonic()
        emit({"phase": "rollout", "event": "start", "iteration": iteration})
        data, rollout = env.collect(model, league.bank, league.eligible(), emit)
        rollout_seconds = time.monotonic() - started
        data.advantage -= data.advantage.mean()
        data.advantage /= max(float(data.advantage.std()), 1e-6)
        passed = data.actions == config.side**2 * 8
        rows, norms = [], []
        for epoch in range(args.epochs):
            permutation = rng.permutation(data.count)
            for offset, (part, size) in enumerate(
                minibatches(data, permutation, args.batch)
            ):
                model, state, metrics, gradient = update(
                    (part, env.topology), model, state
                )
                rows.append(metrics)
                norms.append(gradient)
                if len(rows) % 16 == 0 or (offset + 1) * args.batch == data.count:
                    # One synchronization per progress block, not per microbatch.
                    measured = np.asarray(jnp.stack(rows[-16:]))
                    gradients = np.asarray(jnp.stack(norms[-16:]))
                    if (
                        not np.isfinite(measured).all()
                        or not np.isfinite(gradients).all()
                    ):
                        raise FloatingPointError(
                            "Nonfinite PPO update; the previous iteration checkpoint is intact"
                        )
                    emit(
                        {
                            "phase": "update",
                            "iteration": iteration,
                            "epoch": epoch + 1,
                            "optimized": epoch * data.count + (offset + 1) * args.batch,
                            "total": args.epochs * data.count,
                            "seconds": time.monotonic() - started - rollout_seconds,
                        }
                    )
        measured = np.asarray(jnp.stack(rows)).mean(0).tolist()
        gradients = np.asarray(jnp.stack(norms))
        update_seconds = time.monotonic() - started - rollout_seconds
        league.add(model, iteration, env.carry.opponent)
        emit(dict(event="opponent_snapshot", iteration=iteration, **league.metadata()))
        log = dict(
            phase="ppo",
            run=asdict(config),
            settings=settings,
            iteration=iteration,
            **rollout,
            **dict(
                zip(
                    (
                        "policy_loss",
                        "value_loss",
                        "entropy",
                        "kl",
                        "auxiliary_loss",
                        "opponent_action_nll",
                        "opponent_view_loss",
                        "opponent_action_accuracy",
                    ),
                    measured,
                )
            ),
            gradient_before_clip=float(gradients.mean()),
            gradient_before_clip_max=float(gradients.max()),
            pass_fraction=float(passed.mean()),
            optional_pass_fraction=float(
                (passed & data.movable).sum() / max(data.movable.sum(), 1)
            ),
            player_samples=data.count,
            environment_samples=iteration * data.count,
            optimizer_steps=len(rows),
            rollout_seconds=rollout_seconds,
            update_seconds=update_seconds,
            league=league.metadata(),
            numpy_rng=rng.bit_generator.state,
        )
        saving = time.monotonic()
        # Save before evaluation so a slow match or interruption cannot lose updates.
        save(args.output / "checkpoint.zip", model, log, state, env.carry, league.bank)
        save(args.output / "policy.zip", model, log)
        snapshot = args.output / "snapshots" / f"iteration-{iteration:08d}.zip"
        save(snapshot, model, log)
        log.update(
            save_seconds=time.monotonic() - saving, seconds=time.monotonic() - started
        )
        emit(log)
        with (args.output / "training.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(log) + "\n")
        if args.evaluate and (
            iteration % args.evaluate == 0 or iteration == args.iterations
        ):
            from evaluate import Evaluator, file_hash

            if evaluation is None:
                evaluation = Evaluator(
                    config,
                    model.config.history,
                    args.evaluation_games,
                    args.evaluation_environments,
                    args.seed + 1000000,
                    excluded=env.pool,
                )
            for name, path in [
                ("original", args.output / "references" / "original.zip"),
                ("initial", args.output / "references" / "initial.zip"),
                ("baseline", None),
            ]:
                result = evaluation.matches(
                    model, load(path)[0] if path else None, name
                )
                result.update(
                    iteration=iteration,
                    checkpoint_sha256=file_hash(snapshot),
                    opponent_sha256=file_hash(
                        path or Path(__file__).with_name("baseline.py")
                    ),
                )
                folder = args.output / "evaluation" / f"iteration-{iteration:08d}"
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"{name}.json").write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )
                summary = {
                    key: value for key, value in result.items() if key != "records"
                }
                with (args.output / "evaluation.jsonl").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(json.dumps(summary) + "\n")
                emit(summary)


if __name__ == "__main__":
    main()
