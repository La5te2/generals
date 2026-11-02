"""Q-boosted PPO, an independently trained critic, and an admitted opponent pool.

Every environment contributes one learner trajectory. Evaluation and opponent
decoders never supply oracle inputs to the actor. Checkpoints include unfinished
episodes, PRNGs, optimizer moments and the exact frozen opponent bank.
"""

import argparse
import hashlib
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
from critic import Critic
from critic import make_update as make_critic_update
from environment import Environment, Run
from model import Config, precision
from economy import HORIZONS, prediction_loss
from gathering import FIELDS as GATHERING_FIELDS, losses as gathering_losses
from gathering import prepare as prepare_gathering
from train import load, opponent_loss, optimizer, restore, save, state_losses
from rewards import POTENTIAL_SHA256
from potential import SCHEMA as POTENTIAL_SCHEMA


def emit(record):
    print(json.dumps(record), flush=True)


class Opponents:
    """Approved frozen policies; retiring slots remain immutable while in use."""

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

    def member(self, slot):
        arrays, static = eqx.partition(self.bank, eqx.is_array)
        return eqx.combine(jax.tree.map(lambda x: x[slot], arrays), static)

    def retirement(self, results):
        """Prefer the oldest beaten member, then the oldest member overall."""
        if len(results) != len(self.roster):
            raise ValueError("Admission results must match the current roster")
        if len(self.roster) < self.recent:
            return None
        beaten = [s for s, r in zip(self.roster, results) if r["wins"] > r["losses"]]
        return (beaten or self.roster)[0]

    def add(self, model, version, active, retire=None):
        """Store the new version without overwriting any active or eligible opponent."""
        busy = set(np.asarray(active).tolist())
        if len(self.roster) >= self.recent and retire not in self.roster:
            raise ValueError("Full pool admission requires a retirement decision")
        if retire is not None and retire not in self.roster:
            raise ValueError("Retiring member must belong to the roster")
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
        self.roster = [s for s in self.roster if s != retire] + [slot]
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
    value_auxiliary=0.5,
    economy_weight=0.02,
    checkpoint="full",
    gathering_weight=0.02,
):
    queries = {k: data.targets["gathering_" + k] for k in GATHERING_FIELDS}
    output = jax.vmap(lambda x, a, q: model(x, topology, a, checkpoint, q))(
        data.inputs, data.actions, queries
    )
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
    owner, army, general = (
        x.mean() for x in state_losses(output, data.inputs, data.targets)
    )
    aux = owner + army + general
    command = opponent_loss(output, data.inputs, data.targets).mean()
    agreement = (output.enemy_action.argmax(-1) == data.targets["enemy_action"]).mean()
    economic = prediction_loss(
        output.economy, data.targets["economy"], data.targets["economy_valid"]
    ).mean()
    economic_errors = optax.huber_loss(output.economy, data.targets["economy"]).mean(-1)
    economic_valid = data.targets["economy_valid"]
    gathering = gathering_losses(output.gathering, data.targets)
    loss = (
        policy
        + value_auxiliary * value
        - entropy_coefficient * entropy
        + auxiliary_weight * aux
        + opponent_weight * command
        + economy_weight * economic
        + gathering_weight * gathering[0]
    )
    return loss, jnp.concatenate(
        (
            jnp.array(
                [
                    policy,
                    value,
                    entropy,
                    kl,
                    aux,
                    command,
                    agreement,
                    owner,
                    army,
                    general,
                    economic,
                ]
            ),
            (economic_errors * economic_valid).mean(0),
            economic_valid.mean(0),
            gathering,
        )
    )


def make_update(
    tx,
    entropy,
    auxiliary_weight=0.02,
    opponent_weight=0.02,
    microbatch=64,
    value_auxiliary=0.5,
    economy_weight=0.02,
    checkpoint="full",
    gathering_weight=0.02,
):
    @eqx.filter_jit(donate="all-except-first")
    def update(batch_and_graph, model, state):
        data, topology, timeline = batch_and_graph
        size = len(data.actions)
        chunk = math.gcd(size, min(size, microbatch))
        batches = jax.tree.map(lambda x: x.reshape((-1, chunk, *x.shape[1:])), data)
        zeros = jax.tree.map(jnp.zeros_like, eqx.filter(model, eqx.is_inexact_array))

        def accumulate(carry, part):
            total, gradients = carry
            (_, metrics), gradient = eqx.filter_value_and_grad(objective, has_aux=True)(
                model,
                part.unpack(timeline),
                topology,
                entropy,
                auxiliary_weight,
                opponent_weight,
                value_auxiliary=value_auxiliary,
                economy_weight=economy_weight,
                checkpoint=checkpoint,
                gathering_weight=gathering_weight,
            )
            gradients = jax.tree.map(
                lambda a, b: a + b * (chunk / size), gradients, gradient
            )
            return (total + metrics * (chunk / size), gradients), None

        (metrics, gradient), _ = jax.lax.scan(
            accumulate, (jnp.zeros(15 + 2 * len(HORIZONS)), zeros), batches
        )
        norm = optax.tree.norm(gradient)
        change, state = tx.update(
            gradient, state, eqx.filter(model, eqx.is_inexact_array)
        )
        return eqx.apply_updates(model, change), state, metrics, norm

    return update


def minibatches(data, permutation, batch):
    """Prefetch transition data and indices while shared history stays on device."""
    with ThreadPoolExecutor(max_workers=1) as worker:
        ranges = [permutation[i : i + batch] for i in range(0, len(permutation), batch)]
        future = worker.submit(data.read, ranges[0], indexed=True)
        for i, indices in enumerate(ranges):
            part = future.result()
            if i + 1 < len(ranges):
                future = worker.submit(data.read, ranges[i + 1], indexed=True)
            yield jax.device_put(part), len(indices)


def make_kl_probe(microbatch):
    """Monitor old-to-current sampled KL on a fixed rollout subset after updates."""

    @eqx.filter_jit
    def measure(model, data, topology, timeline=None):
        size = len(data.actions)
        chunk = math.gcd(size, min(size, microbatch))
        batches = jax.tree.map(lambda x: x.reshape((-1, chunk, *x.shape[1:])), data)

        def part(batch):
            sample = batch.unpack(timeline)
            logits = jax.vmap(model, in_axes=(0, None))(sample.inputs, topology).logits
            logp = jax.nn.log_softmax(logits)
            current = jnp.take_along_axis(logp, sample.actions[:, None], -1)[:, 0]
            delta = current - sample.old_logp
            return jnp.mean(jnp.expm1(delta) - delta)

        return jax.lax.map(part, batches).mean()

    return measure


def promotion_passes(results, threshold):
    """Uniform pool mixture; unresolved games contribute zero wins."""
    rate = float(np.mean([r["win_rate"] for r in results]))
    return rate >= threshold, rate


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    source = result.add_mutually_exclusive_group()
    source.add_argument("--resume", type=Path)
    source.add_argument("--init", type=Path, help="Override the bundled initial/r1.zip policy for a fresh run")
    result.add_argument("--width", type=int, default=128)
    result.add_argument("--depth", type=int, default=8)
    result.add_argument("--heads", type=int, default=4)
    result.add_argument("--period", type=int, default=32, help="Rolling read interval, a divisor of 512")
    result.add_argument("--recent", type=int, default=32, help="Ordered recent frames read from the shared archive (1 to 512)")
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
        ("critic-epochs", 2),
        ("evaluation-environments", 16),
        ("promotion-games", 64),
        ("fixed-games", 128),
        ("gathering-samples", 4096),
        ("kl-every", 16),
        ("kl-samples", 2048),
        ("seed", 1015),
    ):
        result.add_argument("--" + name, type=int, default=default)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--entropy", type=float, default=0.01)
    result.add_argument("--auxiliary", type=float, default=0.02)
    result.add_argument("--opponent-loss", type=float, default=0.02)
    result.add_argument("--value-auxiliary", type=float, default=0.5)
    result.add_argument("--economy-loss", type=float, default=0.02)
    result.add_argument("--gathering-loss", type=float, default=0.02)
    result.add_argument("--shaping", type=float, default=1.0)
    result.add_argument("--promotion-threshold", type=float, default=0.5)
    result.add_argument("--target-kl", type=float, default=0.03)
    result.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto")
    result.add_argument(
        "--critic-microbatch",
        type=int,
        default=None,
        help="Critic samples per device batch, default gcd(batch, 128)",
    )
    result.add_argument(
        "--activation-checkpoint",
        choices=("full", "blocks", "none"),
        default="full",
        help="Recompute all branches, graph blocks, or retain activations",
    )
    return result


def main():
    cli = parser()
    args = cli.parse_args()
    if not args.resume:
        args.init = args.init or Path(__file__).resolve().parent / "initial" / "r1.zip"
        if not args.init.is_file():
            cli.error(f"Required initializer missing: {args.init}. Upload initial/r1.zip before starting training.")
    try:
        network_config = Config(args.width, args.depth, args.heads, args.period, args.recent)
    except ValueError as error:
        cli.error(str(error))
    if args.critic_microbatch is None:
        args.critic_microbatch = math.gcd(args.batch, 128)
    for name in (
        "iterations",
        "environments",
        "pool",
        "steps",
        "horizon",
        "batch",
        "microbatch",
        "critic_microbatch",
        "inference_batch",
        "rollout_chunk",
        "epochs",
        "critic_epochs",
        "kl_every",
        "kl_samples",
    ):
        if getattr(args, name) < 1:
            cli.error(name + " must be positive")
    if args.promotion_games < 2 or args.promotion_games % 2:
        cli.error("promotion-games must be positive and even")
    if args.fixed_games < 2 or args.fixed_games % 2:
        cli.error("fixed-games must be positive and even")
    if args.gathering_samples < 1:
        cli.error("gathering-samples must be positive")
    if (
        not 0 < args.promotion_threshold <= 1
        or not math.isfinite(args.target_kl)
        or args.target_kl <= 0
    ):
        cli.error("Require 0 < promotion-threshold <= 1 and positive finite target-kl")
    if args.side < 4 or not 1 <= args.distance <= 2 * (args.side - 1):
        cli.error("Choose side >= 4 and a feasible distance")
    if args.batch % args.microbatch or args.environments * args.steps % args.batch:
        cli.error("batch must divide learner samples, and microbatch must divide batch")
    if args.batch % args.critic_microbatch:
        cli.error("critic-microbatch must divide batch")
    if (
        args.evaluation_environments < 2
        or args.evaluation_environments % 2
    ):
        cli.error(
            "Require positive even evaluation-environments"
        )
    if (
        not all(
            math.isfinite(x) and x >= 0
            for x in (
                args.entropy,
                args.auxiliary,
                args.opponent_loss,
                args.value_auxiliary,
                args.economy_loss,
                args.gathering_loss,
                args.shaping,
            )
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
            "critic_epochs",
            "learning_rate",
            "entropy",
            "auxiliary",
            "opponent_loss",
            "value_auxiliary",
            "economy_loss",
            "gathering_loss",
            "gathering_samples",
            "fixed_games",
            "seed",
            "target_kl",
            "kl_every",
            "kl_samples",
            "promotion_games",
            "promotion_threshold",
        )
    }
    bf16 = args.precision == "bf16" or (
        args.precision == "auto" and jax.default_backend() == "gpu"
    )
    settings["precision"] = "bf16" if bf16 else "fp32"
    settings["estimator"] = "reference-policy-q-boosting-v1"
    settings["league_protocol"] = "seeded-three-beaten-first-v1"
    settings["network"] = asdict(network_config)
    settings["potential_sha256"] = POTENTIAL_SHA256
    settings["potential_schema"] = POTENTIAL_SCHEMA
    from evaluate import Evaluator, file_hash
    reference_path = args.output / "references" / "r1.zip"
    reference, reference_metadata = load(reference_path if args.resume else args.init)
    if reference.config != network_config:
        cli.error("The initializer/reference must match --width, --depth, --heads, --period and --recent")
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(eqx.filter(reference, eqx.is_array)):
        digest.update(np.asarray(leaf).tobytes())
    settings["fixed_opponent_parameters_sha256"] = digest.hexdigest()
    settings["fixed_evaluation_seed"] = args.seed + 1000000
    precision(bf16)
    source = args.resume
    # PPO donates learner buffers. The immutable reference owns separate arrays.
    model, metadata = load(source) if source else (jax.tree.map(lambda x: x.copy(), reference), {})
    if reference.config != model.config:
        cli.error("The reference and learner must share the network configuration")
    if args.resume:
        if (
            metadata.get("workflow") != 1015
            or metadata.get("run") != asdict(config)
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
    league = Opponents(reference, spare=math.ceil(args.horizon / args.steps) + 1)
    if not args.resume:
        league.versions[0] = "r1"
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        initial_metadata = {
            "phase": "initial_seeded_opponent", "name": "r1",
            "source_sha256": file_hash(args.init),
            "source_metadata": reference_metadata,
        }
        save(reference_path, reference, initial_metadata)
        for name in ("r2", "r3"):
            league.add(reference, name, [])
            save(
                args.output / "references" / f"{name}.zip", reference,
                dict(initial_metadata, name=name),
            )
        save(
            args.output / "snapshots" / "iteration-00000000.zip",
            model,
            {"phase": "ppo", "iteration": 0},
        )
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
            "critic_microbatch": args.critic_microbatch,
            "activation_checkpoint": args.activation_checkpoint,
            "inference_batch": args.inference_batch,
            "save_every": 1,
            "startup_evaluation": False,
            "iteration": iteration0,
            "initialization": "resume" if args.resume else str(args.init),
        }
    )
    env = Environment(
        config,
        args.seed,
        args.inference_batch,
        args.rollout_chunk,
    )
    tx = optimizer(args.learning_rate)
    state = tx.init(eqx.filter(model, eqx.is_inexact_array))
    critic = Critic(model.config.period, jax.random.PRNGKey(args.seed + 1015), recent=model.config.recent)
    critic_state = tx.init(eqx.filter(critic, eqx.is_inexact_array))
    rng = np.random.default_rng(args.seed)
    if args.resume:
        state = restore(source, "optimizer", state)
        critic = restore(source, "critic", critic)
        critic_state = restore(source, "critic_optimizer", critic_state)
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
            "workflow": 1015,
            "run": asdict(config),
            "settings": settings,
            "iteration": 0,
            "league": league.metadata(),
            "numpy_rng": rng.bit_generator.state,
            "evaluation_pending": False,
        }
        env.carry = env.carry._replace(
            opponent=jnp.array(league.roster)[
                (jnp.arange(args.environments) // 2) % len(league.roster)
            ]
        )
        save(
            args.output / "checkpoint.zip",
            model,
            initial,
            state,
            env.carry,
            league.bank,
            critic,
            critic_state,
        )
        save(args.output / "policy.zip", model, initial)
    update = make_update(
        tx,
        args.entropy,
        args.auxiliary,
        args.opponent_loss,
        args.microbatch,
        args.value_auxiliary,
        args.economy_loss,
        args.activation_checkpoint,
        args.gathering_loss,
    )
    measure_kl = make_kl_probe(args.microbatch)
    update_critic = make_critic_update(
        tx, args.critic_microbatch, args.activation_checkpoint
    )
    emit({"event": "buffer", "storage": "cpu", "gib": env.buffer.nbytes / 2**30})
    fixed_evaluator = None

    def assess(iteration, log):
        """Commit pool changes only after all promotion matches finish."""
        nonlocal fixed_evaluator

        snapshot = args.output / "snapshots" / f"iteration-{iteration:08d}.zip"
        folder = args.output / "evaluation" / f"iteration-{iteration:08d}"
        folder.mkdir(parents=True, exist_ok=True)

        def record(evaluator, rival, name, kind):
            result = evaluator.matches(model, rival, name)
            opponent_path = (
                args.output / "references" / f"{name}.zip"
                if name in ("r1", "r2", "r3")
                else args.output / "snapshots" / f"iteration-{int(name):08d}.zip"
            )
            result.update(
                iteration=iteration,
                kind=kind,
                checkpoint_sha256=file_hash(snapshot),
                opponent_sha256=file_hash(opponent_path),
            )
            path = folder / f"{kind}-{name}.json"
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            summary = {k: v for k, v in result.items() if k != "records"}
            emit(summary)
            with (args.output / "evaluation.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(summary) + "\n")
            return result

        if fixed_evaluator is None:
            fixed_evaluator = Evaluator(
                config, args.fixed_games,
                args.evaluation_environments, settings["fixed_evaluation_seed"],
                greedy=True, excluded=env.pool,
            )
        fixed_result = record(fixed_evaluator, reference, "r1", "fixed")
        log["fixed_evaluation"] = {
            k: v for k, v in fixed_result.items() if k != "records"
        }
        gate = Evaluator(
            config,
            args.promotion_games,
            args.evaluation_environments,
            args.seed + 2000000 + iteration,
            greedy=True,
            excluded=jax.tree.map(
                lambda a, b: jnp.concatenate((a, b)), env.pool, fixed_evaluator.pool
            ),
        )
        versions = [league.versions[s] for s in league.roster]
        results = [
            record(gate, league.member(s), str(league.versions[s]), "promotion")
            for s in league.roster
        ]
        accepted, rate = promotion_passes(results, args.promotion_threshold)
        retire = league.retirement(results) if accepted else None
        decision = {
            "iteration": iteration,
            "accepted": accepted,
            "win_rate": rate,
            "threshold": args.promotion_threshold,
            "opponents": versions,
            "defeated": [v for v, r in zip(versions, results) if r["wins"] > r["losses"]],
            "removed": league.versions[retire] if retire is not None else None,
            "selection": "argmax",
            "opponent_selection": "argmax",
        }
        if accepted:
            league.add(model, iteration, env.carry.opponent, retire=retire)
            save(
                args.output / "promoted.zip",
                model,
                {"workflow": 1015, "promotion": decision, "iteration": iteration},
            )
        log["promotion"] = decision
        (folder / "promotion.json").write_text(
            json.dumps(decision, indent=2), encoding="utf-8"
        )
        emit(dict(event="promotion", **decision, league=league.metadata()))
        log.update(evaluation_pending=False, league=league.metadata())
        save(
            args.output / "checkpoint.zip",
            model,
            log,
            state,
            env.carry,
            league.bank,
            critic,
            critic_state,
        )
        save(args.output / "policy.zip", model, log)

    if args.resume and metadata.get("evaluation_pending"):
        assess(iteration0, metadata)
    for iteration in range(iteration0 + 1, args.iterations + 1):
        started = time.monotonic()
        emit({"phase": "rollout", "event": "start", "iteration": iteration})
        data, rollout = env.collect(model, critic, league.bank, league.eligible(), emit)
        rollout_seconds = time.monotonic() - started
        gathering_metrics = prepare_gathering(
            data, env.topology, args.gathering_samples,
            np.random.SeedSequence([args.seed, iteration, 1015]),
        )
        emit(dict(phase="gathering_labels", iteration=iteration, **gathering_metrics))
        data.advantage -= data.advantage.mean()
        data.advantage /= max(float(data.advantage.std()), 1e-6)
        passed = data.actions == config.side**2 * 8
        rows, norms = [], []
        probe_indices = np.random.default_rng(args.seed + iteration).choice(
            data.count, min(args.kl_samples, data.count), replace=False
        )
        timeline_started = time.monotonic()
        timeline = data.device_timeline()
        jax.block_until_ready(timeline)
        timeline_seconds = time.monotonic() - timeline_started
        timeline_gib = sum(x.nbytes for x in jax.tree.leaves(timeline)) / 2**30
        emit({"phase": "update", "event": "timeline", "iteration": iteration,
              "gib": timeline_gib, "seconds": timeline_seconds})
        probe = jax.device_put(data.read(probe_indices, indexed=True))
        kl_before = float(measure_kl(model, probe, env.topology, timeline))
        if not math.isfinite(kl_before) or kl_before > 1e-4:
            raise RuntimeError(
                f"Rollout policy/log-probability mismatch: KL={kl_before}"
            )
        stopped, probe_kl = False, kl_before
        for epoch in range(args.epochs):
            permutation = rng.permutation(data.count)
            for offset, (part, size) in enumerate(
                minibatches(data, permutation, args.batch)
            ):
                model, state, metrics, gradient = update(
                    (part, env.topology, timeline), model, state
                )
                rows.append(metrics)
                norms.append(gradient)
                if (
                    len(rows) % args.kl_every == 0
                    or (offset + 1) * args.batch == data.count
                ):
                    # One synchronization per progress block, not per microbatch.
                    measured = np.asarray(jnp.stack(rows[-args.kl_every :]))
                    gradients = np.asarray(jnp.stack(norms[-args.kl_every :]))
                    probe_kl = float(measure_kl(model, probe, env.topology, timeline))
                    if (
                        not np.isfinite(measured).all()
                        or not np.isfinite(gradients).all()
                        or not math.isfinite(probe_kl)
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
                            "probe_kl": probe_kl,
                            "seconds": time.monotonic() - started - rollout_seconds,
                        }
                    )
                    if probe_kl > args.target_kl:
                        stopped = True
                        break
            if stopped:
                break
        measured = np.asarray(jnp.stack(rows)).mean(0).tolist()
        gradients = np.asarray(jnp.stack(norms))
        update_seconds = time.monotonic() - started - rollout_seconds
        # Fixed rollout targets: critic work neither changes actor data nor advances games.
        critic_started = time.monotonic()
        critic_losses, critic_norms = [], []
        for epoch in range(args.critic_epochs):
            for offset, (part, _) in enumerate(
                minibatches(data, rng.permutation(data.count), args.batch)
            ):
                critic, critic_state, qloss, qnorm = update_critic(
                    (part, env.topology, timeline), critic, critic_state
                )
                critic_losses.append(qloss)
                critic_norms.append(qnorm)
                if (offset + 1) % args.kl_every == 0 or (
                    offset + 1
                ) * args.batch == data.count:
                    values = np.asarray(jnp.stack(critic_losses[-args.kl_every :]))
                    magnitudes = np.asarray(jnp.stack(critic_norms[-args.kl_every :]))
                    if (
                        not np.isfinite(values).all()
                        or not np.isfinite(magnitudes).all()
                    ):
                        raise FloatingPointError(
                            "Nonfinite Q-critic update; previous checkpoint is intact"
                        )
                    emit(
                        {
                            "phase": "critic",
                            "iteration": iteration,
                            "epoch": epoch + 1,
                            "optimized": len(critic_losses) * args.batch,
                            "total": args.critic_epochs * data.count,
                            "q_loss": float(values.mean()),
                            "seconds": time.monotonic() - critic_started,
                        }
                    )
        q_losses = np.asarray(jnp.stack(critic_losses))
        q_norms = np.asarray(jnp.stack(critic_norms))
        del probe, part, timeline
        log = dict(
            phase="ppo",
            workflow=1015,
            run=asdict(config),
            settings=settings,
            iteration=iteration,
            **rollout,
            **gathering_metrics,
            **dict(
                zip(
                    (
                        "policy_loss",
                        "value_loss",
                        "entropy",
                        "kl",
                        "auxiliary_loss",
                        "opponent_action_nll",
                        "opponent_action_accuracy",
                        "hidden_owner_nll",
                        "hidden_army_conditional_nll",
                        "general_nll",
                        "economy_prediction_loss",
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
            optimized=len(rows) * args.batch,
            kl_before=kl_before,
            probe_kl=probe_kl,
            kl_early_stop=stopped,
            evaluation_pending=True,
            rollout_seconds=rollout_seconds,
            update_seconds=update_seconds,
            timeline_gib=timeline_gib,
            timeline_seconds=timeline_seconds,
            critic_seconds=time.monotonic() - critic_started,
            q_loss=float(q_losses.mean()),
            q_gradient_before_clip=float(q_norms.mean()),
            q_gradient_before_clip_max=float(q_norms.max()),
            q_optimizer_steps=len(critic_losses),
            league=league.metadata(),
            numpy_rng=rng.bit_generator.state,
        )
        log["economy_prediction_by_horizon"] = {
            str(horizon): {
                "huber": measured[11 + i] / measured[11 + len(HORIZONS) + i]
                if measured[11 + len(HORIZONS) + i] > 0
                else None,
                "trained_valid_fraction": measured[11 + len(HORIZONS) + i],
            }
            for i, horizon in enumerate(HORIZONS)
        }
        log.update(zip(
            ("gathering_loss", "gathering_mass_mse", "gathering_optimal_accuracy",
             "gathering_normalized_regret"), measured[11 + 2 * len(HORIZONS):]
        ))
        saving = time.monotonic()
        # Save before evaluation so a slow match or interruption cannot lose updates.
        save(
            args.output / "checkpoint.zip",
            model,
            log,
            state,
            env.carry,
            league.bank,
            critic,
            critic_state,
        )
        save(args.output / "policy.zip", model, log)
        snapshot = args.output / "snapshots" / f"iteration-{iteration:08d}.zip"
        save(snapshot, model, log)
        log.update(
            save_seconds=time.monotonic() - saving, seconds=time.monotonic() - started
        )
        emit(log)
        with (args.output / "training.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(log) + "\n")
        if log["evaluation_pending"]:
            assess(iteration, log)


if __name__ == "__main__":
    main()
