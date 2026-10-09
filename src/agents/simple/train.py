"""Train GenFormer with compiled JAX self-play and Optax PPO updates."""

import argparse
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
from config import Config, maps
from env import Arena, View, observations, select_device
from features import Encoder, restore, transform
from model import GenFormer
from ppo import advantages, average, optimizer, predict, update
from storage import Progress, Rollout, export, load, pack, save


class Carry(NamedTuple):
    states: object
    rows: jax.Array
    cols: jax.Array
    encoder: Encoder
    inputs: object
    reset: jax.Array
    key: jax.Array


class Collector:
    def __init__(self, arena, config, inference_batch):
        self.arena, self.config = arena, config
        self.inference_batch = inference_batch
        self.bf16 = config.bf16 and arena.backend.platform == "gpu"
        count = 2 * arena.count
        encoder = Encoder.empty(count, config.side, config.history, config.temporal)
        reset = jnp.ones(count, jnp.bool_)
        encoder, inputs = encoder.update(arena.read(), reset)
        self.carry = Carry(arena.states, arena.rows, arena.cols, encoder, inputs, reset, jax.random.PRNGKey(0))
        self.segment = eqx.filter_jit(self._segment)
        self.finish = eqx.filter_jit(self._finish)

    def _segment(self, model, carry, pool, dimensions, orientation, length):
        arena, config = self.arena, self.config
        def step(carry, unused):
            states, rows, cols, encoder, inputs, reset, key = carry
            inputs = transform(inputs, orientation)
            frame = pack(inputs, reset)
            logits, value = predict(model, inputs, self.inference_batch, self.bf16)
            key, choice = jax.random.split(key)
            action = jax.random.categorical(choice, logits).astype(jnp.int32)
            logprob = jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), action[:, None], axis=1)[:, 0]
            canonical = restore(action, config.side, orientation)
            encoder = encoder.submitted(canonical)
            following, next_rows, next_cols, raw, reward, terminal, truncated, final = arena.transition(
                states, rows, cols, canonical, pool, dimensions)
            def bootstrap(unused):
                final_view = View(*observations(final, rows, cols, config.side))
                final_inputs = encoder.update(final_view, jnp.zeros_like(reset))[1]
                values = predict(model, transform(final_inputs, orientation), self.inference_batch, self.bf16)[1]
                return jnp.where(truncated, values, 0)
            limit = jax.lax.cond(jnp.any(truncated), bootstrap, lambda _: jnp.zeros_like(value), operand=None)
            reset = terminal | truncated
            encoder, inputs = encoder.update(View(*raw), reset)
            carry = Carry(following, next_rows, next_cols, encoder, inputs, reset, key)
            return carry, (frame, action, logprob, value, reward, terminal, truncated, limit)
        return jax.lax.scan(step, carry, None, length=length)

    def _finish(self, parts, last_value):
        action, logprob, values, rewards, terminal, truncated, limits = parts
        following = jnp.concatenate((values[1:], last_value[None]))
        following = jnp.where(truncated, limits, following)
        advantage = advantages(rewards, values, following, terminal, truncated, self.config.gamma, self.config.gae)
        data = {"action": action, "logprob": logprob, "return": advantage + values,
                "advantage": (advantage - advantage.mean()) / (advantage.std() + 1e-8)}
        metrics = {"completed": terminal[:, ::2].sum(), "truncated": truncated[:, ::2].sum(),
                   "reward_nonzero": jnp.count_nonzero(rewards), "advantage_std": advantage.std()}
        return data, metrics

    def collect(self, model, key):
        begun = time.perf_counter()
        key, symmetry = jax.random.split(key)
        orientation = jax.random.randint(symmetry, (), 0, 8) if self.config.augment else jnp.int32(0)
        rollout = Rollout.empty(transform(self.carry.inputs, orientation), self.config.steps)
        jax.block_until_ready(rollout)
        print(json.dumps({"phase": "rollout_storage", "gib": rollout.nbytes / 2**30,
                          "seconds": time.perf_counter() - begun}), flush=True)
        self.carry = self.carry._replace(key=key)
        parts = []
        for start in range(0, self.config.steps, 64):
            sampled = time.perf_counter()
            length = min(64, self.config.steps - start)
            self.carry, result = self.segment(model, self.carry, self.arena.pool, self.arena.dimensions, orientation, length)
            rollout = rollout.write(result[0], jnp.int32(start))
            parts.append(result[1:])
            jax.block_until_ready((rollout, parts[-1]))
            del result
            print(json.dumps({"phase": "rollout", "steps": start + length, "total": self.config.steps,
                              "segment_seconds": time.perf_counter() - sampled}), flush=True)
        finished = time.perf_counter()
        # only scalar transition records are joined; spatial history stays in the donated buffer.
        joined = jax.tree.map(lambda *values: jnp.concatenate(values), *parts)
        del parts
        last_value = eqx.filter_jit(predict)(model, transform(self.carry.inputs, orientation), self.inference_batch, self.bf16)[1]
        data, metrics = self.finish(joined, last_value)
        jax.block_until_ready(data)
        self.arena.states, self.arena.rows, self.arena.cols = self.carry.states, self.carry.rows, self.carry.cols
        self.arena.current = None
        metrics = {name: value.item() for name, value in metrics.items()}
        metrics["orientation"] = int(orientation)
        metrics["finish_seconds"] = time.perf_counter() - finished
        return rollout, data, self.carry.key, metrics


def assess(model, config, progress, backend):
    from evaluate import Evaluator
    distribution = maps(config, progress.stage)
    with Evaluator(distribution, backend) as evaluator:
        result = evaluator.matches(model)
    progress.evaluations += 1
    if result["wins"] / result["games"] >= config.threshold and progress.stage < len(config.curriculum):
        progress.stage += 1
    return result


def run(config, backend, output, until=None, checkpoint=None, model=None, inference_batch=None):
    config.validate()
    inference_batch = config.microbatch if inference_batch is None else inference_batch
    if inference_batch < 1:
        raise ValueError("inference-batch must be positive")
    progress = Progress(**checkpoint["progress"]) if checkpoint else Progress()
    limit = config.samples if until is None else until
    if limit <= progress.samples:
        raise ValueError("The stopping point must exceed the saved sample count")
    if checkpoint is None and (output / "checkpoint.eqx").exists():
        raise ValueError("Output already contains a run; resume it or select another directory")
    with jax.default_device(backend):
        key = jax.random.PRNGKey(config.seed)
        if checkpoint:
            state, ema, key = checkpoint["optimizer"], checkpoint["ema"], checkpoint["random"]
        else:
            key, initialization = jax.random.split(key)
            model = GenFormer(config, initialization)
            ema = model
            state = optimizer(config).init(eqx.filter(model, eqx.is_inexact_array))
        output.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"config": asdict(config), "device": str(backend), "inference_batch": inference_batch,
                          "progress": asdict(progress), "until": limit,
                          "parameters": sum(value.size for value in jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array)))}), flush=True)
        save(output / "checkpoint.eqx", model, ema, state, key, config, progress)
        with Arena(config, backend, config.seed + progress.samples) as arena:
            print("Preparing maps and compiling the environment...", flush=True)
            distribution = maps(config, progress.stage)
            arena.prepare(distribution)
            collector = Collector(arena, config, inference_batch)
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
                try:
                    while progress.samples < limit:
                        begun = time.perf_counter()
                        next_distribution = maps(config, progress.stage)
                        if next_distribution != distribution or (progress.iteration and progress.iteration % config.refresh == 0):
                            arena.prepare(next_distribution)
                            distribution = next_distribution
                        trained_stage = progress.stage
                        print(json.dumps({"phase": "rollout", "iteration": progress.iteration + 1,
                                          "environment_samples": config.environments * config.steps}), flush=True)
                        sampled = time.perf_counter()
                        rollout, data, key, metrics = collector.collect(model, key)
                        metrics["rollout_seconds"] = time.perf_counter() - sampled
                        metrics["rollout_storage_gib"] = rollout.nbytes / 2**30
                        optimized = time.perf_counter()
                        model, state, key, updates = update(model, state, rollout, data, config, progress.iteration, key, collector.bf16)
                        metrics.update(updates)
                        metrics["ppo_seconds"] = time.perf_counter() - optimized
                        ema = average(ema, model, config.ema)
                        del rollout, data
                        progress.samples += config.environments * config.steps
                        progress.iteration += 1
                        evaluated, finished = progress.iteration % config.evaluate == 0, progress.samples >= limit
                        if evaluated or finished:
                            metrics["gate"] = assess(model, config, progress, backend)
                        metrics.update(asdict(progress), seconds=time.perf_counter() - begun, player_samples=2 * progress.samples,
                                       distance=distribution.distance, trained_stage=trained_stage, next_stage=progress.stage)
                        line = json.dumps(metrics, allow_nan=False)
                        print(line, flush=True)
                        log.write(line + "\n")
                        log.flush()
                        if evaluated or finished or progress.iteration % config.save == 0:
                            save(output / "checkpoint.eqx", model, ema, state, key, config, progress)
                            export(output / "policy.eqx", ema, config)
                except KeyboardInterrupt:
                    print("Interrupted; resume the latest completed checkpoint. In-progress games will restart.", flush=True)
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--until", type=int, help="Absolute environment sample stopping point; does not change schedules")
    parser.add_argument("--microbatch", type=int, help="Players per PPO forward/backward pass")
    parser.add_argument("--inference-batch", type=int, help="Players per sampling forward pass; defaults to microbatch")
    integers = ("samples", "environments", "horizon", "side", "minimum", "maximum", "pool", "steps", "minibatch",
                "epochs", "width", "depth", "heads", "templates", "history", "temporal", "save", "evaluate", "games")
    for name in integers:
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    overrides = {name: getattr(args, name) for name in (*integers, "fraction", "bf16") if getattr(args, name) is not None}
    if args.resume and (overrides or args.curriculum is not None):
        parser.error("Resume restores training settings; use --until to change only the stopping point")
    try:
        backend = select_device(args.device)
        checkpoint = model = None
        if args.resume:
            checkpoint, config, model = load(args.resume, backend, "model")
            if not checkpoint["resumable"]:
                parser.error("An exported policy cannot resume training; use checkpoint.eqx")
        else:
            config = replace(Config(), **overrides)
            if args.curriculum is False:
                config = replace(config, curriculum=())
        if args.microbatch is not None:
            config = replace(config, microbatch=args.microbatch)
        output = args.output or (args.resume.parent if args.resume else Path(__file__).parent / "runs" / "train")
        run(config, backend, output, args.until, checkpoint, model, args.inference_batch)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
