"""On-policy self-play using the existing JAX game engine and the same actor.

Both seats contribute rollouts. Terminal rewards are +/-1; time limits are
truncations, not fabricated draws or losses. GAE bootstraps truncations but does
not carry advantages through resets. No replay demonstrations enter PPO ratios.
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

# Direct PPO invocations use the same on-demand allocation as train.sh.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from features import action, empty, encode, graph, observe
from generals.core.game import create_initial_state, step
from generals.core.grid import generate_grid
from model import Network
from rollout import PackedBatch, Record, Rollout, pack
from train import auxiliary, load, optimizer, save


@dataclass(frozen=True)
class Run:
    side: int = 24
    distance: int = 17
    environments: int = 512
    pool: int = 4096
    steps: int = 512
    horizon: int = 1024
    gamma: float = 1.0
    gae: float = 0.95


class Carry(NamedTuple):
    state: object
    memory: object
    inputs: object
    key: jax.Array


def initialize_pass(model, scale):
    """Adjust pass odds once before PPO sampling; the bias remains trainable.

    Adding log(scale) preserves relative move probabilities after softmax.
    The existing legal mask keeps pass certain when no move is available.
    """
    if not 0 < scale <= 1:
        raise ValueError("Initial pass scale must be finite and in (0, 1]")
    if scale == 1:
        return model
    return eqx.tree_at(
        lambda m: m.pass_head.bias,
        model,
        model.pass_head.bias + math.log(scale),
    )


def raw(state, player, topology):
    owners = jnp.where(state.ownership[0], 0, jnp.where(state.ownership[1], 1, -1))
    return observe(
        state.armies,
        owners,
        state.mountains,
        state.castles,
        state.generals,
        state.time,
        player,
        (False, False, True),
        jnp.ones(state.armies.shape, bool),
        topology,
    )


def views(states, topology):
    return jax.vmap(lambda s: jax.vmap(lambda p: raw(s, p, topology))(jnp.arange(2)))(
        states
    )


def targets(states):
    """Training labels isolated from observations and action selection."""

    def one(state, player):
        enemy = (state.generals & state.ownership[1 - player]).reshape(-1)
        owners = jnp.where(
            state.ownership[player], 1, jnp.where(state.ownership[1 - player], 2, 0)
        )
        return {
            "general": jnp.where(enemy.any(), jnp.argmax(enemy), -1),
            "owner": owners.reshape(-1),
            "army": jnp.log1p(state.armies.reshape(-1)) / jnp.log(256.0),
        }

    return jax.vmap(lambda s: jax.vmap(lambda p: one(s, p))(jnp.arange(2)))(states)


def encode_players(obs, memory, previous, topology):
    return jax.vmap(jax.vmap(encode, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, None))(
        obs, memory, previous, topology
    )


def choose(done, new, old):
    return jax.tree.map(
        lambda a, b: jnp.where(
            done.reshape(done.shape + (1,) * (a.ndim - done.ndim)), a, b
        ),
        new,
        old,
    )


def advantages(reward, value, following, terminal, truncated, gamma, lam):
    def backward(last, row):
        r, v, n, t, c = row
        delta = r + gamma * (1 - t) * n - v
        advantage = delta + gamma * lam * (~(t | c)) * last
        return advantage, advantage

    return jax.lax.scan(
        backward,
        jnp.zeros_like(value[0]),
        (reward, value, following, terminal, truncated),
        reverse=True,
    )[1]


@eqx.filter_jit
def initial_maps(config, keys):
    """Generate the shared training/evaluation map distribution."""
    castles = (1, max(2, config.side * config.side // 50))
    maps = jax.vmap(
        lambda key: generate_grid(
            key,
            grid_dims=(config.side, config.side),
            pad_to=config.side,
            num_castles_range=castles,
            min_generals_distance=config.distance,
        )
    )(keys)
    return jax.vmap(create_initial_state)(maps)


class Environment:
    def __init__(self, config, history, seed, inference_batch=32, rollout_chunk=8):
        self.config = config
        self.topology = graph(config.side, config.side)
        self.history = history
        self.inference_batch = inference_batch
        self.rollout_chunk = rollout_chunk
        keys = jax.random.split(jax.random.PRNGKey(seed), config.pool)
        pieces = [initial_maps(config, keys[i:i + 128]) for i in range(0, config.pool, 128)]
        self.pool = jax.tree.map(lambda *x: jnp.concatenate(x), *pieces)
        states = jax.tree.map(
            lambda x: x[jnp.arange(config.environments) % config.pool], self.pool
        )
        memory = self.blank()
        previous = jnp.full((config.environments, 2), config.side**2 * 8)
        inputs, memory = encode_players(
            views(states, self.topology), memory, previous, self.topology
        )
        self.carry = Carry(states, memory, inputs, jax.random.PRNGKey(seed + 1))
        self.buffer = Rollout(config.steps, config.environments, config.side**2, history)
        self.advance = eqx.filter_jit(self._advance)
        self.predict = eqx.filter_jit(self._predict)

    def _predict(self, model, inputs):
        flattened = jax.tree.map(lambda x: x.reshape((-1, *x.shape[2:])), inputs)

        def one(x):
            output = model(x, self.topology)
            return output.logits, output.value

        logits, value = jax.lax.map(one, flattened, batch_size=self.inference_batch)
        return logits.reshape((*inputs.legal.shape[:2], -1)), value.reshape(inputs.legal.shape[:2])

    def blank(self):
        one = empty(self.config.side**2, self.history)
        return jax.tree.map(
            lambda x: jnp.broadcast_to(x, (self.config.environments, 2, *x.shape)), one
        )

    def _advance(self, model, carry, count):
        config, topology = self.config, self.topology

        def tick(carry, _):
            state, memory, inputs, key = carry
            key, sample_key, reset_key = jax.random.split(key, 3)
            logits, value = self._predict(model, inputs)
            logp = jax.nn.log_softmax(logits)
            actions = jax.random.categorical(sample_key, logp)
            moves = jax.vmap(
                jax.vmap(lambda a: action(a, config.side, config.side**2))
            )(actions)
            following, info = jax.vmap(lambda s, a: step(s, a, general_trade=True))(
                state, moves
            )
            terminal = info.is_done
            truncated = (following.time >= config.horizon) & ~terminal
            ended = terminal | truncated
            reward = jnp.where(
                terminal[:, None],
                jnp.where(info.winner[:, None] == jnp.arange(2), 1.0, -1.0),
                0.0,
            )
            next_inputs, next_memory = encode_players(
                views(following, topology), memory, actions, topology
            )
            # Consecutive values are reused after sampling. Only truncations need
            # a value for a state that will not appear in the next action batch.
            bootstrap = jax.lax.cond(
                truncated.any(), lambda: self._predict(model, next_inputs)[1],
                lambda: jnp.zeros_like(value),
            )
            indices = jax.random.randint(
                reset_key, (config.environments,), 0, config.pool
            )
            fresh = jax.tree.map(lambda x: x[indices], self.pool)
            reset_inputs, reset_memory = encode_players(
                views(fresh, topology),
                self.blank(),
                jnp.full_like(actions, config.side**2 * 8),
                topology,
            )
            next_carry = Carry(
                choose(ended, fresh, following),
                choose(ended, reset_memory, next_memory),
                choose(ended, reset_inputs, next_inputs),
                key,
            )
            flags, values = pack(inputs)
            labels = targets(state)
            labels["owner"] = labels["owner"].astype(jnp.uint8)
            recorded = Record(
                flags, values, inputs.stats[..., -1, :], inputs.time_valid.sum(-1),
                actions,
                jnp.take_along_axis(logp, actions[..., None], -1)[..., 0],
                value,
                bootstrap,
                reward,
                terminal,
                truncated,
                jnp.any(inputs.legal[..., :-1], -1),
                labels,
            )
            return next_carry, recorded

        return jax.lax.scan(tick, carry, None, length=count)

    def collect(self, model, carry, progress=False):
        buffer = self.buffer
        buffer.begin(carry.inputs)
        started = time.monotonic()
        for offset in range(0, self.config.steps, self.rollout_chunk):
            count = min(self.rollout_chunk, self.config.steps - offset)
            carry, records = self.advance(model, carry, count)
            buffer.write(offset, jax.device_get(records))
            del records
            if progress and ((offset + count) % 64 == 0 or offset + count == self.config.steps):
                print(json.dumps({"phase": "rollout", "steps": offset + count,
                                  "total": self.config.steps, "seconds": time.monotonic() - started}), flush=True)
        following = np.asarray(self.predict(model, carry.inputs)[1]).reshape(-1)
        accumulated = np.zeros(buffer.lanes, np.float32)
        for t in range(self.config.steps - 1, -1, -1):
            next_value = np.where(buffer.truncated[t], buffer.bootstrap[t], following)
            delta = buffer.reward[t] + self.config.gamma * ~buffer.terminal[t] * next_value - buffer.value[t]
            accumulated = delta + self.config.gamma * self.config.gae * ~(buffer.terminal[t] | buffer.truncated[t]) * accumulated
            buffer.advantage[t] = accumulated
            following = buffer.value[t]
        buffer.returns[:] = buffer.advantage + buffer.value
        metrics = np.array([buffer.terminal.sum() // 2, buffer.truncated.sum() // 2,
                            buffer.advantage.std(), np.count_nonzero(buffer.reward)])
        return carry, buffer, metrics


def objective(model, data, topology, entropy_coefficient, auxiliary_weight, clip=0.2):
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
    return (
        policy + 0.5 * value - entropy_coefficient * entropy + auxiliary_weight * aux,
        jnp.array([policy, value, entropy, kl, aux]),
    )


def make_update(tx, entropy, auxiliary_weight=0.02, microbatch=16):
    @eqx.filter_jit
    def update(model, state, data, topology):
        # One optimizer update per logical minibatch. Clipping follows accumulation.
        size = len(data.actions)
        chunk = min(size, microbatch)
        if size % chunk:
            chunk = math.gcd(size, chunk)
        batches = jax.tree.map(lambda x: x.reshape((-1, chunk, *x.shape[1:])), data)
        zeros = jax.tree.map(jnp.zeros_like, eqx.filter(model, eqx.is_inexact_array))

        def accumulate(carry, part):
            loss_sum, metrics_sum, gradients = carry
            if isinstance(part, PackedBatch):
                part = part.unpack()
            (loss, metrics), gradient = eqx.filter_value_and_grad(objective, has_aux=True)(
                model, part, topology, entropy, auxiliary_weight
            )
            fraction = chunk / size
            gradients = jax.tree.map(lambda a, b: a + b * fraction, gradients, gradient)
            return (loss_sum + loss * fraction, metrics_sum + metrics * fraction, gradients), None

        (loss, metrics, gradient), _ = jax.lax.scan(accumulate, (jnp.array(0.), jnp.zeros(5), zeros), batches)
        change, state = tx.update(
            gradient, state, eqx.filter(model, eqx.is_inexact_array)
        )
        return (
            eqx.apply_updates(model, change),
            state,
            loss,
            metrics,
            optax.tree.norm(gradient),
        )

    return update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path)
    parser.add_argument(
        "--init-pass-scale",
        type=float,
        default=1.0,
        help="Scale initial pass odds once before sampling (0 < scale <= 1); requires --init when below 1",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/ppo"))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--side", type=int, default=Run.side)
    parser.add_argument("--distance", type=int, default=Run.distance)
    parser.add_argument("--environments", type=int, default=Run.environments)
    parser.add_argument("--pool", type=int, default=Run.pool)
    parser.add_argument("--steps", type=int, default=Run.steps)
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1024, help="Effective optimizer minibatch")
    parser.add_argument("--microbatch", type=int, default=16, help="Gradient accumulation chunk; divides --batch")
    parser.add_argument("--inference-batch", type=int, default=32)
    parser.add_argument("--rollout-chunk", type=int, default=8, help="Steps per device-to-host transfer")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--auxiliary", type=float, default=0.02)
    parser.add_argument("--save", type=int, default=10)
    parser.add_argument("--evaluate", type=int, default=50,
                        help="Evaluate every N iterations and at the end; no startup evaluation; 0 disables evaluation")
    parser.add_argument("--evaluation-games", type=int, default=128,
                        help="Games per opponent, including both colors of each held-out map")
    parser.add_argument("--evaluation-environments", type=int, default=16)
    parser.add_argument("--evaluation-seed", type=int, default=1001012)
    parser.add_argument("--evaluation-history", type=int, default=2,
                        help="Number of historical snapshots in addition to original, initial and baseline")
    parser.add_argument("--seed", type=int, default=1012)
    args = parser.parse_args()
    if (
        min(
            args.iterations,
            args.environments,
            args.pool,
            args.steps,
            args.batch,
            args.microbatch,
            args.inference_batch,
            args.rollout_chunk,
            args.epochs,
            args.save,
            args.horizon,
        )
        < 1
        or args.side < 4
    ):
        parser.error("Positive counts and side >= 4 required")
    if not 1 <= args.distance <= 2 * (args.side - 1):
        parser.error("Choose a feasible initial distance")
    count = 2 * args.environments * args.steps
    chunk = min(args.microbatch, args.batch, count)
    if args.batch % chunk or count % chunk:
        parser.error("Minibatch and rollout player counts must be divisible by the effective microbatch")
    if args.learning_rate <= 0 or min(args.entropy, args.auxiliary) < 0:
        parser.error("Positive learning rate and nonnegative loss coefficients required")
    if not 0 < args.init_pass_scale <= 1:
        parser.error("--init-pass-scale must be finite and in (0, 1]")
    if args.init_pass_scale != 1 and args.init is None:
        parser.error("--init-pass-scale below 1 requires --init")
    if args.evaluate < 0 or args.evaluation_history < 0:
        parser.error("Evaluation interval and history count must be nonnegative")
    if (args.evaluation_games < 2 or args.evaluation_games % 2
            or args.evaluation_environments < 2 or args.evaluation_environments % 2):
        parser.error("Evaluation games and environments must be positive even counts")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new --output directory; existing checkpoints and evaluations are preserved")
    original = (
        load(args.init)[0] if args.init else Network(key=jax.random.PRNGKey(args.seed))
    )
    model = initialize_pass(original, args.init_pass_scale)
    initialization = {
        "init": str(args.init) if args.init else None,
        "init_pass_scale": args.init_pass_scale,
    }
    config = Run(
        side=args.side,
        distance=args.distance,
        environments=args.environments,
        pool=args.pool,
        steps=args.steps,
        horizon=args.horizon,
    )
    print(
        json.dumps({"event": "start", "phase": "ppo", "config": asdict(config),
                    "batch": args.batch, "microbatch": chunk, "epochs": args.epochs,
                    "inference_batch": args.inference_batch, "storage": "cpu",
                    "player_samples_per_iteration": count, "startup_evaluation": False, **initialization}),
        flush=True,
    )
    env = Environment(config, model.config.history, args.seed, args.inference_batch, args.rollout_chunk)
    print(json.dumps({"event": "buffer", "storage": "cpu", "gib": env.buffer.nbytes / 2**30}), flush=True)
    initial_log = {"phase": "ppo", "run": asdict(config), "iteration": 0, **initialization}
    initial_snapshot = args.output / "snapshots" / "iteration-00000000.zip"
    save(initial_snapshot, model, initial_log)
    evaluation = None
    tx = optimizer(args.learning_rate)
    state = tx.init(eqx.filter(model, eqx.is_inexact_array))
    update = make_update(tx, args.entropy, args.auxiliary, chunk)
    rng = np.random.default_rng(args.seed)
    for iteration in range(1, args.iterations + 1):
        start = time.monotonic()
        print(json.dumps({"phase": "rollout", "event": "start", "iteration": iteration}), flush=True)
        env.carry, data, rollout = env.collect(model, env.carry, progress=True)
        rollout_seconds = time.monotonic() - start
        data.advantage -= data.advantage.mean()
        data.advantage /= max(float(data.advantage.std()), 1e-6)
        movable = data.movable
        passed = data.actions == config.side**2 * 8
        pass_metrics = (passed.mean(), np.sum(passed & movable) / max(movable.sum(), 1), movable.sum())
        rows = []
        norms = []
        weights = []
        for epoch in range(args.epochs):
            permutation = rng.permutation(data.count)
            for offset in range(0, len(permutation), args.batch):
                ids = permutation[offset : offset + args.batch]
                part = jax.device_put(data.read(ids))
                model, state, loss, metrics, gradient = update(
                    model, state, part, env.topology
                )
                if not bool(jnp.isfinite(loss) & jnp.isfinite(gradient)):
                    raise FloatingPointError("Nonfinite PPO update")
                rows.append(np.asarray(metrics))
                norms.append(float(gradient))
                weights.append(len(ids))
                if len(rows) % 32 == 0 or offset + len(ids) == data.count:
                    print(json.dumps({"phase": "update", "iteration": iteration, "epoch": epoch + 1,
                                      "optimized": epoch * data.count + offset + len(ids),
                                      "total": args.epochs * data.count, "seconds": time.monotonic() - start}), flush=True)
        measured = np.average(rows, axis=0, weights=weights).tolist()
        log = {
            "phase": "ppo",
            "run": asdict(config),
            **initialization,
            "iteration": iteration,
            "completed": int(rollout[0]),
            "truncated": int(rollout[1]),
            "advantage_std": float(rollout[2]),
            "reward_nonzero": int(rollout[3]),
            "policy_loss": measured[0],
            "value_loss": measured[1],
            "entropy": measured[2],
            "kl": measured[3],
            "auxiliary_loss": measured[4],
            "gradient_before_clip": float(np.mean(norms)),
            "gradient_before_clip_max": max(norms),
            "gradient_clip": 0.5,
            "batch": args.batch,
            "microbatch": chunk,
            "epochs": args.epochs,
            "optimizer_steps": len(rows),
            "player_samples": data.count,
            "rollout_seconds": rollout_seconds,
            "pass_fraction": float(pass_metrics[0]),
            "optional_pass_fraction": float(pass_metrics[1]),
            "movable_samples": int(pass_metrics[2]),
            "environment_samples": iteration * args.environments * args.steps,
            "seconds": time.monotonic() - start,
        }
        print(json.dumps(log), flush=True)
        with (args.output / "training.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(log) + "\n")
        evaluate_now = args.evaluate > 0 and (
            iteration % args.evaluate == 0 or iteration == args.iterations
        )
        if iteration % args.save == 0 or iteration == args.iterations or evaluate_now:
            save(args.output / "checkpoint.zip", model, log, state)
            save(args.output / "policy.zip", model, log)
            snapshot = args.output / "snapshots" / f"iteration-{iteration:08d}.zip"
            save(snapshot, model, log)
        if evaluate_now:
            if evaluation is None:
                from evaluate import EvaluationSuite

                evaluation = EvaluationSuite(
                    args.output, original, load(initial_snapshot)[0], config, initialization,
                    args.evaluation_games, args.evaluation_environments,
                    args.evaluation_seed, env.pool,
                )
            evaluation.run(model, iteration, snapshot, args.evaluation_history)


if __name__ == "__main__":
    main()
