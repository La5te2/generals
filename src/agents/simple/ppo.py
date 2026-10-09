"""Optimize PPO with Optax; gradient accumulation precedes one clipped Adam update."""

import json
import math
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import optax


def optimizer(config):
    return optax.chain(optax.clip_by_global_norm(config.gradient),
                       optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8))


def predict(model, inputs, batch_size, bf16=False):
    """Bound activation memory while keeping all sampling computation inside JAX."""
    count = len(inputs[0])
    batch_size = min(batch_size, count)
    blocks, remainder = divmod(count, batch_size)
    shaped = jax.tree.map(lambda value: value[:blocks * batch_size].reshape((blocks, batch_size, *value.shape[1:])), inputs)
    result = jax.lax.map(lambda batch: model(*batch, bf16=bf16)[:2], shaped)
    result = jax.tree.map(lambda value: value.reshape((-1, *value.shape[2:])), result)
    if remainder:
        tail = model(*(value[blocks * batch_size:] for value in inputs), bf16=bf16)[:2]
        result = jax.tree.map(lambda a, b: jnp.concatenate((a, b)), result, tail)
    return result


def histogram(target, centers, sigma):
    target = jnp.clip(target, centers[0], centers[-1])[..., None]
    half = (centers[1] - centers[0]) / 2
    upper = (centers + half - target) / (sigma * math.sqrt(2))
    lower = (centers - half - target) / (sigma * math.sqrt(2))
    weights = 0.5 * (jax.lax.erf(upper) - jax.lax.erf(lower))
    return weights / jnp.maximum(weights.sum(-1, keepdims=True), 1e-8)


@eqx.filter_jit
def average(ema, model, decay):
    return jax.tree.map(lambda old, new: old + (new - old) * (1 - decay), ema, model)


@jax.jit
def advantages(rewards, values, following, terminal, truncated, gamma, gae):
    def step(carried, row):
        reward, value, next_value, end, limit = row
        delta = reward + gamma * ~end * next_value - value
        carried = delta + gamma * gae * ~(end | limit) * carried
        return carried, carried
    return jax.lax.scan(step, jnp.zeros_like(rewards[0]),
                        (rewards, values, following, terminal, truncated), reverse=True)[1]


def loss(model, inputs, action, old_logprob, advantage, returns, config, coefficient, bf16):
    logits, _, value_logits = model(*inputs, bf16=bf16)
    logs = jax.nn.log_softmax(logits, axis=-1)
    logprob = jnp.take_along_axis(logs, action[:, None], axis=1)[:, 0]
    logratio = logprob - old_logprob
    ratio = jnp.exp(logratio)
    policy = -jnp.minimum(ratio * advantage, jnp.clip(ratio, 1 - config.clip, 1 + config.clip) * advantage).mean()
    divergence = ((ratio - 1) - logratio).mean()
    target = histogram(returns, jnp.linspace(-1, 1, config.bins), config.sigma)
    critic = -(target * jax.nn.log_softmax(value_logits, axis=-1)).sum(-1).mean()
    entropy = -(jnp.exp(logs) * logs).sum(-1).mean()
    total = policy + config.value * critic - coefficient * entropy
    return total, jnp.stack((policy, critic, entropy, divergence))


@eqx.filter_jit
def minibatch(model, state, rollout, ids, data, config, rate, coefficient, allow_stop, bf16):
    size = len(data["action"])
    chunk = min(config.microbatch, size)
    blocks, remainder = divmod(size, chunk)
    gradient = jax.tree.map(jnp.zeros_like, eqx.filter(model, eqx.is_inexact_array))
    stats = jnp.zeros(4)
    finite = jnp.array(True)
    differentiate = eqx.filter_value_and_grad(loss, has_aux=True)

    def accumulate(carry, index):
        gradient, stats, finite = carry
        # reconstruct history only for the microbatch that is about to use it.
        batch = rollout.read(jax.lax.dynamic_slice_in_dim(ids, index * chunk, chunk))
        selected = jax.tree.map(lambda value: jax.lax.dynamic_slice_in_dim(value, index * chunk, chunk), data)
        (value, metrics), partial = differentiate(model, batch, selected["action"], selected["logprob"],
                                                  selected["advantage"], selected["return"], config, coefficient, bf16)
        weight = chunk / size
        gradient = jax.tree.map(lambda old, new: old + weight * new, gradient, partial)
        return (gradient, stats + weight * metrics, finite & jnp.isfinite(value)), None

    (gradient, stats, finite), _ = jax.lax.scan(accumulate, (gradient, stats, finite), jnp.arange(blocks))
    if remainder:
        batch = rollout.read(ids[blocks * chunk:])
        selected = jax.tree.map(lambda value: value[blocks * chunk:], data)
        (value, metrics), partial = differentiate(model, batch, selected["action"], selected["logprob"],
                                                  selected["advantage"], selected["return"], config, coefficient, bf16)
        gradient = jax.tree.map(lambda old, new: old + (remainder / size) * new, gradient, partial)
        stats = stats + (remainder / size) * metrics
        finite = finite & jnp.isfinite(value)
    norm = optax.global_norm(gradient)
    finite = finite & jnp.isfinite(norm)
    stopped = (stats[3] > config.kl) & allow_stop if config.kl is not None else jnp.array(False)
    def apply(args):
        model, state = args
        updates, state = optimizer(config).update(gradient, state, eqx.filter(model, eqx.is_inexact_array))
        updates = jax.tree.map(lambda value: -rate * value, updates)
        return eqx.apply_updates(model, updates), state
    model, state = jax.lax.cond(finite & ~stopped, apply, lambda args: args, (model, state))
    return model, state, stats, norm, finite, stopped


def update(model, state, rollout, data, config, iteration, key, bf16=False):
    begun = time.perf_counter()
    data = jax.tree.map(lambda value: value.reshape(-1), data)
    count = data["action"].size
    rate = min(config.learning_rate, max(config.final_rate, 0.5 / (iteration + 1)**1.1))
    coefficient = max(config.final_entropy, config.entropy / (iteration + 1)**0.2)
    retained = max(1, math.ceil(count * config.fraction))
    indices = jax.lax.top_k(jnp.abs(data["advantage"]), retained)[1]
    totals = jnp.zeros(4)
    samples, largest_gradient, updates, stopped = 0, 0., 0, False
    for epoch in range(config.epochs):
        key, shuffle = jax.random.split(key)
        order = jax.random.permutation(shuffle, indices)
        for start in range(0, retained, config.minibatch):
            selected = order[start:start + config.minibatch]
            chosen = jax.tree.map(lambda value, ids=selected: value[ids], data)
            model, state, stats, norm, finite, stop = minibatch(
                model, state, rollout, selected, chosen, config, jnp.float32(rate), jnp.float32(coefficient), jnp.array(samples > 0), bf16)
            if not bool(finite):
                raise FloatingPointError("Nonfinite PPO loss or gradient; parameters were not updated")
            if bool(stop):
                stopped = True
                break
            largest_gradient = max(largest_gradient, float(norm))
            totals = totals + stats * len(selected)
            samples += len(selected)
            updates += 1
            if updates == 1 or updates % 32 == 0:
                print(json.dumps({"phase": "ppo", "optimized": samples, "total": retained * config.epochs,
                                  "seconds": time.perf_counter() - begun}), flush=True)
        if stopped:
            break
    metrics = dict(zip(("policy_loss", "value_loss", "entropy", "kl"), (totals / samples).tolist(), strict=True),
                   gradient=largest_gradient, batch_samples=count, retained_samples=retained, optimized=samples,
                   early_stop=stopped, optimizer_steps=updates, learning_rate=rate, entropy_coefficient=coefficient)
    return model, state, key, metrics
