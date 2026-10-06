"""Self-play, PPO updates and checkpoint storage run independently of the graphical application."""

import argparse
import copy
import json
import time
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from config import Config, profile
from environment import Arena
from model import Model, inputs
from ppo import advantages, average, objective


def precision(device, config):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and config.bf16 else nullcontext()


@torch.inference_mode()
def collect(model, arena, config, stage, device):
    count, steps, side = 2 * arena.count, config.steps, config.side
    board = torch.empty((steps, count, 38, side, side), dtype=torch.bfloat16)
    legal = torch.empty((steps, count, 9, side, side), dtype=torch.bool)
    history = torch.empty((steps, count, 2, 512))
    actions = torch.empty((steps, count), dtype=torch.long)
    logprobs, values, rewards, limitValues = [torch.zeros(steps, count) for field in range(4)]
    terminal, truncated = [torch.zeros(steps, count, dtype=torch.bool) for field in range(2)]
    completed = 0
    for step in range(steps):
        raw = arena.read()
        board[step].copy_(torch.from_numpy(raw[0]))
        legal[step].copy_(torch.from_numpy(raw[1]))
        history[step].copy_(torch.from_numpy(raw[2]))
        with precision(device, config):
            logits, prediction = model(*inputs(raw, device))[:2]
        distribution = torch.distributions.Categorical(logits=logits)
        choice = distribution.sample()
        actions[step].copy_(choice)
        logprobs[step].copy_(distribution.log_prob(choice))
        values[step].copy_(prediction)
        reward, ended, limited = arena.step(choice.cpu().numpy())
        rewards[step].copy_(torch.from_numpy(reward))
        terminal[step].copy_(torch.from_numpy(ended))
        truncated[step].copy_(torch.from_numpy(limited))
        if np.any(limited):
            ids = np.flatnonzero(limited)
            final = tuple(array[ids] for array in arena.read())
            with precision(device, config):
                finalValue = model(*inputs(final, device))[1]
            limitValues[step, ids] = finalValue.cpu()
        if np.any(ended | limited):
            completed += int(np.count_nonzero(ended[::2]))
            arena.reset(config.curriculum[stage], config.horizon)
    with precision(device, config):
        finalValue = model(*inputs(arena.read(), device))[1]
    following = torch.cat((values[1:], finalValue.cpu().unsqueeze(0)))
    following[truncated] = limitValues[truncated]
    advantage = advantages(rewards, values, following, terminal, truncated, config.gamma, config.gae)
    returns = advantage + values
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    batches = [board, legal, history, actions, logprobs, advantage, returns]
    return [data.flatten(0, 1) for data in batches], {
        "completed": completed, "truncated": int(truncated[:, ::2].sum()),
        "terminal_rewards": int(rewards.count_nonzero()), "advantage_std": float(returns.sub(values).std(unbiased=False)),
    }


def update(model, optimizer, batch, config, iteration, device):
    board, legal, history, actions, previous, advantage, returns = batch
    kept = max(1, int(advantage.numel() * config.fraction))
    indices = advantage.abs().topk(kept, sorted=False).indices
    rate = min(1e-4, max(5e-6, 0.5 / (iteration + 1) ** 1.1))
    entropy = max(0.001, 0.05 / (iteration + 1) ** 0.2)
    for group in optimizer.param_groups:
        group["lr"] = rate
    totals = torch.zeros(4, device=device)
    samples = 0
    gradient = 0.0
    for epoch in range(config.epochs):
        order = indices[torch.randperm(kept)]
        for ids in order.split(config.minibatch):
            optimizer.zero_grad(set_to_none=True)
            with precision(device, config):
                prediction = model(board[ids].to(device=device, dtype=torch.float32), legal[ids].to(device), history[ids].to(device))
                loss, stats = objective(prediction[0], prediction[2], actions[ids].to(device), previous[ids].to(device),
                                        advantage[ids].to(device), returns[ids].to(device), model.centers, config, entropy)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss became nonfinite")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad, error_if_nonfinite=True)
            gradient = max(gradient, float(norm))
            optimizer.step()
            totals += stats * len(ids)
            samples += len(ids)
    values = (totals / samples).cpu().tolist()
    return dict(zip(("policy_loss", "value_loss", "entropy", "kl"), values, strict=True),
                grad_norm=gradient, selected=kept, learning_rate=rate, entropy_coefficient=entropy)


@torch.inference_mode()
def evaluate(model, config, stage, device, path, seed):
    # a random legal mover is the curriculum gate from the released training recipe, rather than a strength benchmark.
    rng = np.random.default_rng(seed)
    count = min(config.games, config.environments, 32)
    wins, completed, cutoffs = 0, 0, 0
    with Arena(count, config.side, seed, path) as arena:
        arena.reset(config.curriculum[stage], config.horizon)
        while completed < config.games:
            raw = arena.read()
            with precision(device, config):
                logits = model(*inputs(raw, device))[0]
            moves = logits.argmax(-1).cpu().numpy().astype(np.int32)
            for game in range(count):
                opponent = 2 * game + (1 - game % 2)
                possible = np.flatnonzero(raw[1][opponent, :8])
                moves[opponent] = rng.choice(possible) if possible.size else 8 * config.side ** 2
            rewards, ended, limited = arena.step(moves)
            for game in range(count):
                if completed >= config.games:
                    break
                if ended[2 * game] or limited[2 * game]:
                    wins += int(rewards[2 * game + game % 2] > 0)
                    cutoffs += int(limited[2 * game])
                    completed += 1
            if np.any(ended | limited):
                arena.reset(config.curriculum[stage], config.horizon)
    return {"gate_win_rate": wins / completed, "gate_games": completed, "gate_truncated": cutoffs}


def save(path, model, ema, optimizer, config, iteration, stage):
    temporary = path.with_suffix(".tmp")
    torch.save({"config": asdict(config), "model": model.state_dict(), "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(), "iteration": iteration, "stage": stage,
                "random": torch.get_rng_state(),
                "cuda_random": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("check", "paper"), default="check")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/nebula"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    loaded = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    config = Config(**loaded["config"]) if loaded else profile(args.profile)
    if args.updates is not None:
        config = replace(config, updates=args.updates)
    config.validate()
    torch.manual_seed(config.seed)
    model = Model(config).to(device)
    ema = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    start, stage = 0, 0
    if loaded:
        model.load_state_dict(loaded["model"])
        ema.load_state_dict(loaded["ema"])
        optimizer.load_state_dict(loaded["optimizer"])
        start, stage = loaded["iteration"], loaded["stage"]
        torch.set_rng_state(loaded["random"])
        if device.type == "cuda" and loaded["cuda_random"]:
            torch.cuda.set_rng_state_all(loaded["cuda_random"])
    args.output.mkdir(parents=True, exist_ok=True)
    count = 2 * config.environments * config.steps
    storage = count * (38 * config.side ** 2 * 2 + 9 * config.side ** 2 + 2 * 512 * 4)
    print(json.dumps({"device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
                      "rollout_storage_gib": round(storage / 1024 ** 3, 3), "start": start, "config": asdict(config)}), flush=True)
    iteration = start
    # resuming starts fresh games but restores weights, optimizer, schedules, EMA and curriculum stage.
    with Arena(config.environments, config.side, config.seed + start, args.library) as arena:
        arena.reset(config.curriculum[stage], config.horizon)
        with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as log:
            try:
                for iteration in range(start, config.updates):
                    began = time.perf_counter()
                    batch, metrics = collect(model, arena, config, stage, device)
                    metrics.update(update(model, optimizer, batch, config, iteration, device))
                    del batch
                    average(ema, model, config.ema)
                    if config.evaluate and (iteration + 1) % config.evaluate == 0:
                        gate = evaluate(model, config, stage, device, args.library, config.seed + 100000 + iteration)
                        metrics.update(gate)
                        if gate["gate_win_rate"] >= config.threshold and stage + 1 < len(config.curriculum):
                            stage += 1
                            # ongoing games finish at their current distance. new games use the next range.
                    metrics.update(iteration=iteration + 1, stage=stage, seconds=time.perf_counter() - began)
                    line = json.dumps(metrics, allow_nan=False)
                    print(line, flush=True)
                    log.write(line + "\n")
                    log.flush()
                    if (iteration + 1) % config.save == 0 or iteration + 1 == config.updates:
                        save(args.output / "checkpoint.pt", model, ema, optimizer, config, iteration + 1, stage)
            except KeyboardInterrupt:
                print("Interrupted. The latest completed checkpoint remains in the output directory.", flush=True)


if __name__ == "__main__":
    main()
