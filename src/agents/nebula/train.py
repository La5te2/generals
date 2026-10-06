"""Self-play, PPO updates and checkpoint storage run independently of the graphical application."""

import argparse
import copy
import json
import time
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

import torch
from config import Config, profile
from env import Arena
from model import Model, inputs
from ppo import advantages, average, objective


def precision(device, config):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and config.bf16 else nullcontext()


@torch.inference_mode()
def collect(model, arena, config, device, storage):
    count, steps, side = 2 * arena.count, config.steps, config.side
    # keep sampling and optimization on the selected device. CPU storage is an explicit memory-saving option.
    board = torch.empty((steps, count, 38, side, side), dtype=torch.bfloat16, device=storage)
    legal = torch.empty((steps, count, 9, side, side), dtype=torch.bool, device=storage)
    history = torch.empty((steps, count, 2, 512), device=storage)
    actions = torch.empty((steps, count), dtype=torch.long, device=storage)
    logprobs, values, rewards, limitValues = [torch.zeros(steps, count, device=storage) for field in range(4)]
    terminal, truncated = [torch.zeros(steps, count, dtype=torch.bool, device=storage) for field in range(2)]
    for step in range(steps):
        raw = arena.read()
        board[step].copy_(raw[0])
        legal[step].copy_(raw[1])
        history[step].copy_(raw[2])
        with precision(device, config):
            logits, prediction = model(*inputs(raw, device))[:2]
        distribution = torch.distributions.Categorical(logits=logits)
        choice = distribution.sample()
        actions[step].copy_(choice)
        logprobs[step].copy_(distribution.log_prob(choice))
        values[step].copy_(prediction)
        reward, ended, limited = arena.step(choice)
        rewards[step].copy_(reward)
        terminal[step].copy_(ended)
        truncated[step].copy_(limited)
        if limited.any():
            # final() retains the time-limited game, while read() already contains its replacement game.
            final = tuple(array[limited] for array in arena.final())
            with precision(device, config):
                finalValue = model(*inputs(final, device))[1]
            limitValues[step, limited.to(storage)] = finalValue.to(storage)
    with precision(device, config):
        finalValue = model(*inputs(arena.read(), device))[1]
    following = torch.cat((values[1:], finalValue.to(storage).unsqueeze(0)))
    following[truncated] = limitValues[truncated]
    advantage = advantages(rewards, values, following, terminal, truncated, config.gamma, config.gae)
    returns = advantage + values
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    batches = [board, legal, history, actions, logprobs, advantage, returns]
    return [data.flatten(0, 1) for data in batches], {
        "completed": int(terminal[:, ::2].sum()), "truncated": int(truncated[:, ::2].sum()),
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
        order = indices[torch.randperm(kept, device=indices.device)]
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
def evaluate(model, config, stage, device, seed):
    # a random legal mover is the curriculum gate from the released training recipe, rather than a strength benchmark.
    count = min(config.games, config.environments, 32)
    wins, completed, cutoffs = 0, 0, 0
    generator = torch.Generator(device=device).manual_seed(seed)
    players = torch.arange(count, device=device) * 2
    opponents = players + 1 - torch.arange(count, device=device) % 2
    candidates = players + torch.arange(count, device=device) % 2
    evaluation = replace(config, pool=min(config.pool, max(count, (config.maximum - config.minimum + 1) ** 2)))
    with Arena(count, evaluation, seed, device) as arena:
        arena.reset(config.curriculum[stage])
        while completed < config.games:
            raw = arena.read()
            with precision(device, config):
                logits = model(*inputs(raw, device))[0]
            moves = logits.argmax(-1)
            possible = raw[1][opponents, :8].flatten(1)
            # equal weights sample uniformly from legal moves. a blocked opponent uses a pass.
            weights = torch.cat((possible, ~possible.any(-1, keepdim=True)), dim=1).float()
            moves[opponents] = torch.multinomial(weights, 1, generator=generator).squeeze(1)
            rewards, ended, limited = arena.step(moves)
            finished = (ended[::2] | limited[::2]).nonzero().flatten()[:config.games - completed]
            wins += int((rewards[candidates[finished]] > 0).sum())
            cutoffs += int(limited[players[finished]].sum())
            completed += len(finished)
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
    parser.add_argument("--storage", choices=("device", "cpu"), default="device")
    parser.add_argument("--output", type=Path, default=Path("runs/nebula"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    storageDevice = torch.device("cpu") if args.storage == "cpu" else device
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
    print(json.dumps({"device": str(device), "environment": "generals-bots", "storage": str(storageDevice),
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "rollout_storage_gib": round(storage / 1024 ** 3, 3), "start": start, "config": asdict(config)}), flush=True)
    iteration = start
    # resuming starts fresh games but restores weights, optimizer, schedules, EMA and curriculum stage.
    with Arena(config.environments, config, config.seed + start, device) as arena:
        print("Preparing the JAX map pool and observation encoder...", flush=True)
        arena.reset(config.curriculum[stage])
        with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as log:
            try:
                for iteration in range(start, config.updates):
                    began = time.perf_counter()
                    batch, metrics = collect(model, arena, config, device, storageDevice)
                    metrics.update(update(model, optimizer, batch, config, iteration, device))
                    del batch
                    average(ema, model, config.ema)
                    previousStage = stage
                    if config.evaluate and (iteration + 1) % config.evaluate == 0:
                        gate = evaluate(model, config, stage, device, config.seed + 100000 + iteration)
                        metrics.update(gate)
                        if gate["gate_win_rate"] >= config.threshold and stage + 1 < len(config.curriculum):
                            stage += 1
                    if stage != previousStage or (iteration + 1) % config.refresh == 0:
                        # ongoing games keep their current boards. auto-resets draw from the refreshed pool.
                        arena.reset(config.curriculum[stage])
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
