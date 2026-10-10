"""Export native GenFormer weights and observation memory for standalone LibTorch CPU inference."""

import argparse
from pathlib import Path

import jax
import numpy as np
import torch
from storage import load
from torch import Tensor, nn
from torch.nn import functional as F


def tensor(value):
    return torch.from_numpy(np.array(jax.device_get(value), dtype=np.float32, copy=True))


def linear(source):
    layer = nn.Linear(source.weight.shape[1], source.weight.shape[0])
    layer.weight = nn.Parameter(tensor(source.weight), requires_grad=False)
    layer.bias = nn.Parameter(tensor(source.bias), requires_grad=False)
    return layer


def conv(source):
    layer = nn.Conv2d(source.weight.shape[1], source.weight.shape[0], 3, padding=1)
    layer.weight = nn.Parameter(tensor(source.weight), requires_grad=False)
    layer.bias = nn.Parameter(tensor(source.bias), requires_grad=False)
    return layer


def number(value: Tensor) -> Tensor:
    return torch.stack((value / 1024, value.sign() * value.abs().log1p() / 8), dim=-1)


def categories(value: Tensor, count: int) -> Tensor:
    return F.one_hot(value.to(torch.int64), count).to(torch.float32)


def adjacent(value: Tensor) -> Tensor:
    padded = F.pad(value, [1, 1, 1, 1])
    return torch.stack((padded[..., :-2, 1:-1], padded[..., 2:, 1:-1],
                        padded[..., 1:-1, :-2], padded[..., 1:-1, 2:]), dim=-3)


def coverage(value: Tensor) -> Tensor:
    rows, cols = value.shape[-2:]
    return F.avg_pool2d(value.float().reshape(-1, 1, rows, cols), 3, 1, 1,
                        divisor_override=1).reshape(value.shape)


class Norm(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.weight = nn.Parameter(tensor(source.weight), requires_grad=False)
        self.bias = nn.Parameter(tensor(source.bias), requires_grad=False)

    def forward(self, value: Tensor) -> Tensor:
        centered = value - value.mean(-1, keepdim=True)
        return centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + 1e-5) * self.weight + self.bias


class History(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.cell = linear(source.cell)
        self.public = nn.Sequential(linear(source.public[0]), nn.SiLU(), linear(source.public[1]))
        self.memory, self.norm = linear(source.memory), Norm(source.norm)
        self.qkv, self.output = linear(source.qkv), linear(source.output)
        self.heads = source.heads

    def forward(self, board: Tensor, scalars: Tensor, valid: Tensor, memory: Tensor, inside: Tensor):
        batch, times, _, rows, cols = board.shape
        area = rows * cols
        terrain, owner, army = board[:, :, 0], board[:, :, 1], board[:, :, 2]
        visible = (terrain != 0) & (terrain != 5) & inside[:, None]
        cells = torch.arange(area).reshape(1, 1, rows, cols)
        action = scalars[:, :, 5].long()
        source = cells == (action % area)[:, :, None, None]
        kind = action // area
        move = (action >= 0) & (action < 8 * area)
        submitted = (kind[:, :, None] == torch.arange(8)) & move[:, :, None]
        commands = submitted[:, :, None, None, :] * source[..., None]
        fields = torch.cat((categories(terrain, 6), categories(owner, 3), number(army),
                            inside[:, None, :, :, None].expand(batch, times, rows, cols, 1),
                            visible[..., None], coverage((owner == 1) & visible)[..., None] / 9,
                            coverage((owner == 2) & visible)[..., None] / 9, commands), dim=-1)
        fields = (fields * inside[:, None, :, :, None]).reshape(batch, times, area, 23)
        age = (scalars[:, :1, 0] - scalars[:, :, 0]).clamp_min(0)
        height = inside.any(2).sum(1) / 32
        width = inside.any(1).sum(1) / 32
        public = torch.cat((number(scalars[:, :, :5]).reshape(batch, times, 10),
                            scalars[:, :, :1] % 2 / 2, scalars[:, :, :1] % 50 / 50,
                            number(age), (action >= 0)[..., None], (action == 8 * area)[..., None],
                            height[:, None, None].expand(batch, times, 1),
                            width[:, None, None].expand(batch, times, 1)), dim=-1)
        terrain, owner, army = memory[:, 0], memory[:, 1], memory[:, 2]
        stamp, exposure, enemy_army, enemy_stamp = memory[:, 3], memory[:, 4], memory[:, 5], memory[:, 6]
        seen, exposed, enemy_seen = stamp >= 0, exposure >= 0, enemy_stamp >= 0
        tick = scalars[:, 0, 0, None, None]
        age = torch.where(seen, tick - stamp, 0).clamp_min(0)
        exposure_age = torch.where(exposed, tick - exposure, 0).clamp_min(0)
        enemy_age = torch.where(enemy_seen, tick - enemy_stamp, 0).clamp_min(0)
        record = torch.cat((categories(terrain, 6), categories(owner, 3), number(army), number(age),
                            seen[..., None], number(exposure_age), exposed[..., None], number(enemy_army),
                            number(enemy_age), enemy_seen[..., None]), dim=-1).reshape(batch, area, 22)
        known = (seen | exposed).reshape(batch, area)
        summary = self.public(public)
        encoded = self.cell(fields) + summary[:, :, None]
        current = encoded[:, 0]
        sequence = torch.cat((encoded, self.memory(record)[:, None]), dim=1).transpose(1, 2)
        available = torch.cat((valid[:, :, None].expand(batch, times, area), known[:, None]), dim=1)
        sequence = self.norm(sequence).reshape(batch * area, times + 1, -1)
        width = current.shape[-1]
        projected = self.qkv(sequence).reshape(batch * area, times + 1, 3, self.heads, width // self.heads)
        query, key, value = projected[:, 0, 0], projected[:, :, 1], projected[:, :, 2]
        scores = (query[:, None] * key).sum(-1) * (width // self.heads)**-0.5
        available = available.transpose(1, 2).reshape(batch * area, times + 1)
        weights = scores.masked_fill(~available[..., None], -float('inf')).softmax(1)
        correction = (weights[..., None] * value).sum(1).reshape(batch, area, width)
        return (current + self.output(correction)) * inside.reshape(batch, area, 1), summary[:, 0]


class Bias(nn.Module):
    def __init__(self, source, side):
        super().__init__()
        self.coefficients, self.source, self.target = (linear(source.coefficients), linear(source.source), linear(source.target))
        self.heads, self.templates = source.heads, source.templates
        row, col = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
        positions = torch.stack((row.flatten(), col.flatten()), dim=-1)
        relative = positions[None] - positions[:, None]
        pairs = (relative[..., 0] + side - 1) * (2 * side - 1) + relative[..., 1] + side - 1
        delta = torch.arange(1 - side, side)
        offsets = torch.stack(torch.meshgrid(delta, delta, indexing="ij"), dim=-1).reshape(-1, 2) / 32
        # Geometry depends only on fixed weights and the padded board size, so evaluate it once at export.
        with torch.no_grad():
            geometry = linear(source.geometry[1])(F.silu(linear(source.geometry[0])(offsets)))
        self.register_buffer("geometry", geometry)
        self.register_buffer("pair_geometry", geometry[pairs])
        self.register_buffer("pairs", pairs)

    def forward(self, cells: Tensor, context: Tensor, relations: Tensor):
        batch, area = cells.shape[:2]
        weights = self.coefficients(context).reshape(batch, self.heads, self.templates + 3)
        source = self.source(cells).reshape(batch, area, self.heads, self.templates + 3)
        target = self.target(cells).reshape(batch, area, self.heads, self.templates + 3)
        bias = (weights[:, :, :self.templates] @ self.geometry.T)[:, :, self.pairs]
        bias = bias + torch.einsum("bihr,ijr->bhij", source[..., :self.templates], self.pair_geometry)
        bias = bias + torch.einsum("bjhr,ijr->bhij", target[..., :self.templates], self.pair_geometry)
        for index in range(3):
            slot = self.templates + index
            coefficient = (weights[:, :, slot, None, None] + source[..., slot].transpose(1, 2)[..., None]
                           + target[..., slot].transpose(1, 2)[:, :, None])
            bias = bias + coefficient * relations[:, index, None]
        return bias


class Block(nn.Module):
    def __init__(self, source, side):
        super().__init__()
        self.heads, self.norm = source.heads, Norm(source.norm)
        self.pool, self.qkv, self.output = linear(source.pool), linear(source.qkv), linear(source.output)
        self.bias = Bias(source.bias, side)
        self.local, self.project = conv(source.local), conv(source.project)
        self.feed = nn.Sequential(Norm(source.feed[0]), linear(source.feed[1]), nn.SiLU(), linear(source.feed[2]))

    def forward(self, cells: Tensor, inside: Tensor, valid: Tensor, relations: Tensor):
        batch, count, width = cells.shape
        rows, cols = inside.shape[-2:]
        area = rows * cols
        normalized = self.norm(cells)
        projected = self.qkv(normalized).reshape(batch, count, 3, self.heads, width // self.heads)
        query, key, value = projected[:, :, 0], projected[:, :, 1], projected[:, :, 2]
        scores = self.pool(normalized)[..., 0].masked_fill(~valid, -float('inf'))
        context = (normalized * scores.softmax(1)[..., None]).sum(1)
        bias = F.pad(self.bias(normalized[:, :area], context, relations), [0, count - area, 0, count - area])
        bias = bias.masked_fill(~valid[:, None, None, :], -float('inf'))
        attended = F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=bias)
        cells = (cells + self.output(attended.transpose(1, 2).reshape(batch, count, width))) * valid[..., None]
        grid = cells[:, :area].transpose(1, 2).reshape(batch, width, rows, cols)
        local = self.project(F.silu(self.local(grid)) * inside[:, None]) * inside[:, None]
        cells = cells + F.pad(local.reshape(batch, width, area).transpose(1, 2), [0, 0, 0, count - area])
        return (cells + self.feed(cells)) * valid[..., None]


class Network(nn.Module):
    """Float32 CPU evaluation of model.py with the same weights, relations and action ordering."""

    def __init__(self, source, config):
        super().__init__()
        self.side, self.width = config.side, config.width
        self.history = History(source.history)
        self.scores = nn.Sequential(linear(source.scores[0]), nn.SiLU(), linear(source.scores[1]))
        self.register_buffer("identity", tensor(source.identity))
        self.blocks = nn.ModuleList([Block(block, config.side) for block in source.blocks])
        self.norm, self.context = Norm(source.norm), linear(source.context)
        self.register_buffer("action", tensor(source.action))
        self.moves = nn.ModuleList([linear(part) for part in source.moves])
        self.source, self.target = linear(source.source), linear(source.target)
        self.wait, self.critic = linear(source.wait), linear(source.critic)
        self.register_buffer("centers", torch.linspace(-1, 1, config.bins))
        row, col = torch.meshgrid(torch.arange(self.side), torch.arange(self.side), indexing="ij")
        positions = torch.stack((row.flatten(), col.flatten()), dim=-1)
        relative = positions[None] - positions[:, None]
        self.register_buffer("neighbor", relative.abs().sum(-1) == 1)
        self.register_buffer("sight", relative.abs().amax(-1) <= 1)

    def forward(self, board: Tensor, scalars: Tensor, valid: Tensor, memory: Tensor,
                inside: Tensor, legal: Tensor, timeline: Tensor):
        batch, area, width = board.shape[0], self.side * self.side, self.width
        cells, public = self.history(board, scalars, valid, memory, inside)
        observed = timeline[:, :, 0] >= 0
        age = (scalars[:, :1, 0] - timeline[:, :, 0]).clamp_min(0)
        numbers = number(timeline[:, :, 1:].transpose(1, 2))
        dates = number(age)[:, None].expand(batch, 4, timeline.shape[1], 2)
        flags = observed[:, None, :, None].expand(batch, 4, timeline.shape[1], 1)
        series = torch.cat((numbers, dates, flags), dim=-1) * flags
        temporal = self.scores(series.reshape(batch, 4, -1)) + self.identity[None]
        cells = torch.cat((cells, temporal), dim=1)
        valid = F.pad(inside.reshape(batch, area), [0, 4], value=1.)
        visible = (board[:, 0, 0] != 0) & (board[:, 0, 0] != 5)
        terrain = torch.where(visible, board[:, 0, 0], memory[:, 0]).reshape(batch, area)
        known = (visible | (memory[:, 3] >= 0)).reshape(batch, area)
        possible = inside.reshape(batch, area) & ~(known & (terrain == 2))
        confirmed = possible & known
        relations = torch.stack((self.neighbor[None] & confirmed[:, :, None] & confirmed[:, None, :],
                                 self.neighbor[None] & possible[:, :, None] & possible[:, None, :],
                                 self.sight[None].expand(batch, area, area)), dim=1)
        for block in self.blocks:
            cells = block(cells, inside, valid, relations)
        cells = self.norm(cells) * valid[..., None]
        mean = cells.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        maximum = cells.masked_fill(~valid[..., None], -float('inf')).amax(1)
        context = F.silu(self.context(torch.cat((mean, maximum, public), dim=-1)))
        cells = cells[:, :area]
        source_weight, target_weight, command_weight = self.moves[0].weight.split(width, dim=1)
        source = (cells @ source_weight.T)[:, None]
        grid = (cells @ target_weight.T).transpose(1, 2).reshape(batch, width, self.side, self.side)
        target = adjacent(grid).permute(0, 2, 3, 4, 1).reshape(batch, 4, area, width).repeat(1, 2, 1, 1)
        command = ((self.action[None] + context[:, None]) @ command_weight.T + self.moves[0].bias)[:, :, None]
        moves = self.moves[1](F.silu(source + target + command)).reshape(batch, 8 * area)
        source = self.source(cells).reshape(batch, area, 2, width).transpose(1, 2)
        grid = self.target(cells).transpose(1, 2).reshape(batch, 2 * width, self.side, self.side)
        target = adjacent(grid).permute(0, 2, 3, 4, 1).reshape(batch, 4, area, 2, width).permute(0, 3, 1, 2, 4)
        moves = moves + ((source[:, :, None] * target).sum(-1) * width**-0.5).reshape(batch, 8 * area)
        logits = torch.cat((moves, self.wait(context)), dim=1).masked_fill(~legal, -1e9)
        value_logits = self.critic(context)
        value = (value_logits.softmax(-1) * self.centers).sum(-1)
        return logits, value, value_logits


class Policy(nn.Module):
    """One player's persistent memory and network; reset starts a new board and submitted records its command."""

    def __init__(self, source, config):
        super().__init__()
        self.network = Network(source, config)
        self.side = config.side
        self.tick, self.action = -1, -1
        self.register_buffer("board", torch.zeros(1, config.history + 1, 3, self.side, self.side))
        self.register_buffer("scalars", torch.zeros(1, config.history + 1, 6))
        self.register_buffer("valid", torch.zeros(1, config.history + 1, dtype=torch.bool))
        self.register_buffer("memory", torch.zeros(1, 7, self.side, self.side))
        self.register_buffer("timeline", torch.zeros(1, config.temporal, 5))
        self.register_buffer("inside", torch.zeros(1, self.side, self.side, dtype=torch.bool))
        self.reset(self.side, self.side)

    @torch.jit.export
    def reset(self, rows: int, cols: int):
        if not (0 < rows <= self.side and 0 < cols <= self.side):
            raise ValueError("Board exceeds the model input dimensions")
        self.board.zero_()
        self.scalars.zero_()
        self.valid.zero_()
        self.memory.zero_()
        self.memory[:, 3].fill_(-1)
        self.memory[:, 4].fill_(-1)
        self.memory[:, 6].fill_(-1)
        self.timeline.zero_()
        self.timeline[:, :, 0].fill_(-1)
        self.inside.zero_()
        self.inside[:, :rows, :cols] = True
        self.tick, self.action = -1, -1

    @torch.jit.export
    def submitted(self, action: int):
        if not 0 <= action <= 8 * self.side * self.side:
            raise ValueError("Action index outside the policy head")
        self.action = action

    @torch.jit.export
    def encode(self, raw: Tensor, stats: Tensor):
        terrain, owner, army = raw[:, 0], raw[:, 1], raw[:, 2]
        tick = int(stats[0, 0].item())
        reset = self.tick < 0 or tick < self.tick
        if reset:
            rows = int(self.inside.any(2).sum().item())
            cols = int(self.inside.any(1).sum().item())
            action = self.action
            self.reset(rows, cols)
            self.action = action
        visible = self.inside & (terrain != 0) & (terrain != 5)
        owner, army = torch.where(visible, owner, 0), torch.where(visible, army, 0)
        if reset or tick != self.tick:
            current = torch.stack((terrain, owner, army), dim=1) * self.inside[:, None]
            previous = -1 if reset else self.action
            scalar = torch.cat((stats, torch.full((1, 1), float(previous))), dim=1)
            self.board = torch.cat((current[:, None], self.board[:, :-1]), dim=1)
            self.scalars = torch.cat((scalar[:, None], self.scalars[:, :-1]), dim=1)
            self.valid = torch.cat((torch.ones_like(self.valid[:, :1]), self.valid[:, :-1]), dim=1)
            self.timeline = torch.cat((stats[:, None], self.timeline[:, :-1]), dim=1)
            self.memory[:, :3] = torch.where(visible[:, None], current, self.memory[:, :3])
            self.memory[:, 3] = torch.where(visible, tick, self.memory[:, 3])
            exposed = (coverage((owner == 2) & visible) > 0) & self.inside
            self.memory[:, 4] = torch.where(exposed, tick, self.memory[:, 4])
            enemy = (owner == 2) & visible
            self.memory[:, 5] = torch.where(enemy, army, self.memory[:, 5])
            self.memory[:, 6] = torch.where(enemy, tick, self.memory[:, 6])
        self.tick = tick
        mountain = ((self.memory[:, 0] == 2) & (self.memory[:, 3] >= 0)) | (terrain == 2)
        moves = (self.inside & (owner == 1) & (army >= 2))[:, None] & adjacent(self.inside & ~mountain)
        legal = torch.cat((moves.repeat(1, 2, 1, 1).reshape(1, -1), torch.ones((1, 1), dtype=torch.bool)), dim=1)
        return self.board, self.scalars, self.valid, self.memory, self.inside, legal, self.timeline

    def forward(self, raw: Tensor, stats: Tensor):
        board, scalars, valid, memory, inside, legal, timeline = self.encode(raw, stats)
        return self.network(board, scalars, valid, memory, inside, legal, timeline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", choices=("selected", "model", "ema"), default="selected")
    args = parser.parse_args()
    if args.output.suffix != ".pt" or args.output.resolve() == args.checkpoint.resolve():
        parser.error("Output must be a separate .pt file")
    torch.set_num_threads(1)
    backend = jax.devices("cpu")[0]
    config, model = load(args.checkpoint, backend, args.weights)[1:]
    policy = torch.jit.script(Policy(model, config).eval())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    policy.save(str(args.output))
    print(args.output)


if __name__ == "__main__":
    main()
