"""Patch attention combines board memory with two public-score histories for policy and value prediction."""

import torch
from config import Config
from torch import nn
from torch.nn import functional as F


def inputs(arrays, device):
    board, legal, history = arrays
    # collection and PPO see exactly the same quantized board values.
    return (torch.from_numpy(board).to(dtype=torch.bfloat16).to(device=device, dtype=torch.float32),
            torch.from_numpy(legal).to(device=device, dtype=torch.bool),
            torch.from_numpy(history).to(device=device))


class Block(nn.Module):
    def __init__(self, width, heads, expansion):
        super().__init__()
        self.heads = heads
        self.first = nn.LayerNorm(width)
        self.second = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.feed = nn.Sequential(nn.Linear(width, expansion * width), nn.SiLU(),
                                  nn.Linear(expansion * width, width))

    def forward(self, tokens):
        batch, length, width = tokens.shape
        projected = self.qkv(self.first(tokens)).reshape(batch, length, 3, self.heads, width // self.heads)
        query, key, value = projected.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(query, key, value)
        tokens = tokens + self.output(attended.transpose(1, 2).reshape(batch, length, width))
        return tokens + self.feed(self.second(tokens))


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.side = config.side
        self.patch = config.patch
        self.width = config.width
        self.embed = nn.Linear(38 * self.patch * self.patch, self.width)
        self.army = nn.Sequential(nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, self.width))
        self.land = nn.Sequential(nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, self.width))
        self.valueToken = nn.Parameter(torch.randn(1, 1, self.width) * 0.02)
        self.types = nn.Parameter(torch.randn(1, 2, self.width) * 0.02)
        count = (self.side // self.patch) ** 2 + 3
        self.positions = nn.Parameter(torch.empty(1, count, self.width))
        nn.init.trunc_normal_(self.positions, std=0.1, a=-0.2, b=0.2)
        self.blocks = nn.ModuleList([Block(self.width, config.heads, config.expansion) for layer in range(config.depth)])
        self.norm = nn.LayerNorm(self.width)
        self.policy = nn.Linear(self.width, 9 * self.patch * self.patch)
        self.critic = nn.Linear(self.width, config.bins)
        self.register_buffer("centers", torch.linspace(-1, 1, config.bins))
        divisors = torch.ones(38)
        divisors[[0, 1, 2, 3, 14, 16, 17, 18, 19, 20] + list(range(24, 38))] = 50
        self.register_buffer("divisors", divisors.reshape(1, 38, 1, 1))

    def forward(self, spatial: torch.Tensor, legal: torch.Tensor, temporal: torch.Tensor):
        batch = spatial.shape[0]
        grid, patch = self.side // self.patch, self.patch
        board = spatial / self.divisors
        # each patch keeps channel, local row, local column order. logits use the inverse arrangement.
        board = board.reshape(batch, 38, grid, patch, grid, patch)
        board = board.permute(0, 2, 4, 1, 3, 5).reshape(batch, grid * grid, 38 * patch * patch)
        history = torch.stack((self.army(temporal[:, 0] / 50), self.land(temporal[:, 1] / 50)), dim=1)
        tokens = torch.cat((self.valueToken.expand(batch, -1, -1), history + self.types, self.embed(board)), dim=1)
        tokens = tokens + self.positions
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        valueLogits = self.critic(tokens[:, 0]).float()
        value = (valueLogits.softmax(-1) * self.centers).sum(-1)
        logits = self.policy(tokens[:, 3:]).reshape(batch, grid, grid, 9, patch, patch)
        logits = logits.permute(0, 3, 1, 4, 2, 5).reshape(batch, -1).float()
        logits = logits.masked_fill(~legal.reshape(batch, -1).to(torch.bool), -1e9)
        return logits, value, valueLogits
