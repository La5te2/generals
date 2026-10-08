"""Store observations compactly and reconstruct unchanged model inputs for each PPO minibatch."""

import torch
from torch.nn import functional as F


class Rollout:
    def __init__(self, steps, count, side, device):
        self.count, self.side = count, side
        self.device = torch.device(device)
        self.spatial = torch.empty((steps * count, 6, side, side), dtype=torch.bfloat16, device=device)
        self.markers = torch.empty((steps * count, 2, side, side), dtype=torch.uint8, device=device)
        self.scalars = torch.empty((steps * count, 6), dtype=torch.bfloat16, device=device)
        self.coordinates = torch.empty((2, side, side), dtype=torch.bfloat16, device=device)
        # keep each new difference and score once, preceded by the history at the rollout boundary.
        self.differences = torch.empty((steps + 6, count, 2, side, side), dtype=torch.bfloat16, device=device)
        self.scores = torch.empty((steps + 511, count, 2), device=device)
        self.starts = torch.empty((steps, count), dtype=torch.long, device=device)

    @property
    def nbytes(self):
        return sum(array.numel() * array.element_size() for array in (
            self.spatial, self.markers, self.scalars, self.coordinates, self.differences, self.scores, self.starts))

    def write(self, step, raw, reset):
        board, legal, history = raw
        section = slice(step * self.count, (step + 1) * self.count)
        self.spatial[section].copy_(board[:, [0, 1, 2, 3, 20, 21]])
        self.scalars[section].copy_(board[:, 14:20, 0, 0])
        # ten binary board channels and four move directions fit in two bytes per cell.
        # full and half moves share the same mask; every pass is legal.
        flags = torch.cat((board[:, 4:14].to(torch.uint8), legal[:, :4].to(torch.uint8)), dim=1)
        flags = F.pad(flags, (0, 0, 0, 0, 0, 2)).reshape(self.count, 2, 8, self.side, self.side)
        bits = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=board.device)
        self.markers[section].copy_((flags * bits[None, None, :, None, None]).sum(2, dtype=torch.uint8))
        if step == 0:
            self.coordinates.copy_(board[0, 22:24])
            self.differences[:6].copy_(torch.stack((board[:, 25:31], board[:, 32:38]), dim=2).flip(1).transpose(0, 1))
            self.scores[:511].copy_(history[:, :, :-1].permute(2, 0, 1))
            self.starts[0].fill_(-511)
        else:
            # a reset discards the previous game's history, including after time-limit truncation.
            self.starts[step].copy_(torch.where(reset.to(self.device), step, self.starts[step - 1]))
        self.differences[step + 6].copy_(board[:, [24, 31]])
        self.scores[step + 511].copy_(history[:, :, -1])

    def read(self, ids, device):
        device = torch.device(device)
        ids = ids.to(self.device)
        steps, players = ids // self.count, ids % self.count
        starts = self.starts[steps, players]

        def transfer(array):
            # pin only the selected minibatch, not the multi-gigabyte host buffer.
            if self.device.type == "cpu" and device.type == "cuda":
                return array.pin_memory().to(device, non_blocking=True)
            return array.to(device)

        board = torch.empty((ids.numel(), 38, self.side, self.side), device=device)
        board[:, [0, 1, 2, 3, 20, 21]] = transfer(self.spatial[ids]).float()
        board[:, 14:20] = transfer(self.scalars[ids])[:, :, None, None]
        board[:, 22:24] = transfer(self.coordinates)
        markers = transfer(self.markers[ids])
        bits = torch.arange(8, dtype=torch.uint8, device=device)
        flags = ((markers[:, :, None] >> bits[None, None, :, None, None]) & 1).reshape(-1, 16, self.side, self.side)
        board[:, 4:14] = flags[:, :10]
        moves = flags[:, 10:14].bool()
        legal = torch.cat((moves, moves, torch.ones_like(moves[:, :1])), dim=1)

        times = steps[:, None] + 6 - torch.arange(7, device=self.device)
        differences = self.differences[times, players[:, None]]
        differences = transfer(differences)
        valid = transfer(times - 6 >= starts[:, None])
        differences = differences.masked_fill(~valid[:, :, None, None, None], 0)
        board[:, 24:38] = differences.permute(0, 2, 1, 3, 4).flatten(1, 2)

        times = steps[:, None] + torch.arange(512, device=self.device)
        history = transfer(self.scores[times, players[:, None]])
        valid = transfer(times - 511 >= starts[:, None])
        history = history.masked_fill(~valid[:, :, None], 0).transpose(1, 2)
        return board, legal, history
