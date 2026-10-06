"""Run a trained policy as an independent process using the existing five-integer action protocol."""

import argparse
import sys
from pathlib import Path

import torch
from config import Config
from environment import Encoder
from model import Model, inputs


def read(stream, count):
    line = stream.readline()
    if not line:
        raise EOFError("Incomplete observation")
    values = [int(token) for token in line.split()]
    if len(values) != count:
        raise ValueError("Unexpected protocol field count")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = Config(**saved["config"]).validate()
    model = Model(config).eval()
    model.load_state_dict(saved["ema"])
    player, rows, cols = read(sys.stdin, 3)
    memory = Encoder(config.side, rows, cols, player, args.library)
    try:
        with torch.inference_mode():
            for line in sys.stdin:
                values = [int(token) for token in line.split()]
                if len(values) != 5:
                    raise ValueError("Expected tick and four public scores")
                for row in range(3 * rows):
                    values.extend(read(sys.stdin, cols))
                logits = model(*inputs(memory.update(values), torch.device("cpu")))[0]
                choice = torch.distributions.Categorical(logits=logits).sample() if args.sample else logits.argmax(-1)
                kind, pos = divmod(int(choice.item()), config.side ** 2)
                row, col = divmod(pos, config.side)
                action = (1, 0, 0, 0, 0) if kind == 8 else (0, row, col, kind % 4, int(kind >= 4))
                print(*action, flush=True)
    finally:
        memory.close()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, EOFError) as error:
        print(f"agent: {error}", file=sys.stderr)
        sys.exit(1)
