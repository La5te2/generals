"""Export EMA inference weights to a TorchScript archive for the Nebula executable."""

import argparse
from pathlib import Path

import torch
from config import Config
from model import Model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = Model(Config(**saved["config"]).validate()).eval()
    model.load_state_dict(saved["ema"])
    scripted = torch.jit.script(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(args.output))
    print(args.output)


if __name__ == "__main__":
    main()
