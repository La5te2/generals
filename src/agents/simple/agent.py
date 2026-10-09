"""Serve the existing five-integer action protocol using observation-only CPU inference."""

import argparse
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from features import Encoder, View, encode_action
from storage import load


def read_view(stream, rows, cols, side):
    line = stream.readline()
    if not line:
        return None
    stats = list(map(int, line.split()))
    if len(stats) != 5 or min(stats) < 0:
        raise ValueError("Expected five nonnegative observation statistics")
    grids = []
    for name in ("terrain", "owner", "army"):
        values = []
        for row in range(rows):
            line = stream.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in {name}")
            cells = list(map(int, line.split()))
            if len(cells) != cols or min(cells) < 0:
                raise ValueError(f"Invalid {name} row")
            values.append(cells)
        grids.append(np.asarray(values, dtype=np.int64))
    terrain, owner, army = grids
    if (terrain > 5).any() or (owner > 2).any():
        raise ValueError("Invalid terrain or ownership code")
    hidden = (terrain == 0) | (terrain == 5)
    if (owner[hidden] != 0).any() or (army[hidden] != 0).any():
        raise ValueError("Hidden cells must contain zero ownership and army placeholders")
    padded = [np.zeros((1, side, side), dtype=np.float32) for grid in range(3)]
    padded[0].fill(2)
    for target, source in zip(padded, grids, strict=True):
        target[0, :rows, :cols] = source
    inside = np.zeros((1, side, side), dtype=bool)
    inside[0, :rows, :cols] = True
    return View(*(jnp.asarray(value) for value in (*padded, np.asarray([stats], np.float32), inside)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--weights", choices=("selected", "model", "ema"), default="selected")
    args = parser.parse_args()
    try:
        backend = jax.devices("cpu")[0]
        config, model = load(args.checkpoint, backend, args.weights)[1:]
        infer = eqx.filter_jit(lambda model, inputs: model(*inputs)[0])
        encode = jax.jit(lambda encoder, view, reset: encoder.update(view, reset))
        line = sys.stdin.readline()
        if not line:
            return
        fields = list(map(int, line.split()))
        if len(fields) != 3:
            raise ValueError("Expected player rows cols")
        player, rows, cols = fields
        if player not in (0, 1) or not 1 <= rows <= config.side or not 1 <= cols <= config.side:
            raise ValueError("Board exceeds the checkpoint input dimensions or player is invalid")
        with jax.default_device(backend):
            encoder = Encoder.empty(1, config.side, config.history, config.temporal)
            reset = jnp.ones(1, jnp.bool_)
            key = jax.random.PRNGKey(config.seed)
            while (view := read_view(sys.stdin, rows, cols, config.side)) is not None:
                encoder, inputs = encode(encoder, view, reset)
                logits = infer(model, inputs)
                key, choice = jax.random.split(key)
                action = jax.random.categorical(choice, logits).astype(jnp.int32) if args.sample else logits.argmax(1)
                encoder = encoder.submitted(action)
                print(*encode_action(int(action[0]), config.side), flush=True)
                reset = jnp.zeros_like(reset)
    except (ValueError, OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
