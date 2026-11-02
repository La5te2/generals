"""Checkpoint persistence and auxiliary supervision for the shared legal encoder."""

import io
import json
import os
import zipfile
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from model import Config, Network


def save(path, model, metadata, optimizer_state=None, runtime=None, opponents=None):
    """Replace one archive atomically; full checkpoints include rollout and pool state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            "metadata.json",
            json.dumps(dict(metadata, format=1013, config=asdict(model.config))),
        )
        for name, tree in (
            ("model", model),
            ("optimizer", optimizer_state),
            ("runtime", runtime),
            ("opponents", opponents),
        ):
            if tree is not None:
                stream = io.BytesIO()
                eqx.tree_serialise_leaves(stream, tree)
                archive.writestr(name + ".eqx", stream.getvalue())
    os.replace(temporary, path)


def load(path):
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        if metadata["format"] != 1013:
            raise ValueError("Expected an agent-1013 policy or checkpoint")
        model = Network(Config(**metadata["config"]), key=jax.random.PRNGKey(0))
        model = eqx.tree_deserialise_leaves(
            io.BytesIO(archive.read("model.eqx")), model
        )
    return model, metadata


def restore(path, name, template):
    with zipfile.ZipFile(path) as archive:
        return eqx.tree_deserialise_leaves(
            io.BytesIO(archive.read(name + ".eqx")), template
        )


def optimizer(rate):
    return optax.chain(optax.clip_by_global_norm(0.5), optax.adam(rate))


def auxiliary(output, inputs, targets):
    """Predict hidden marginals; truth is used only as supervision, never as input."""
    hidden = inputs.board & (inputs.cells[..., 0] == 0)
    owner = optax.softmax_cross_entropy_with_integer_labels(
        output.owner, targets["owner"]
    )
    army = optax.huber_loss(output.army, targets["army"])
    spatial = jnp.sum((owner + army) * hidden, -1) / jnp.maximum(hidden.sum(-1), 1)
    general = optax.softmax_cross_entropy_with_integer_labels(
        output.general, jnp.maximum(targets["general"], 0)
    )
    return spatial + jnp.where(targets["general"] >= 0, general, 0)


def opponent_loss(output, inputs, targets):
    """Decode the frozen opponent's command and masked view from OUR history only."""
    command = optax.softmax_cross_entropy_with_integer_labels(
        output.enemy_action, targets["enemy_action"]
    )
    view = targets["enemy_view"]
    binary = optax.sigmoid_binary_cross_entropy(
        output.enemy_view[..., :9], view[..., :9]
    ).mean(-1)
    army = optax.huber_loss(output.enemy_view[..., 9], view[..., 9]) * view[..., 0]
    reconstruction = jnp.sum((binary + army) * inputs.board, -1) / jnp.maximum(
        inputs.board.sum(-1), 1
    )
    return command, reconstruction
