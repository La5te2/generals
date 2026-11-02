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
from model import Config, Network, army_classes

FORMAT = "1015-unified-memory-v2"


def save(
    path,
    model,
    metadata,
    optimizer_state=None,
    runtime=None,
    opponents=None,
    critic=None,
    critic_optimizer=None,
):
    """Replace one archive atomically; full checkpoints include rollout and pool state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            "metadata.json",
            json.dumps(dict(metadata, format=FORMAT, config=asdict(model.config))),
        )
        for name, tree in (
            ("model", model),
            ("optimizer", optimizer_state),
            ("runtime", runtime),
            ("opponents", opponents),
            ("critic", critic),
            ("critic_optimizer", critic_optimizer),
        ):
            if tree is not None:
                stream = io.BytesIO()
                eqx.tree_serialise_leaves(stream, tree)
                archive.writestr(name + ".eqx", stream.getvalue())
    os.replace(temporary, path)


def load(path):
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        if metadata["format"] != FORMAT:
            raise ValueError(
                "Expected a 1015-unified-memory-v2 archive"
            )
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


def state_losses(output, inputs, targets):
    """Joint owner/army-bin likelihood; labels select a loss, never actor features."""
    hidden = inputs.board & (inputs.cells[..., 0] == 0) & (inputs.cells[..., 23] == 0)
    owner = optax.softmax_cross_entropy_with_integer_labels(
        output.owner, targets["owner"]
    )
    conditional = jnp.take_along_axis(
        output.army, targets["owner"][..., None, None].astype(jnp.int32), -2
    )[..., 0, :]
    army = optax.softmax_cross_entropy_with_integer_labels(
        conditional, army_classes(targets["army"])
    )
    count = jnp.maximum(hidden.sum(-1), 1)
    owner = jnp.sum(owner * hidden, -1) / count
    army = jnp.sum(army * hidden, -1) / count
    general = optax.softmax_cross_entropy_with_integer_labels(
        output.general, jnp.maximum(targets["general"], 0)
    )
    return owner, army, jnp.where(targets["general"] >= 0, general, 0)


def opponent_loss(output, inputs, targets):
    """Predict the frozen opponent's command from our legal history."""
    return optax.softmax_cross_entropy_with_integer_labels(
        output.enemy_action, targets["enemy_action"]
    )
