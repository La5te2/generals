"""Small CPU checks: observation boundaries, replay semantics and trainability."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from data import Simulator, split_for
from features import empty, encode, graph
from generals.core.game import create_initial_state, get_observation
from model import Config, Network
from ppo import Environment, Run, advantages, raw
from ppo import make_update as ppo_update
from train import epoch, load, make_update, optimizer, restore_optimizer, save, validate


def fixture(side=4, history=4):
    grid = (
        jnp.zeros((side, side), jnp.int32)
        .at[0, 0]
        .set(1)
        .at[-1, -1]
        .set(2)
        .at[1, 1]
        .set(-2)
    )
    state = create_initial_state(grid)
    state = state._replace(armies=state.armies.at[0, 0].set(8))
    topology = graph(side, side)
    obs = raw(state, 0, topology)
    x, memory = encode(
        obs, empty(side * side, history), jnp.int32(side * side * 8), topology
    )
    return state, topology, obs, x, memory


class AgentTests(unittest.TestCase):
    def test_observation_matches_environment(self):
        state, topology, obs, _, _ = fixture()
        view = get_observation(state, 0)
        np.testing.assert_array_equal(
            obs.cells[:, 0], (~(view.fog_cells | view.structures_in_fog)).reshape(-1)
        )
        np.testing.assert_array_equal(obs.cells[:, 1], view.owned_cells.reshape(-1))
        np.testing.assert_array_equal(
            obs.cells[:, 8], view.structures_in_fog.reshape(-1)
        )
        changed = state._replace(armies=state.armies.at[-1, -1].add(12))
        # Public total changes legitimately. Spatial view must remain identical.
        np.testing.assert_array_equal(obs.cells, raw(changed, 0, topology).cells)

    def test_hidden_state_invariance(self):
        state, topology, obs, x, _ = fixture(side=6)
        own = state.ownership.at[1, 5, 5].set(False).at[1, 4, 5].set(True)
        altered = state._replace(
            ownership=own,
            armies=state.armies.at[5, 5].set(0).at[4, 5].set(1),
            generals=state.generals.at[5, 5].set(False).at[4, 5].set(True),
        )
        other = raw(altered, 0, topology)
        for a, b in zip(obs, other):
            np.testing.assert_array_equal(a, b)
        second, _ = encode(other, empty(36, 4), jnp.int32(288), topology)
        model = Network(
            Config(width=16, depth=2, heads=4, history=4), key=jax.random.PRNGKey(2)
        )
        np.testing.assert_array_equal(
            model(x, topology).logits, model(second, topology).logits
        )

    def test_history_and_action_mask(self):
        _, topology, obs, x, memory = fixture()
        self.assertEqual(int(x.legal.sum()), 5)  # two directions, two amounts, pass
        hidden = obs._replace(
            cells=obs.cells.at[0, :].set(jnp.zeros(10)).at[0, 7].set(1),
            stats=obs.stats.at[0].set(3 / 512),
        )
        y, remembered = encode(hidden, memory, jnp.int32(2), topology)
        self.assertEqual(float(remembered.last[0, 0]), 1.0)
        self.assertEqual(int(remembered.seen_at[0]), 0)
        self.assertGreater(float(y.cells[0, 18]), 0.0)
        self.assertEqual(int(y.time_valid.sum()), 2)
        reset, _ = encode(hidden, empty(16, 4), jnp.int32(128), topology)
        self.assertEqual(float(reset.cells[0, 10]), 0.0)

    def test_permutation_equivariance(self):
        _, topology, _, x, _ = fixture()
        model = Network(
            Config(width=16, depth=2, heads=4, history=4), key=jax.random.PRNGKey(3)
        )
        permutation = jnp.array(np.random.default_rng(5).permutation(16))
        inverse = jnp.argsort(permutation)
        altered_graph = topology._replace(
            neighbors=inverse[topology.neighbors[permutation]],
            valid=topology.valid[permutation],
            positions=topology.positions[permutation],
        )
        altered_x = x._replace(
            cells=x.cells[permutation],
            recent=x.recent[:, permutation],
            board=x.board[permutation],
            movement=x.movement[permutation],
            legal=jnp.concatenate(
                (x.legal[:-1].reshape(16, 8)[permutation].reshape(-1), x.legal[-1:])
            ),
        )
        original, altered = model(x, topology), model(altered_x, altered_graph)
        np.testing.assert_allclose(
            original.logits[:-1].reshape(16, 8)[permutation],
            altered.logits[:-1].reshape(16, 8),
            atol=2e-5,
            rtol=2e-5,
        )
        np.testing.assert_allclose(original.value, altered.value, atol=2e-5)

    def test_official_replay_and_versions(self):
        replay = {
            "version": 5,
            "id": "unit",
            "mapWidth": 4,
            "mapHeight": 4,
            "usernames": ["a", "b"],
            "stars": [50, 50],
            "cities": [5],
            "cityArmies": [20],
            "generals": [0, 15],
            "mountains": [],
            "moves": [],
            "afks": [{"index": 1, "turn": 3}],
        }
        simulator = Simulator()
        try:
            old = simulator.replay(replay, 100)
            self.assertEqual(old["frames"][2]["armies"][5], 21)
            self.assertEqual(old["frames"][0]["armies"][0], 1)
            replay["version"] = 13
            newer = simulator.replay(replay, 100)
            self.assertEqual(newer["frames"][2]["armies"][5], 20)
            self.assertEqual(newer["winner"], 0)
            split = split_for(replay)
            replay["generals"].reverse()
            self.assertEqual(split, split_for(replay))
        finally:
            simulator.close()

    def test_supervised_gradient_and_checkpoint(self):
        _, topology, _, x, _ = fixture(history=2)
        config = Config(width=16, depth=2, heads=4, history=2)
        model = Network(config, key=jax.random.PRNGKey(9))
        x = jax.tree.map(lambda a: jnp.stack((a, a)), x)
        target = {
            "action": jnp.array([2, 2]),
            "weight": jnp.ones(2),
            "general": jnp.array([15, 15]),
            "owner": jnp.zeros((2, 16), jnp.int32),
            "army": jnp.zeros((2, 16)),
        }
        initial = float(validate(model, x, target, topology)[0]) / 2
        tx = optimizer(0.002)
        state = tx.init(eqx.filter(model, eqx.is_inexact_array))
        update = make_update(tx, 0.02)
        for _ in range(32):
            model, state, loss, _, gradient = update(model, state, x, target, topology)
        self.assertTrue(bool(jnp.isfinite(loss) & jnp.isfinite(gradient)))
        final = float(validate(model, x, target, topology)[0]) / 2
        self.assertLess(final, initial * 0.3)
        print(
            json.dumps(
                {"check": "tiny_batch_fit", "initial_nll": initial, "final_nll": final}
            ),
            flush=True,
        )
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / "model.zip"
            save(path, model, {"test": True})
            restored, _ = load(path)
            for a, b in zip(jax.tree.leaves(model), jax.tree.leaves(restored)):
                np.testing.assert_array_equal(a, b)

    def test_gae_boundaries(self):
        r, v, n = (
            jnp.array([[1.0], [0.0]]),
            jnp.array([[0.2], [0.3]]),
            jnp.array([[99.0], [0.7]]),
        )
        terminal, truncated = jnp.array([[True], [False]]), jnp.array([[False], [True]])
        result = advantages(r, v, n, terminal, truncated, 1.0, 0.95)
        np.testing.assert_allclose(result[:, 0], jnp.array([0.8, 0.4]), rtol=1e-6)

    def test_supervised_resume_matches_uninterrupted(self):
        _, topology, _, x, _ = fixture(history=2)
        model = Network(
            Config(width=16, depth=2, heads=4, history=2), key=jax.random.PRNGKey(7)
        )
        x = jax.tree.map(lambda a: jnp.stack((a, a)), x)
        targets = {
            "action": jnp.array([2, 6]),
            "weight": jnp.ones(2),
            "general": jnp.array([15, 15]),
            "owner": jnp.zeros((2, 16), jnp.int32),
            "army": jnp.zeros((2, 16)),
        }
        tx = optimizer(1e-4)
        state = tx.init(eqx.filter(model, eqx.is_inexact_array))
        update = make_update(tx, 0.02)

        def examples(*_):
            for _ in range(3):
                yield x, dict(targets), topology

        files = [Path("a.npz"), Path("b.npz")]
        with patch("train.batches", examples):
            reference, expected_state, expected_metrics, _, complete = epoch(
                model, files, 2, 2, 0, update, state, seed=42
            )
            self.assertTrue(complete)
            paused, paused_state, _, cursor, complete = epoch(
                model, files, 2, 2, 0, update, state, seed=42, max_batches=2
            )
            self.assertFalse(complete)
            with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
                path = Path(folder) / "checkpoint.zip"
                save(path, paused, {"cursor": cursor}, paused_state)
                restored, metadata = load(path)
                restored_state = restore_optimizer(path, state)
                resumed, actual_state, metrics, _, complete = epoch(
                    restored,
                    files,
                    2,
                    2,
                    0,
                    update,
                    restored_state,
                    seed=42,
                    progress=metadata["cursor"],
                )
            self.assertTrue(complete)
            self.assertEqual(expected_metrics, metrics)
            for expected, actual in zip(
                jax.tree.leaves((reference, expected_state)),
                jax.tree.leaves((resumed, actual_state)),
            ):
                np.testing.assert_array_equal(expected, actual)

    def test_real_rollout_and_ppo_update(self):
        config = Config(width=16, depth=2, heads=4, history=2)
        model = Network(config, key=jax.random.PRNGKey(2))
        env = Environment(
            Run(side=4, distance=3, environments=2, pool=2, steps=5, horizon=4), 2, 12
        )
        carry, buffer, metrics = env.collect(model, env.carry)
        data = jax.device_put(buffer.read(np.arange(buffer.count))).unpack()
        self.assertEqual(len(data.actions), 20)
        self.assertGreater(int(metrics[1]), 0)
        self.assertEqual(
            int(carry.inputs.time_valid.sum()), 8
        )  # reset at step4, then one observation
        self.assertTrue(bool(jnp.any(data.inputs.legal.sum(-1) > 1)))
        self.assertTrue(bool(jnp.all(data.inputs.legal[jnp.arange(20), data.actions])))
        tx = optimizer(1e-4)
        state = tx.init(eqx.filter(model, eqx.is_inexact_array))
        updated, _, loss, _, gradient = ppo_update(tx, 0.01)(
            model, state, data, env.topology
        )
        self.assertTrue(bool(jnp.isfinite(loss) & jnp.isfinite(gradient)))
        self.assertTrue(
            any(
                not np.array_equal(a, b)
                for a, b in zip(jax.tree.leaves(model), jax.tree.leaves(updated))
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
