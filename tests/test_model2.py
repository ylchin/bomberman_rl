import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

import events as e
from agent_code.our_agent import train
from agent_code.our_agent.rewards import potential, reward_from_events
from agent_code.our_agent.features import state_to_channels
from agent_code.our_agent.model_net import (
    NetQ,
    INPUT_DIM,
    encode_state,
    transform_input,
)
from agent_code.our_agent.symmetry import apply_to_state, sym_transforms


def board():
    field = np.zeros((17, 17), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    field[5, 6] = 1
    return dict(
        field=field,
        coins=[(4, 3)],
        bombs=[((7, 8), 3)],
        explosion_map=np.zeros_like(field),
        others=[("opponent", 0, True, (9, 2))],
        self=("agent", 0, True, (2, 3)),
        step=1,
        round=1,
    )


class Model2Tests(unittest.TestCase):
    def test_channel_contract(self):
        state = board()
        channels = state_to_channels(state)
        self.assertEqual(channels.shape, (7, 17, 17))
        self.assertEqual(channels.dtype, np.float32)
        self.assertTrue(np.isfinite(channels).all())
        self.assertTrue(((channels >= 0) & (channels <= 1)).all())
        self.assertEqual(channels[3, 2, 3], 1)
        self.assertGreater(channels[5, 7, 8], 0)
        self.assertEqual(channels[5, 1, 1], 0)
        self.assertIsNone(encode_state(None))

    def test_spatial_symmetry_matches_raw_state(self):
        state = board()
        for op in sym_transforms():
            np.testing.assert_array_equal(
                transform_input(encode_state(state), op),
                state_to_channels(apply_to_state(state, op)).reshape(-1),
            )

    def test_double_dqn_and_terminal_mask(self):
        model = NetQ(seed=3)
        with torch.no_grad():
            for network in (model.online, model.target):
                for parameter in network.parameters():
                    parameter.zero_()
            model.online.layers[-1].bias.copy_(
                torch.tensor([0.0, 3.0, 0.0, 0.0, 0.0, 0.0])
            )
            model.target.layers[-1].bias.copy_(
                torch.tensor([10.0, 2.0, 0.0, 0.0, 0.0, 0.0])
            )
        inputs = np.zeros((2, INPUT_DIM), dtype=np.float32)
        np.testing.assert_array_equal(model.bootstrap_value(inputs, [0, 1]), [2, 0])
        np.testing.assert_array_equal(
            model.bootstrap_value(inputs, [0, 1], [0, 0], rule="sarsa"), [10, 0]
        )

    def test_learning_target_sync_and_checkpoint(self):
        model = NetQ(seed=1, target_update_every=2)
        phi = encode_state(board())[None, :]
        before = model.q_values(phi).copy()
        frozen = {
            key: value.clone() for key, value in model.target.state_dict().items()
        }
        model.update(phi, [0], [2.0], 1e-3)
        self.assertFalse(np.array_equal(before, model.q_values(phi)))
        for key, value in model.target.state_dict().items():
            torch.testing.assert_close(value, frozen[key])
        model.update(phi, [0], [2.0], 1e-3)
        for key, value in model.target.state_dict().items():
            torch.testing.assert_close(value, model.online.state_dict()[key])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            original_replace = Path.replace
            attempts = []

            def temporarily_locked(source, target):
                attempts.append(1)
                if len(attempts) == 1:
                    raise PermissionError("Checkpoint temporarily locked")
                return original_replace(source, target)

            with (
                patch.object(Path, "replace", temporarily_locked),
                patch("agent_code.our_agent.model_net.time.sleep"),
            ):
                model.save(path)
            self.assertEqual(len(attempts), 2)
            loaded = NetQ.load(path, training=True)
            self.assertEqual(loaded.updates, 2)
            np.testing.assert_array_equal(model.q_values(phi), loaded.q_values(phi))
            model.update(phi, [1], [-1.0], 1e-3)
            loaded.update(phi, [1], [-1.0], 1e-3)
            np.testing.assert_array_equal(model.q_values(phi), loaded.q_values(phi))
            data = torch.load(path, weights_only=True)
            data["actions"] = list(reversed(data["actions"]))
            torch.save(data, path)
            with self.assertRaises(ValueError):
                NetQ.load(path)

    def test_n_step_returns_and_terminal_tail(self):
        agent = SimpleNamespace(
            feature_dim=INPUT_DIM,
            episode=0,
            t=dict(gamma=0.5, n_step=2, use_symmetry=False),
            buffer=Mock(),
            _sym=None,
        )
        phi = np.zeros(INPUT_DIM, dtype=np.float32)
        agent.traj = [(phi, 0, 1.0), (phi, 1, 2.0), (phi, 2, 4.0)]
        train._flush_episode_to_buffer(agent)
        records = [call.args for call in agent.buffer.push.call_args_list]
        self.assertEqual([record[2] for record in records], [2.0, 4.0, 4.0])
        self.assertEqual([record[5] for record in records], [0.0, 1.0, 1.0])

    def test_survivor_final_step_not_duplicated(self):
        phi = np.zeros(INPUT_DIM, dtype=np.float32)
        successor = board()
        gamma = 0.9
        expected_reward = (
            1.0 + reward_from_events([e.SURVIVED_ROUND]) - gamma * potential(successor)
        )
        agent = SimpleNamespace(
            traj=[(phi, 0, 1.0)],
            ep_reward=1.0,
            ep_counts=train.Counter({e.COIN_COLLECTED: 1}),
            episode=0,
            t=dict(learn_iters_end=0, gamma=gamma),
            _last_post_state=successor,
            model=Mock(),
            _weights_path="unused",
        )
        with (
            patch.object(train, "_flush_episode_to_buffer") as flush,
            patch.object(train, "_learn"),
            patch.object(train, "_save_periodic_candidate"),
            patch.object(train, "_update_best_checkpoint"),
            patch.object(train, "_write_csv_row"),
            patch.object(train, "_epsilon", return_value=0.0),
            patch.object(train, "_alpha", return_value=0.0),
        ):
            def check_final_transition(obj):
                self.assertEqual(len(obj.traj), 1)
                self.assertAlmostEqual(obj.traj[0][2], expected_reward)
                self.assertAlmostEqual(obj.ep_reward, expected_reward)
                self.assertEqual(obj.ep_counts[e.COIN_COLLECTED], 1)
                self.assertEqual(obj.ep_counts[e.SURVIVED_ROUND], 1)

            flush.side_effect = check_final_transition
            train.end_of_round(agent, board(), "UP", [e.COIN_COLLECTED, e.SURVIVED_ROUND])
            self.assertIsNone(agent._last_post_state)


if __name__ == "__main__":
    unittest.main()
