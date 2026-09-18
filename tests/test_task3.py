import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from agent_code.our_agent.features import ACTIONS, FEATURE_DIM
from agent_code.our_agent.q_linear import LinearQ
from agent_code.our_agent import callbacks
from evaluate import parse_result_file


class Task3Tests(unittest.TestCase):
    def test_init_checkpoint_setup_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "task2.pkl"
            source.write_bytes(pickle.dumps(dict(W=np.ones((6, 32)), feature_dim=32,
                                                 n_actions=6, actions=ACTIONS)))
            cfg = dict(init_checkpoint=str(source), resume=False, seed=1,
                       weights_out=str(root / "task3.pkl"),
                       best_weights_out=str(root / "task3_best.pkl"),
                       log_csv=str(root / "training.csv"))
            agent = SimpleNamespace(train=True, logger=Mock())
            with patch.dict(callbacks.config.TRAIN, cfg):
                callbacks.setup(agent)
                self.assertEqual(agent.model.W.shape, (6, FEATURE_DIM))
                agent.model.save(cfg["weights_out"])
                with self.assertRaises(FileExistsError):
                    callbacks.setup(agent)

    def test_padding_preserves_predictions_and_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.pkl"
            weights = np.arange(6 * 32, dtype=float).reshape(6, 32)
            path.write_bytes(pickle.dumps(dict(W=weights, feature_dim=32,
                                              n_actions=6, actions=ACTIONS)))
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                LinearQ.load(path)
            model = LinearQ.load(path, allow_legacy_padding=True)
            phi = np.arange(FEATURE_DIM, dtype=float)
            np.testing.assert_array_equal(model.q_values(phi), weights @ phi[:32])
            np.testing.assert_array_equal(model.W[:, 32:], 0)
            self.assertEqual(path.read_bytes(), original)
            model.save(Path(folder) / "new.pkl")
            np.testing.assert_array_equal(LinearQ.load(Path(folder) / "new.pkl").W, model.W)

    def test_incompatible_checkpoints_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.pkl"
            for dim, actions, weights in (
                (31, ACTIONS, np.zeros((6, 31))),
                (34, list(reversed(ACTIONS)), np.zeros((6, 34))),
                (34, ACTIONS, np.zeros((5, 34))),
                (34, ACTIONS, np.full((6, 34), np.nan)),
            ):
                path.write_bytes(pickle.dumps(dict(W=weights, feature_dim=dim,
                                                  n_actions=6, actions=actions)))
                with self.assertRaises(ValueError):
                    LinearQ.load(path, allow_legacy_padding=True)

    def test_competitive_outcomes_and_solo(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stats.json"
            for score, expected in [(7, (1, 0, 0)), (5, (0, 1, 0)), (0, (0, 0, 1))]:
                path.write_text(json.dumps({"by_agent": {
                    "our_agent_0": {"score": score},
                    "opponent_0": {"score": 5}, "opponent_1": {"score": 2}}}))
                row = parse_result_file(path, "our_agent", 22000, 0, 3)
                self.assertEqual(tuple(row[k] for k in ("win", "tie", "loss")), expected)
                self.assertEqual(len(json.loads(row["opponent_scores"])), 2)
            path.write_text(json.dumps({"by_agent": {"our_agent": {}}}))
            self.assertEqual(parse_result_file(path, "our_agent", 1, 0, 2)["win"], "")


if __name__ == "__main__":
    unittest.main()
