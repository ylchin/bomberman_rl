import csv
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from agent_code.our_agent import callbacks, config
from agent_code.our_agent.q_linear import LinearQ
from experiments.compare_task4_guard import arm_order, configure_arm, guard_settings
from experiments.analyze_task4_guard import summarize_comparison


class CheckpointComparisonTests(unittest.TestCase):
    def test_model_order_alternates_and_single_arm_is_unchanged(self):
        self.assertEqual(arm_order(0), (False, True))
        self.assertEqual(arm_order(1), (True, False))
        self.assertEqual(arm_order(2), (False, True))
        self.assertEqual(arm_order(0, "on"), (True,))
        self.assertEqual(arm_order(1, "off"), (False,))

    def test_both_checkpoint_arms_have_identical_policy_settings(self):
        expected = dict(bomb_collision_guard=True, escape_collision_guard=True,
                        optimistic_fallback=False, coin_preference=False)
        self.assertEqual(guard_settings("checkpoint", False), expected)
        self.assertEqual(guard_settings("checkpoint", True), expected)

    def test_fresh_agents_load_alternating_checkpoints_despite_stale_config(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ), \
                patch.object(config, "MODEL", "net"), patch.object(config, "EVAL_WEIGHTS", ["stale.pkl"]), \
                patch.dict(config.TRAIN, survival_filter=False, bomb_collision_guard=False,
                           escape_collision_guard=False, optimistic_fallback=True, coin_preference=True):
            paths = {False: Path(tmp) / "baseline.pkl", True: Path(tmp) / "candidate.pkl"}
            for arm, path in paths.items():
                model = LinearQ()
                model.W[:] = 2 if arm else 1
                model.save(path)
            for index in range(2):
                for arm in arm_order(index):
                    configure_arm(config, "checkpoint", arm, 35000 + index, paths)
                    agent = SimpleNamespace(train=False, logger=Mock())
                    callbacks.setup(agent)
                    np.testing.assert_array_equal(agent.model.W, np.full_like(agent.model.W, 2 if arm else 1))
                    self.assertEqual(config.TRAIN["seed"], 35000 + index)
                    self.assertTrue(config.TRAIN["survival_filter"])
                    self.assertEqual(config.EVAL_WEIGHTS, [str(paths[arm])])
                    self.assertEqual(os.environ["AGENT_EVAL_CHECKPOINT"], str(paths[arm]))
                    for key, value in guard_settings("checkpoint", arm).items():
                        self.assertEqual(config.TRAIN[key], value)

    def fixture(self, folder, kind="checkpoint", seeds=(35000, 35001)):
        for arm in ("off", "on"):
            candidate = arm == "on"
            rows = [dict(seed=seed, score=3 if candidate else 1, coins=3 if candidate else 1,
                         kills=0, self_kill=0, win=int(candidate), tie=0, loss=int(not candidate),
                         score_margin=1 if candidate else -1, survived=1, died=0)
                    for seed in seeds]
            with (folder / f"guard_{arm}.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (folder / "self_kills.jsonl").write_text("")
        return dict(rounds_per_arm=2, seed_start=35000, guard_kind=kind,
                    comparison_kind="checkpoint" if kind == "checkpoint" else "flag",
                    checkpoints={"baseline": "baseline.pkl", "candidate": "candidate.pkl"},
                    checkpoint_hashes={"baseline": "baseline-hash", "candidate": "candidate-hash"},
                    arm_settings={arm: guard_settings(kind, arm == "on") for arm in ("off", "on")})

    def test_summary_labels_delta_direction_and_primary_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            summary = summarize_comparison(folder, self.fixture(folder))
            self.assertEqual(summary["primary_metric"], "win")
            self.assertEqual(next(iter(summary["metrics"])), "win")
            win = summary["metrics"]["win"]
            self.assertEqual(win["baseline"], 0)
            self.assertEqual(win["candidate"], 1)
            self.assertEqual(win["candidate_minus_baseline"], 1)
            self.assertEqual(win["paired_bootstrap_95_interval"], [1, 1])
            self.assertEqual(summary["candidate_self_kill_count"], 0)
            self.assertEqual(summary["baseline_self_kill_count"], 0)
            self.assertNotIn("guard_on_deaths", summary)
            self.assertEqual(json.loads((folder / "summary.json").read_text()), summary)

    def test_legacy_flag_comparison_keeps_off_on_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            protocol = self.fixture(folder, kind="escape")
            protocol.pop("comparison_kind")  # Old protocol files lack this key.
            summary = summarize_comparison(folder, protocol)
            self.assertEqual(summary["metrics"]["score"]["on_minus_off"], 2)
            self.assertEqual(summary["guard_on_deaths"], 0)

    def test_rejects_duplicated_world_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            protocol = self.fixture(folder, seeds=(35000, 35000))
            with self.assertRaisesRegex(ValueError, "duplicated"):
                summarize_comparison(folder, protocol)

    def test_rejects_incomplete_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            protocol = self.fixture(folder, seeds=(35000,))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                summarize_comparison(folder, protocol)


if __name__ == "__main__":
    unittest.main()
