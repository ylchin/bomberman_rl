"""Verify no-environment submission defaults and the exact frozen artifact."""

import hashlib
import os
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

from agent_code.our_agent.q_linear import LinearQ

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agent_code/our_agent"
CHECKPOINT = "weights/q_linear_safety_v2_symavg.pkl"
EXPECTED_SHA256 = "c5671d5e806e8d563fd1b21832eaee8d01a54e86ba592a4a5898273c553d973f"


class Task4SubmissionTests(unittest.TestCase):
    def test_clean_environment_selects_frozen_model_and_validated_guards(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = runpy.run_path(str(AGENT / "config.py"))
        self.assertEqual(cfg["MODEL"], "linear")
        self.assertEqual(cfg["EVAL_WEIGHTS"], [CHECKPOINT])
        for key in ("survival_filter", "bomb_collision_guard", "escape_collision_guard"):
            self.assertTrue(cfg["TRAIN"][key], key)
        for key in ("optimistic_fallback", "coin_preference"):
            self.assertFalse(cfg["TRAIN"][key], key)

    def test_submission_checkpoint_is_present_unchanged_and_loadable(self):
        path = AGENT / CHECKPOINT
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), EXPECTED_SHA256)
        model = LinearQ.load(path)
        self.assertEqual(model.W.shape, (6, 37))


if __name__ == "__main__":
    unittest.main()
