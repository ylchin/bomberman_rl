from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from agent_code.our_agent.q_linear import LinearQ
from experiments.analyze_task4_nearby_coins import inspect_nearby, main, nearby_summary
from experiments.audit_task4_scoring import encode


def board():
    field = np.zeros((9, 9), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    return dict(field=field, coins=[(5, 3)], bombs=[], others=[],
                explosion_map=np.zeros_like(field), self=("me", 0, True, (3, 3)),
                step=1, round=1)


class NearbyCoinAuditTests(unittest.TestCase):
    def test_distance_limit_is_inclusive(self):
        reason, row, _ = inspect_nearby(board(), "WAIT", LinearQ(), max_distance=2)
        self.assertEqual(reason, "included")
        self.assertEqual(row["coin_distance"], 2)
        self.assertTrue(row["feature_matches_clear_route"])
        self.assertEqual(inspect_nearby(board(), "WAIT", LinearQ(), max_distance=1)[0],
                         "coin_beyond_distance_limit")

    def test_distant_bomb_and_fire_exclude_the_entire_state(self):
        s = board()
        s["bombs"] = [((1, 7), 3)]
        self.assertEqual(inspect_nearby(s, "WAIT", LinearQ())[0], "bomb_present")
        s["bombs"] = []
        s["explosion_map"][1, 7] = 1
        self.assertEqual(inspect_nearby(s, "WAIT", LinearQ())[0], "active_fire_present")

    def test_opponent_detour_uses_walk_distance_and_reports_feature_mismatch(self):
        s = board()
        s["others"] = [("opp", 0, True, (4, 3))]
        self.assertEqual(inspect_nearby(s, "WAIT", LinearQ(), max_distance=3)[0],
                         "coin_beyond_distance_limit")
        reason, row, _ = inspect_nearby(s, "WAIT", LinearQ(), max_distance=4)
        self.assertEqual(reason, "included")
        self.assertEqual(row["coin_distance"], 4)
        self.assertEqual(row["coin_feature_directions"], ["RIGHT"])
        self.assertFalse(row["feature_matches_clear_route"])
        self.assertNotIn("RIGHT", row["coin_actions"])

    def test_any_equally_short_direction_counts_as_feature_agreement(self):
        s = board()
        s["coins"] = [(4, 4)]
        _, row, _ = inspect_nearby(s, "RIGHT", LinearQ())
        self.assertEqual(set(row["coin_actions"]), {"RIGHT", "DOWN"})
        self.assertTrue(row["feature_matches_clear_route"])
        self.assertFalse(row["nonprogress_coin_choice"])

    def test_wait_tie_is_not_a_strict_preference(self):
        _, row, _ = inspect_nearby(board(), "WAIT", LinearQ())
        self.assertFalse(row["wait_strictly_above_all_coin_moves"])
        self.assertEqual(row["chosen_minus_best_coin_q"], 0)

    def test_excluded_steps_break_wait_streaks(self):
        rows = []
        for step in range(1, 7):
            s = board()
            s["step"] = step
            if step == 3:
                s["bombs"] = [((1, 7), 3)]
            _, row, _ = inspect_nearby(s, "WAIT", LinearQ())
            if row is not None:
                rows.append({**row, "seed": 1})
        result = nearby_summary(rows)
        self.assertEqual(result["safe_wait_streaks"], 1)
        self.assertEqual(result["longest_streak_examples"][0]["start_step"], 4)
        self.assertEqual(result["longest_safe_wait_streak"], 3)

    def test_no_eligible_rows_produces_serializable_empty_report(self):
        result = nearby_summary([])
        self.assertEqual(result["coin_opportunities"], 0)
        self.assertIsNone(result["actual_coin_progress_fraction"])
        json.dumps(result, default=encode)

    def test_offline_cli_preserves_saved_inputs_and_serializes_output(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as stdout:
            source = Path(tmp)
            checkpoint = source / "checkpoint.pkl"
            LinearQ().save(checkpoint)
            protocol = {"checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        "seed_start": 7, "rounds": 1}
            (source / "protocol.json").write_text(json.dumps(protocol))
            s = board()
            s["self"] = ("me", 0, True, (np.int64(3), np.int64(3)))
            (source / "states.jsonl").write_text(json.dumps({"seed": 7, "state": s, "action": "WAIT"}, default=encode) + "\n")
            originals = {p.name: p.read_bytes() for p in source.iterdir()}
            with patch("sys.argv", ["analyze_task4_nearby_coins.py", tmp,
                                    "--out-dir", str(source / "nearby")]):
                main()
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["source_decisions"], 1)
            self.assertEqual(result["overall"]["coin_opportunities"], 1)
            self.assertEqual(json.loads((source / "nearby/summary.json").read_text()), result)
            for name, data in originals.items():
                self.assertEqual((source / name).read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
