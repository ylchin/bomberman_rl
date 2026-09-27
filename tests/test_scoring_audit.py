from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from agent_code.our_agent.features import ACTIONS, state_to_features
from agent_code.our_agent.q_linear import LinearQ
from experiments.audit_task4_scoring import (
    build_summary, inspect_decision, main, rebuild_summary, summarize, wait_streaks,
    write_summary,
)


def board():
    field = np.zeros((9, 9), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    return dict(field=field, coins=[(5, 3)], bombs=[], others=[],
                explosion_map=np.zeros_like(field), self=("me", 0, True, (3, 3)),
                step=1, round=1)


def waiting_policy(state, tie=False):
    model = LinearQ()
    phi = state_to_features(state)
    feature = np.flatnonzero(phi)[0]
    model.W[ACTIONS.index("WAIT"), feature] = 3 / phi[feature]
    if tie:
        model.W[ACTIONS.index("RIGHT"), feature] = 3 / phi[feature]
    return model


class ScoringAuditTests(unittest.TestCase):
    def test_wait_instead_of_reachable_coin_has_ranking_explanation(self):
        s = board()
        row, example = inspect_decision(s, "WAIT", waiting_policy(s))
        self.assertEqual(row["category"], "safe_coin_opportunity")
        self.assertEqual(row["coin_actions"], ["RIGHT"])
        self.assertEqual(row["coin_distance"], 2)
        self.assertTrue(row["nonprogress_coin_choice"])
        self.assertEqual(row["chosen_minus_best_coin_q"], 3)
        self.assertAlmostEqual(sum(e["chosen_minus_coin_contribution"]
                                   for e in example["largest_ranking_contributions"]), 3)

    def test_tied_wait_does_not_imply_strict_preference_against_coin(self):
        s = board()
        row, _ = inspect_decision(s, "WAIT", waiting_policy(s, tie=True))
        self.assertEqual(row["greedy_coin_probability"], 0.5)
        self.assertEqual(row["chosen_minus_best_coin_q"], 0)
        self.assertTrue(row["nonprogress_coin_choice"])

    def test_progress_action_is_not_a_missed_opportunity(self):
        s = board()
        row, example = inspect_decision(s, "RIGHT", waiting_policy(s))
        self.assertFalse(row["nonprogress_coin_choice"])
        self.assertIsNone(example)

    def test_danger_no_coins_and_blocked_routes_are_separate(self):
        s = board()
        s["bombs"] = [((3, 3), 3)]
        row, _ = inspect_decision(s, "WAIT", LinearQ())
        self.assertEqual(row["category"], "in_known_danger")
        s = board()
        s["coins"] = []
        row, _ = inspect_decision(s, "WAIT", LinearQ())
        self.assertEqual(row["category"], "safe_no_visible_coins")
        s = board()
        s["field"][:] = -1
        s["field"][3, 3] = s["field"][5, 3] = 0
        row, _ = inspect_decision(s, "WAIT", LinearQ())
        self.assertEqual(row["category"], "safe_no_clear_coin_route")

    def rows(self, steps, seed=1, coins=True):
        s = board()
        if not coins:
            s["coins"] = []
        row, _ = inspect_decision(s, "WAIT", LinearQ())
        return [{**row, "seed": seed, "step": step} for step in steps]

    def test_wait_streaks_respect_gaps_round_boundaries_and_actions(self):
        rows = self.rows([1, 2, 3, 5, 6]) + self.rows([7, 8, 9], seed=2)
        rows += [{**self.rows([10], seed=2)[0], "action": "RIGHT"}]
        rows += self.rows([11, 12], seed=2)
        streaks = wait_streaks(rows)
        self.assertEqual([(s["seed"], s["start_step"], s["length"]) for s in streaks],
                         [(1, 1, 3), (2, 7, 3)])

    def test_danger_breaks_wait_streak(self):
        rows = self.rows([1, 2, 3, 4, 5])
        rows[2]["outside_known_danger"] = False
        self.assertEqual(wait_streaks(rows), [])

    def test_no_coin_waits_do_not_count_as_missed_coin_opportunities(self):
        result = summarize(self.rows([1, 2, 3, 4], coins=False))
        self.assertEqual(result["safe_wait_streaks"], 1)
        self.assertEqual(result["safe_wait_streaks_with_coin_opportunities"], 0)
        self.assertEqual(result["actual_nonprogress_coin_choices"], 0)
        self.assertIsNone(result["actual_coin_progress_fraction"])

    def game(self):
        return dict(seed=1, score=0, coins=0, kills=0, self_kill=0, win=0, tie=1, loss=0)

    def test_summary_file_and_stdout_accept_numpy_positions(self):
        rows = self.rows([1, 2, 3], coins=False)
        for row in rows:
            row["position"] = [np.int64(3), np.int64(3)]
        summary = build_summary(rows, [self.game()], 3)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as stdout:
            output = Path(tmp)
            write_summary(output, summary)
            saved = json.loads((output / "summary.json").read_text())
            printed = json.loads(stdout.getvalue())
        self.assertEqual(saved["overall"]["longest_streak_examples"][0]["position"], [3, 3])
        self.assertEqual(printed["overall"], saved["overall"])

    def saved_run(self, output):
        files = {
            "protocol.json": json.dumps(dict(seed_start=1, rounds=1, wait_streak_minimum=3)),
            "games.json": json.dumps([self.game()]),
            "decisions.jsonl": "".join(json.dumps(row) + "\n" for row in self.rows([1, 2, 3], coins=False)),
            "examples.json": "[]\n",
            "states.jsonl": "preserved original snapshots\n",
        }
        for name, content in files.items():
            (output / name).write_text(content)
        return files

    def test_recovery_cli_uses_saved_data_without_loading_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            output = Path(tmp)
            originals = self.saved_run(output)
            with patch("sys.argv", ["audit_task4_scoring.py", "--summarize-only", tmp]), \
                    patch.object(LinearQ, "load", side_effect=AssertionError("must not load a model")):
                main()
            saved = json.loads((output / "summary.json").read_text())
            self.assertEqual(saved["rounds"], 1)
            self.assertEqual(saved["overall"]["decisions"], 3)
            for name, content in originals.items():
                self.assertEqual((output / name).read_text(), content)

    def test_recovery_rejects_missing_game_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            self.saved_run(output)
            (output / "games.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "all protocol rounds"):
                rebuild_summary(output)
            self.assertFalse((output / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
