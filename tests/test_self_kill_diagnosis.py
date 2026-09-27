from copy import deepcopy
import unittest

import numpy as np

from experiments.analyze_task4_guard import diagnose


def state():
    field = np.zeros((9, 9), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    return dict(field=field, self=("me", 0, True, (3, 3)), others=[], bombs=[],
                coins=[], explosion_map=np.zeros_like(field), step=1, round=1)


def trace(before, action, mask, after=None, events=()):
    return dict(seed=1, guard=True, step=1, final_state=after,
                history=[dict(state=before, action=action, mask=mask, events=list(events))])


class SelfKillDiagnosisTests(unittest.TestCase):
    def test_empty_screen_records_fallback_without_bomb(self):
        result = diagnose(trace(state(), "WAIT", [False] * 6))
        self.assertEqual(result["fallback_steps"], [1])
        self.assertEqual(result["bomb_requests_during_fallback"], [])
        self.assertTrue(result["final_mask_empty"])

    def test_only_illegal_mask_entries_also_trigger_fallback(self):
        before = state()
        before["field"][3, 2] = -1
        result = diagnose(trace(before, "WAIT", [True, False, False, False, False, False]))
        self.assertEqual(result["fallback_steps"], [1])
        self.assertEqual(result["actions_outside_nonempty_screen"], [])

    def test_fatal_turn_interception_is_recorded(self):
        before = state()
        after = deepcopy(before)
        after["others"] = [("opp", 0, True, (4, 3))]
        result = diagnose(trace(before, "RIGHT", [False, True, False, False, False, False], after))
        self.assertEqual(result["observed_interceptions"], [1])
        self.assertEqual(result["fallback_steps"], [])

    def test_successful_bomb_is_distinguished_from_failed_request(self):
        before = state()
        after = deepcopy(before)
        after["bombs"] = [((3, 3), 3)]
        dropped = diagnose(trace(before, "BOMB", [False] * 6, after, ["BOMB_DROPPED"]))
        self.assertEqual(dropped["confirmed_bomb_placements_during_fallback"], [1])
        self.assertEqual(dropped["own_bomb_placement_step"], 1)
        failed = diagnose(trace(before, "BOMB", [False] * 6, after, ["INVALID_ACTION"]))
        self.assertEqual(failed["bomb_requests_during_fallback"], [1])
        self.assertEqual(failed["confirmed_bomb_placements_during_fallback"], [])
        self.assertIsNone(failed["own_bomb_placement_step"])

    def test_action_outside_available_screen_is_reported(self):
        result = diagnose(trace(state(), "WAIT", [False, True, False, False, False, False]))
        self.assertEqual(result["actions_outside_nonempty_screen"], [1])
        self.assertEqual(result["fallback_steps"], [])

    def test_optimistic_screen_is_distinct_from_empty_base_screen(self):
        death = trace(state(), "WAIT", [False] * 6)
        death["history"][0]["effective_mask"] = [False, True, False, False, False, False]
        result = diagnose(death)
        self.assertEqual(result["fallback_steps"], [1])
        self.assertEqual(result["optimistic_fallback_steps"], [1])
        self.assertEqual(result["actions_outside_nonempty_screen"], [])
        self.assertEqual(result["actions_outside_effective_screen"], [1])


if __name__ == "__main__":
    unittest.main()
