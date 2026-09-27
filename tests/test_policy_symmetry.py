import unittest

import numpy as np

from agent_code.our_agent.features import ACTIONS, FEATURE_NAMES
from agent_code.our_agent.q_linear import LinearQ
from agent_code.our_agent.symmetry import apply_to_action, sym_transforms
from experiments.check_policy_symmetry import inspect_state


def state():
    field = np.zeros((9, 9), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    return dict(field=field, self=("me", 0, True, (3, 3)), others=[],
                coins=[(5, 3)], bombs=[], explosion_map=np.zeros_like(field), step=1, round=1)


class PolicySymmetryTests(unittest.TestCase):
    def test_action_mapping_round_trip_including_wait_and_bomb(self):
        for op in sym_transforms():
            permutation = np.asarray([apply_to_action(a, op) for a in range(len(ACTIONS))])
            self.assertEqual(sorted(permutation.tolist()), list(range(len(ACTIONS))))
            original = np.arange(len(ACTIONS)) + 10
            transformed = np.empty_like(original)
            transformed[permutation] = original
            np.testing.assert_array_equal(transformed[permutation], original)
            self.assertEqual(permutation[4:].tolist(), [4, 5])

    def test_equivariant_coin_weights_pass_all_transforms(self):
        model = LinearQ()
        for i, direction in enumerate(ACTIONS[:4]):
            model.W[i, FEATURE_NAMES.index("COIN_DIR_" + direction)] = 1.0
        report = inspect_state(state(), {"model": model})
        self.assertEqual(len(report["comparisons"]), 7)
        self.assertFalse(report["mask_mismatches"])
        for row in report["comparisons"]:
            self.assertTrue(row["raw_exact_match"])
            self.assertTrue(row["feature_only_exact_match"])

    def test_fixed_down_preference_is_detected(self):
        model = LinearQ()
        model.W[ACTIONS.index("DOWN"), FEATURE_NAMES.index("BOMB_POSSIBLE")] = 1.0
        report = inspect_state(state(), {"model": model})
        self.assertTrue(any(r["raw_disjoint"] for r in report["comparisons"]))
        self.assertTrue(any(r["feature_only_disjoint"] for r in report["comparisons"]))

    def test_ties_and_asymmetric_occupancy_map_as_sets(self):
        s = state()
        s["field"][3, 2] = -1
        s["others"] = [("opp", 0, True, (3, 4))]
        original_field = s["field"].copy()
        report = inspect_state(s, {"model": LinearQ()})
        self.assertFalse(report["mask_mismatches"])
        self.assertTrue(all(r["raw_exact_match"] for r in report["comparisons"]))
        np.testing.assert_array_equal(s["field"], original_field)


if __name__ == "__main__":
    unittest.main()
