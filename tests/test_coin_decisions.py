import unittest

import numpy as np

from agent_code.our_agent.features import ACTIONS, state_to_features
from agent_code.our_agent.q_linear import LinearQ
from experiments.compare_coin_decisions import compare_state, safe_coin_actions


def board(coin=(5, 3)):
    field = np.zeros((9, 9), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    return dict(field=field, coins=[coin], bombs=[], others=[],
                explosion_map=np.zeros_like(field), self=("me", 0, True, (3, 3)),
                step=1, round=1)


class CoinDecisionTests(unittest.TestCase):
    def test_all_equally_short_coin_directions_are_retained(self):
        actions, distance = safe_coin_actions(board((4, 4)))
        self.assertEqual({ACTIONS[a] for a in actions}, {"RIGHT", "DOWN"})
        self.assertEqual(distance, 2)

    def test_opponent_occupancy_blocks_direct_coin_route(self):
        state = board()
        state["others"] = [("opp", 0, True, (4, 3))]
        actions, distance = safe_coin_actions(state)
        self.assertNotIn(ACTIONS.index("RIGHT"), actions)
        self.assertEqual(distance, 4)

    def test_danger_and_unreachable_coins_are_excluded(self):
        state = board()
        state["bombs"] = [((3, 3), 3)]
        self.assertEqual(safe_coin_actions(state), ([], None))
        state = board()
        state["field"][:] = -1
        state["field"][3, 3] = state["field"][5, 3] = 0
        self.assertEqual(safe_coin_actions(state), ([], None))

    def policies(self, state, tie=False):
        baseline, candidate = LinearQ(), LinearQ()
        phi = state_to_features(state)
        feature = np.flatnonzero(phi)[0]
        baseline.W[ACTIONS.index("RIGHT"), feature] = 3.0 / phi[feature]
        candidate.W[ACTIONS.index("WAIT"), feature] = 3.0 / phi[feature]
        if tie:
            candidate.W[ACTIONS.index("RIGHT"), feature] = 3.0 / phi[feature]
        return baseline, candidate

    def test_strict_loss_and_feature_attribution(self):
        state = board()
        baseline, candidate = self.policies(state)
        row, example = compare_state(state, baseline, candidate)
        self.assertTrue(row["strict_coin_preference_loss"])
        self.assertEqual(row["baseline_greedy"], ["RIGHT"])
        self.assertEqual(row["candidate_greedy"], ["WAIT"])
        self.assertEqual(example["baseline_alternative_minus_coin"], -3.0)
        self.assertEqual(example["candidate_alternative_minus_coin"], 3.0)
        self.assertEqual(sum(x["shift_toward_alternative"] for x in example["largest_feature_shifts"]), 6.0)

    def test_tied_coin_action_is_not_reported_as_strict_loss(self):
        state = board()
        row, example = compare_state(state, *self.policies(state, tie=True))
        self.assertFalse(row["strict_coin_preference_loss"])
        self.assertEqual(row["candidate_coin_probability"], 0.5)
        self.assertIsNone(example)


if __name__ == "__main__":
    unittest.main()
