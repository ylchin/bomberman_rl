from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from agent_code.our_agent import callbacks, config
from agent_code.our_agent.coin_navigation import prefer_nearby_coin, safe_coin_actions
from agent_code.our_agent.features import ACTIONS, state_to_features
from agent_code.our_agent.q_linear import LinearQ
from experiments.compare_task4_guard import guard_settings
from experiments.analyze_task4_guard import metric_fields


def board(coin=(5, 3)):
    field = np.zeros((9, 9), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    return dict(field=field, coins=[coin], bombs=[], others=[],
                explosion_map=np.zeros_like(field), self=("me", 0, True, (3, 3)),
                step=1, round=1)


def policy(s, **values):
    model = LinearQ()
    phi = state_to_features(s)
    feature = np.flatnonzero(phi)[0]
    for action, value in {"WAIT": 10, **values}.items():
        model.W[ACTIONS.index(action), feature] = value / phi[feature]
    return model


class CoinPreferenceTests(unittest.TestCase):
    def choose(self, s, action="WAIT", mask=None, model=None):
        model = model or policy(s)
        mask = np.ones(6, dtype=bool) if mask is None else mask
        return ACTIONS[prefer_nearby_coin(s, state_to_features(s), ACTIONS.index(action),
                                        model, mask, np.random.default_rng(1))]

    def test_chooses_highest_q_among_all_shortest_screened_moves(self):
        s = board((4, 4))
        model = policy(s, RIGHT=1, DOWN=2, UP=8)
        self.assertEqual(self.choose(s, model=model), "DOWN")
        mask = np.ones(6, dtype=bool)
        mask[ACTIONS.index("DOWN")] = False
        self.assertEqual(self.choose(s, mask=mask, model=model), "RIGHT")

    def test_missing_empty_or_coin_excluding_screen_preserves_wait(self):
        s = board()
        model = policy(s)
        for mask in (None, np.zeros(6, dtype=bool), [False, False, False, False, True, True]):
            with self.subTest(mask=mask):
                self.assertEqual(prefer_nearby_coin(s, state_to_features(s), 4, model,
                                                   mask, np.random.default_rng(1)), 4)

    def test_bombs_and_fire_anywhere_preserve_wait(self):
        s = board()
        s["bombs"] = [((1, 7), 3)]
        self.assertEqual(self.choose(s), "WAIT")
        s["bombs"] = []
        s["explosion_map"][1, 7] = 1
        self.assertEqual(self.choose(s), "WAIT")

    def test_six_step_boundary_and_absent_or_unreachable_coins(self):
        s = board((7, 5))
        self.assertEqual(safe_coin_actions(s)[1], 6)
        self.assertNotEqual(self.choose(s), "WAIT")
        s["coins"] = [(7, 6)]
        self.assertEqual(self.choose(s), "WAIT")
        s["coins"] = []
        self.assertEqual(self.choose(s), "WAIT")
        s = board()
        s["field"][:] = -1
        s["field"][3, 3] = s["field"][5, 3] = 0
        self.assertEqual(self.choose(s), "WAIT")

    def test_opponent_blocks_direct_step_and_distance_counts_detour(self):
        s = board((7, 4))  # Five steps before the detour.
        # Block both shortest forward exits with opponents, forcing LEFT/UP.
        s["others"] = [("a", 0, True, (4, 3)), ("b", 0, True, (3, 4))]
        self.assertEqual(safe_coin_actions(s)[1], 7)
        self.assertEqual(self.choose(s), "WAIT")

    def test_nonwait_choices_consume_no_extra_randomness(self):
        s = board()
        model = policy(s)
        rng = np.random.default_rng(9)
        original = deepcopy(rng.bit_generator.state)
        for action in (0, 1, 2, 3, 5):
            self.assertEqual(prefer_nearby_coin(s, state_to_features(s), action,
                                              model, np.ones(6, dtype=bool), rng), action)
        self.assertEqual(rng.bit_generator.state, original)

    def agent(self, s, train=False):
        return SimpleNamespace(model=policy(s), model_kind="linear", train=train,
                               encode_state=state_to_features, epsilon=0.0,
                               rng=np.random.default_rng(1), logger=Mock(),
                               _last_pos=None, _last_action_id=None, _repeat_count=0)

    def test_callback_flag_switches_only_evaluation_wait_and_records_change(self):
        s = board()
        original = deepcopy(s)
        agent = self.agent(s)
        weights = agent.model.W.copy()
        with patch.object(config, "MODEL", "linear"), patch.dict(config.TRAIN, {
            "survival_filter": True, "bomb_collision_guard": True,
            "escape_collision_guard": True, "optimistic_fallback": False,
            "coin_preference": False, "softmax_beta": None,
        }):
            self.assertEqual(callbacks.act(agent, s), "WAIT")
            self.assertFalse(agent._coin_preference_applied)
            with patch.dict(config.TRAIN, coin_preference=True):
                self.assertEqual(callbacks.act(agent, s), "RIGHT")
                self.assertTrue(agent._coin_preference_applied)
                self.assertEqual(agent._pre_coin_action, "WAIT")
                self.assertEqual(callbacks.act(self.agent(s, train=True), s), "WAIT")
            self.assertEqual(callbacks.act(agent, s), "WAIT")
            self.assertFalse(agent._coin_preference_applied)
        np.testing.assert_array_equal(agent.model.W, weights)
        np.testing.assert_array_equal(s["field"], original["field"])
        self.assertEqual(s["coins"], original["coins"])

    def test_comparison_changes_only_coin_preference(self):
        for enabled in (False, True):
            self.assertEqual(guard_settings("coin", enabled), {
                "bomb_collision_guard": True, "escape_collision_guard": True,
                "optimistic_fallback": False, "coin_preference": enabled,
            })
            for kind in ("bomb", "escape", "fallback"):
                self.assertFalse(guard_settings(kind, enabled)["coin_preference"])
        self.assertIn("coin_preference_steps", metric_fields([{"coin_preference_steps": "2"}]))


if __name__ == "__main__":
    unittest.main()
