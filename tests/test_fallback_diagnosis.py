from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from experiments.analyze_task4_fallback import action_routes, inspect_death
from agent_code.our_agent.features import state_to_features
from agent_code.our_agent.q_linear import LinearQ
from agent_code.our_agent import callbacks, config
from agent_code.our_agent.escape_planner import optimistic_fallback_actions
from experiments.compare_task4_guard import guard_settings
from experiments.analyze_task4_guard import metric_fields


def trapped_state(step=192):
    """Local geometry and bombs from seed 30089, with distant crates omitted."""
    field = np.full((17, 17), -1, dtype=int)
    field[1:-1, 1:-1] = 0
    field[2:-1:2, 2:-1:2] = -1
    field[13, 14] = 1
    return dict(
        field=field, self=("me", 3, False, (13, 12)),
        others=[("a", 0, False, (13, 10)), ("b", 0, False, (12, 13)),
                ("c", 0, True, (14, 13))],
        bombs=[((13, 11), 2), ((12, 13), 3), ((13, 12), 3)],
        coins=[], explosion_map=np.zeros_like(field), step=step, round=1,
    )


def placement_state():
    s = trapped_state(191)
    s["self"] = ("me", 3, True, (13, 12))
    s["others"] = [("a", 0, False, (13, 11)), ("b", 0, True, (12, 13)),
                   ("c", 0, True, (13, 13))]
    s["bombs"] = [((13, 11), 3)]
    return s


def frame(s, action):
    return dict(state=s, action=action, mask=[False] * 6, events=[])


class FallbackDiagnosisTests(unittest.TestCase):
    def test_newly_opened_down_exit_has_only_optimistic_route(self):
        death = dict(seed=30089, step=195, history=[
            frame(placement_state(), "BOMB"), frame(trapped_state(), "WAIT"),
        ])
        result = inspect_death(death)["fallback_decisions"][-1]
        self.assertEqual(result["newly_opened_moves_while_stationary"], ["DOWN"])
        self.assertEqual(result["routes_by_legal_action"]["DOWN"], {
            "static_opponent_route": False, "optimistic_route": True,
        })
        self.assertFalse(result["chosen_has_optimistic_route"])
        self.assertEqual(result["alternatives_with_optimistic_route"], ["DOWN"])
        self.assertTrue(result["missed_newly_opened_exit"])

    def test_optimism_does_not_allow_initially_occupied_moves(self):
        s = placement_state()
        original = deepcopy(s)
        routes = action_routes(s, LinearQ.legal_actions(state_to_features(s)))
        self.assertEqual(set(routes), {"WAIT", "BOMB"})
        self.assertEqual(s["bombs"], original["bombs"])
        self.assertEqual(s["others"], original["others"])
        np.testing.assert_array_equal(s["field"], original["field"])

    def test_taking_possible_exit_is_not_flagged_as_missed(self):
        death = dict(seed=1, step=195, history=[frame(trapped_state(), "DOWN")])
        result = inspect_death(death)["fallback_decisions"][0]
        self.assertTrue(result["chosen_has_optimistic_route"])
        self.assertFalse(result["missed_possible_escape"])
        self.assertFalse(result["missed_newly_opened_exit"])

    def test_bomb_candidate_includes_its_own_future_explosion(self):
        s = trapped_state()
        s["field"][:] = -1
        s["field"][13, 12] = 0
        s["self"] = ("me", 0, True, (13, 12))
        s["others"] = []
        s["bombs"] = []
        routes = action_routes(s, LinearQ.legal_actions(state_to_features(s)))
        self.assertTrue(routes["WAIT"]["optimistic_route"])
        self.assertFalse(routes["BOMB"]["optimistic_route"])

    def test_nonempty_legal_screen_is_not_fallback(self):
        f = frame(trapped_state(), "DOWN")
        f["mask"][2] = True
        result = inspect_death(dict(seed=1, step=195, history=[f]))
        self.assertEqual(result["fallback_decisions"], [])

    def test_missing_steps_do_not_establish_newly_opened_exit(self):
        death = dict(seed=1, step=195, history=[
            frame(placement_state(), "BOMB"), frame(trapped_state(193), "WAIT"),
        ])
        result = inspect_death(death)["fallback_decisions"][-1]
        self.assertEqual(result["newly_opened_moves_while_stationary"], [])
        self.assertFalse(result["missed_newly_opened_exit"])


class OptimisticFallbackPolicyTests(unittest.TestCase):
    def test_flag_selects_down_instead_of_wait_and_masks_bootstrap(self):
        s = trapped_state()
        phi = state_to_features(s)
        model = LinearQ()
        model.W[4] = phi * 100  # WAIT outranks DOWN without the new preference.
        model.W[2] = phi
        with patch.object(config, "MODEL", "linear"), patch.dict(config.TRAIN, {
            "survival_filter": True, "bomb_collision_guard": True,
            "escape_collision_guard": True, "optimistic_fallback": False,
        }):
            baseline = callbacks.action_options(s)["action_mask"]
            self.assertFalse(baseline.any())
            self.assertEqual(model.act(phi, action_mask=baseline), 4)
            with patch.dict(config.TRAIN, optimistic_fallback=True):
                improved = callbacks.action_options(s, phi=phi)["action_mask"]
                np.testing.assert_array_equal(improved, [False, False, True, False, False, False])
                self.assertEqual(model.act(phi, action_mask=improved), 2)
                for options in ({"epsilon": 1.0}, {"beta": 1.0}):
                    self.assertEqual(model.act(phi, action_mask=improved, **options), 2)
                # Training calls action_options without an already encoded phi.
                np.testing.assert_array_equal(callbacks.action_options(s)["action_mask"], improved)
                np.testing.assert_allclose(
                    model.bootstrap_value(phi[None], [0], next_action_mask=improved[None]),
                    [model.q_values(phi)[2]],
                )

    def test_nonempty_legal_screen_is_preserved_even_without_optimistic_route(self):
        s = trapped_state()
        legal = LinearQ.legal_actions(state_to_features(s))
        mask = np.array([False, False, False, False, True, False])
        np.testing.assert_array_equal(optimistic_fallback_actions(s, mask, legal), mask)

    def test_mask_with_only_illegal_actions_still_uses_fallback(self):
        s = trapped_state()
        legal = LinearQ.legal_actions(state_to_features(s))
        mask = np.array([False, True, False, False, False, False])  # RIGHT is a wall.
        np.testing.assert_array_equal(
            optimistic_fallback_actions(s, mask, legal),
            [False, False, True, False, False, False],
        )

    def test_no_possible_route_preserves_original_legal_fallback(self):
        s = trapped_state(193)
        s["bombs"] = [(pos, timer - 1) for pos, timer in s["bombs"]]
        legal = LinearQ.legal_actions(state_to_features(s))
        mask = np.zeros(6, dtype=bool)
        result = optimistic_fallback_actions(s, mask, legal)
        np.testing.assert_array_equal(result, mask)
        np.testing.assert_array_equal(
            LinearQ.available_actions(state_to_features(s), result), legal,
        )

    def test_initial_opponent_occupancy_remains_blocked_and_bomb_still_possible(self):
        s = placement_state()
        legal = LinearQ.legal_actions(state_to_features(s))
        original = deepcopy(s)
        result = optimistic_fallback_actions(s, np.zeros(6, dtype=bool), legal)
        np.testing.assert_array_equal(result, [False, False, False, False, True, True])
        self.assertEqual(s["bombs"], original["bombs"])
        np.testing.assert_array_equal(s["field"], original["field"])

    def test_bomb_with_no_escape_is_not_preferred(self):
        s = placement_state()
        s["field"][:] = -1
        s["field"][13, 12] = 0
        s["bombs"] = []
        s["others"] = []
        legal = LinearQ.legal_actions(state_to_features(s))
        result = optimistic_fallback_actions(s, np.zeros(6, dtype=bool), legal)
        np.testing.assert_array_equal(result, [False, False, False, False, True, False])

    def test_disabled_survival_filter_does_not_enable_fallback(self):
        with patch.object(config, "MODEL", "linear"), patch.dict(config.TRAIN, {
            "survival_filter": False, "optimistic_fallback": True,
        }):
            self.assertEqual(callbacks.action_options(trapped_state()), {})

    def test_comparison_changes_only_fallback_flag(self):
        self.assertEqual(guard_settings("fallback", False), {
            "bomb_collision_guard": True, "escape_collision_guard": True,
            "optimistic_fallback": False,
            "coin_preference": False,
        })
        self.assertEqual(guard_settings("fallback", True), {
            "bomb_collision_guard": True, "escape_collision_guard": True,
            "optimistic_fallback": True,
            "coin_preference": False,
        })
        for kind in ("bomb", "escape"):
            for enabled in (False, True):
                self.assertFalse(guard_settings(kind, enabled)["optimistic_fallback"])

    def test_survival_metrics_are_not_inferred_for_legacy_results(self):
        self.assertNotIn("survived", metric_fields([{"self_kill": "0"}]))
        fields = metric_fields([{"survived": "0", "died": "1", "fallback_steps": "3",
                                 "optimistic_fallback_steps": "1"}])
        for key in ("survived", "died", "fallback_steps", "optimistic_fallback_steps"):
            self.assertIn(key, fields)


if __name__ == "__main__":
    unittest.main()
