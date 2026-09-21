"""Coin progress must not depend on BFS direction tie-breaking."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from agent_code.our_agent import rewards


def state(coins):
    field = np.full((9, 9), -1, dtype=int)
    field[1:-1, 1:-1] = 0
    return dict(field=field, self=("me", 0, True, (3, 3)), others=[],
                bombs=[], coins=list(coins), explosion_map=np.zeros_like(field),
                step=1, round=1)


def moved_state(old, action):
    new = deepcopy(old)
    dx, dy = rewards.DIRECTION_VECTORS[action]
    x, y = old["self"][3]
    pos = (x + dx, y + dy)
    new["self"] = (*old["self"][:3], pos)
    new["coins"] = [coin for coin in old["coins"] if coin != pos]
    return new


class CoinTieRewardTests(unittest.TestCase):
    def coin_events(self, old, action, new, enabled=True):
        with patch.object(rewards, "COIN_TIE_REWARD", enabled):
            events = rewards.detect_custom_events(old, action, new)
        return [ev for ev in events if ev in
                (rewards.MOVED_TOWARD_COIN, rewards.MOVED_AWAY_FROM_COIN)]

    def test_both_shortest_routes_to_one_coin_get_progress_reward(self):
        old = state([(5, 1)])
        for action in ("UP", "RIGHT"):
            with self.subTest(action=action):
                self.assertEqual(self.coin_events(old, action, moved_state(old, action)),
                                 [rewards.MOVED_TOWARD_COIN])

    def test_disabled_preserves_original_direction_tie_break(self):
        old = state([(5, 1)])
        self.assertEqual(self.coin_events(old, "RIGHT", moved_state(old, "RIGHT"), False),
                         [rewards.MOVED_AWAY_FROM_COIN])
        self.assertEqual(self.coin_events(old, "UP", moved_state(old, "UP"), False),
                         [rewards.MOVED_TOWARD_COIN])

    def test_equal_nearest_coins_and_collection_use_old_targets(self):
        old = state([(3, 2), (4, 3)])
        new = moved_state(old, "RIGHT")
        self.assertNotIn((4, 3), new["coins"])
        self.assertEqual(self.coin_events(old, "RIGHT", new),
                         [rewards.MOVED_TOWARD_COIN])
        self.assertEqual(old["self"][3], (3, 3))
        self.assertEqual(old["coins"], [(3, 2), (4, 3)])

    def test_true_nonprogress_stays_penalized(self):
        old = state([(5, 1)])
        for action in ("DOWN", "LEFT"):
            with self.subTest(action=action):
                self.assertEqual(self.coin_events(old, action, moved_state(old, action)),
                                 [rewards.MOVED_AWAY_FROM_COIN])

    def test_obstacles_use_path_distance_not_manhattan_distance(self):
        old = state([(5, 3)])
        old["field"][4, 2:5] = 1
        # Equal shortest detours around the top and bottom of this crate wall.
        for action in ("UP", "DOWN"):
            with self.subTest(action=action):
                self.assertEqual(self.coin_events(old, action, moved_state(old, action)),
                                 [rewards.MOVED_TOWARD_COIN])

    def test_newly_cleared_crates_do_not_reclassify_old_route(self):
        old = state([(5, 3)])
        old["field"][4, 2:5] = 1
        new = moved_state(old, "DOWN")
        new["field"][4, 2:5] = 0
        self.assertEqual(self.coin_events(old, "DOWN", new),
                         [rewards.MOVED_TOWARD_COIN])
        np.testing.assert_array_equal(old["field"][4, 2:5], [1, 1, 1])

    def test_no_shaping_for_failed_move_wait_or_bomb_request(self):
        old = state([(5, 1)])
        for action in ("RIGHT", "WAIT", "BOMB"):
            with self.subTest(action=action):
                self.assertEqual(self.coin_events(old, action, deepcopy(old)), [])

    def test_no_shaping_without_reachable_target(self):
        for coins in ([], [(3, 3)], [(7, 7)]):
            with self.subTest(coins=coins):
                old = state(coins)
                old["field"][6, 7] = old["field"][7, 6] = -1
                self.assertEqual(self.coin_events(old, "RIGHT", moved_state(old, "RIGHT")), [])

    def test_danger_handling_keeps_precedence_over_coin_progress(self):
        old = state([(5, 1)])
        old["bombs"] = [((3, 3), 3)]
        self.assertEqual(self.coin_events(old, "RIGHT", moved_state(old, "RIGHT")), [])


if __name__ == "__main__":
    unittest.main()
