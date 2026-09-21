"""Terminal shaping must remove the successor potential exactly once."""

from collections import Counter
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import events as e
from agent_code.our_agent import rewards, train
from agent_code.our_agent.features import FEATURE_DIM


class TerminalShapingTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        # Isolate accounting from board geometry, using nonzero potentials so
        # missing or double terminal corrections cannot accidentally pass.
        self.stack.enter_context(patch.object(
            rewards, "potential", side_effect=lambda s: 0.0 if s is None else s["phi"]
        ))
        self.stack.enter_context(patch.object(train, "detect_custom_events", return_value=[]))
        for name in ["_learn", "_save_periodic_candidate", "_update_best_checkpoint", "_write_csv_row"]:
            self.stack.enter_context(patch.object(train, name))
        for name in ["_epsilon", "_alpha"]:
            self.stack.enter_context(patch.object(train, name, return_value=0.0))
        self.stack.enter_context(patch.object(train, "action_options", return_value={}))
        self.saved = {}

        def capture(agent):
            self.saved["rewards"] = [transition[2] for transition in agent.traj]
            self.saved["total"] = agent.ep_reward
            self.saved["counts"] = agent.ep_counts.copy()

        self.stack.enter_context(patch.object(train, "_flush_episode_to_buffer", side_effect=capture))
        self.agent = SimpleNamespace(
            t={"gamma": 0.9, "learn_every": 10000, "learn_iters_end": 0},
            logger=Mock(), encode_state=lambda s: np.zeros(FEATURE_DIM),
            traj=[], traj_masks=[], ep_reward=0.0, ep_counts=Counter(),
            step_count=0, episode=0, model=Mock(), _weights_path="unused.pkl",
        )

    def test_death_gets_negative_last_state_potential(self):
        train.end_of_round(self.agent, {"phi": 2.0}, "WAIT", [e.GOT_KILLED])
        expected = rewards.reward_from_events([e.GOT_KILLED]) - 2.0
        self.assertEqual(self.saved["rewards"], [expected])
        self.assertEqual(self.saved["total"], expected)

    def test_survivor_corrects_successor_without_duplicating_final_events(self):
        before, after = {"phi": 2.0}, {"phi": 3.0}
        train.game_events_occurred(self.agent, before, "RIGHT", after, [e.COIN_COLLECTED])
        train.end_of_round(self.agent, before, "RIGHT", [e.COIN_COLLECTED, e.SURVIVED_ROUND])
        expected = rewards.reward_from_events([e.COIN_COLLECTED, e.SURVIVED_ROUND]) - 2.0
        self.assertEqual(len(self.saved["rewards"]), 1)
        self.assertAlmostEqual(self.saved["rewards"][0], expected)
        self.assertAlmostEqual(self.saved["total"], expected)
        self.assertEqual(self.saved["counts"][e.COIN_COLLECTED], 1)
        self.assertEqual(self.saved["counts"][e.SURVIVED_ROUND], 1)
        self.assertIsNone(self.agent._last_post_state)

    def test_discounted_shaping_telescopes_through_survivor_terminal(self):
        first, second, final = {"phi": 2.0}, {"phi": -1.0}, {"phi": 4.0}
        train.game_events_occurred(self.agent, first, "RIGHT", second, [])
        train.game_events_occurred(self.agent, second, "RIGHT", final, [])
        train.end_of_round(self.agent, second, "RIGHT", [e.SURVIVED_ROUND])
        r0, r1 = self.saved["rewards"]
        terminal_bonus = rewards.reward_from_events([e.SURVIVED_ROUND])
        self.assertAlmostEqual(r0 + 0.9 * (r1 - terminal_bonus), -first["phi"])

    def test_discounted_shaping_telescopes_through_death(self):
        first, last = {"phi": 2.0}, {"phi": -1.0}
        train.game_events_occurred(self.agent, first, "RIGHT", last, [])
        train.end_of_round(self.agent, last, "WAIT", [e.GOT_KILLED])
        r0, r1 = self.saved["rewards"]
        terminal_reward = rewards.reward_from_events([e.GOT_KILLED])
        self.assertAlmostEqual(r0 + 0.9 * (r1 - terminal_reward), -first["phi"])

    def test_successor_is_not_carried_into_next_round(self):
        train.game_events_occurred(self.agent, {"phi": 2.0}, "RIGHT", {"phi": 8.0}, [])
        train.end_of_round(self.agent, {"phi": 2.0}, "RIGHT", [e.SURVIVED_ROUND])
        train.end_of_round(self.agent, {"phi": -1.0}, "WAIT", [e.GOT_KILLED])
        self.assertEqual(self.saved["rewards"], [rewards.reward_from_events([e.GOT_KILLED]) + 1.0])


if __name__ == "__main__":
    unittest.main()
