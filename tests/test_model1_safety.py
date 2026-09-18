import unittest

import numpy as np

from agent_code.our_agent.escape_planner import escape_route
from agent_code.our_agent.features import FEATURE_DIM, state_to_features, proposed_bomb_analysis
from agent_code.our_agent.q_linear import LinearQ


def state(pos=(3, 3)):
    field = np.full((9, 9), -1, dtype=int)
    field[1:-1, 1:-1] = 0
    return dict(field=field, self=('me', 0, True, pos), others=[], bombs=[],
                coins=[], explosion_map=np.zeros_like(field), step=1, round=1)


class Model1SafetyTests(unittest.TestCase):
    def test_wait_for_fire_then_escape(self):
        s = state((3, 3))
        s['field'][:] = -1
        for p in [(3, 3), (4, 3), (4, 4)]:
            s['field'][p] = 0
        s['bombs'] = [((3, 3), 3)]
        s['explosion_map'][4, 3] = 1
        # The only exit is on fire next step; wait, then turn the corner.
        self.assertEqual(escape_route(s, (3, 3)), (True, None, 3))

    def test_later_blast_is_not_hidden_by_earlier_blast(self):
        s = state((3, 3))
        s['field'][:] = -1
        for p in [(3, 3), (4, 3), (5, 3)]:
            s['field'][p] = 0
        s['bombs'] = [((3, 3), 0), ((5, 3), 3)]
        self.assertFalse(escape_route(s, (3, 3))[0])

    def test_bomb_placement_consumes_a_turn(self):
        s = state()
        s['bombs'] = [((3, 5), 0)]
        self.assertTrue(escape_route(s, (3, 3))[0])
        # Walking out succeeds, but spending that turn dropping a bomb dies.
        self.assertFalse(proposed_bomb_analysis(s)['can_escape'])

    def test_cannot_cross_live_bomb(self):
        s = state((2, 3))
        s['field'][:] = -1
        for p in [(2, 3), (3, 3), (3, 4)]:
            s['field'][p] = 0
        s['bombs'] = [((3, 3), 1)]
        self.assertFalse(escape_route(s, (2, 3))[0])

    def test_dynamic_occupancy_and_all_action_selection_modes(self):
        s = state()
        s['field'][3, 2] = -1
        s['others'] = [('opp', 0, True, (4, 3))]
        s['bombs'] = [((3, 4), 3)]
        s['self'] = ('me', 0, False, (3, 3))
        phi = state_to_features(s)
        model = LinearQ()
        # Invalid actions have overwhelmingly larger Q values.
        model.W[[0, 1, 2, 5], 0] = 100
        self.assertEqual(model.legal_actions(phi).tolist(), [False, False, False, True, True, False])
        rng = np.random.default_rng(4)
        for kwargs in ({}, {'epsilon': 1}, {'beta': 1}, {'beta': 0}):
            for _ in range(30):
                self.assertIn(model.act(phi, rng=rng, **kwargs), (3, 4))
        np.testing.assert_array_equal(model.bootstrap_value(phi[None], [0]), [0])
        np.testing.assert_array_equal(model.bootstrap_value(phi[None], [1]), [0])

    def test_escape_search_does_not_mutate_input(self):
        s = state()
        original = s['field'].copy()
        proposed_bomb_analysis(s)
        np.testing.assert_array_equal(s['field'], original)
        self.assertEqual(s['bombs'], [])
        self.assertEqual(state_to_features(s).shape, (FEATURE_DIM,))


if __name__ == '__main__':
    unittest.main()
