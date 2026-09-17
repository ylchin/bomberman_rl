"""
check_opponent_features.py

Verifies OPP_TRAPPED / OPP_NEAR_DEADEND / is_dead_end / opponent_trapped
against hand-built boards with a known right answer, so we're not just
trusting the code compiles.

Run from the repo root:
    python -m agent_code.our_agent.check_opponent_features
"""

import numpy as np
from .features import (
    is_dead_end, opponent_trapped, state_to_features, FEATURE_NAMES,
)

OPP_TRAPPED_IDX = FEATURE_NAMES.index('OPP_TRAPPED')
OPP_DEADEND_IDX = FEATURE_NAMES.index('OPP_NEAR_DEADEND')


def _make_board():
    """9x9 board: border walls, open interior, one genuine 2-tile dead-end
    pocket at (1,1)-(1,2) with no room to outrun a blast, and open space
    elsewhere for a free/untrapped opponent."""
    field = np.full((9, 9), -1, dtype=np.int8)
    field[1:8, 1:8] = 0
    field[2, 1] = -1   # seal the pocket's only other side
    field[2, 2] = -1
    field[1, 3] = -1   # seal off further corridor -- pocket is ONLY (1,1),(1,2)
    return field


def check_dead_end_and_trapped():
    field = _make_board()
    ok = True

    if not is_dead_end(field, 1, 1):
        print("FAIL: (1,1) should be a dead end (only 1 walkable neighbor)")
        ok = False
    if is_dead_end(field, 5, 5):
        print("FAIL: (5,5) is open, should NOT be a dead end")
        ok = False

    if not opponent_trapped(field, (1, 1)):
        print("FAIL: opponent at (1,1) is genuinely cornered, should be trapped")
        ok = False
    if opponent_trapped(field, (5, 5)):
        print("FAIL: opponent at (5,5) is in open space, should NOT be trapped")
        ok = False

    if ok:
        print("OK: is_dead_end / opponent_trapped correct on hand-built boards.")
    return ok


def check_state_to_features_wires_it_through():
    """Confirms the feature vector's indices 32/33 actually reflect the
    board, not just that the standalone functions work in isolation."""
    field = _make_board()
    ok = True

    trapped_state = {
        'field': field, 'bombs': [], 'explosion_map': np.zeros_like(field),
        'coins': [], 'self': ('me', 0, True, (5, 3)),
        'others': [('opp', 0, True, (1, 1))],
        'round': 1, 'step': 1, 'user_input': None,
    }
    free_state = {**trapped_state, 'others': [('opp', 0, True, (5, 5))]}

    trapped_feats = state_to_features(trapped_state)
    free_feats = state_to_features(free_state)

    if trapped_feats[OPP_TRAPPED_IDX] != 1.0:
        print(f"FAIL: OPP_TRAPPED should be 1.0 when nearest opp is cornered, "
              f"got {trapped_feats[OPP_TRAPPED_IDX]}")
        ok = False
    if trapped_feats[OPP_DEADEND_IDX] != 1.0:
        print(f"FAIL: OPP_NEAR_DEADEND should be 1.0, got {trapped_feats[OPP_DEADEND_IDX]}")
        ok = False
    if free_feats[OPP_TRAPPED_IDX] != 0.0:
        print(f"FAIL: OPP_TRAPPED should be 0.0 for an opponent in open space, "
              f"got {free_feats[OPP_TRAPPED_IDX]}")
        ok = False

    if ok:
        print("OK: state_to_features correctly wires OPP_TRAPPED/OPP_NEAR_DEADEND "
              "through to indices 32/33.")
    return ok


if __name__ == "__main__":
    ok1 = check_dead_end_and_trapped()
    ok2 = check_state_to_features_wires_it_through()
    if ok1 and ok2:
        print("\nAll checks passed.")
    else:
        print("\nSome checks failed -- see above.")
        raise SystemExit(1)
