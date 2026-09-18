"""Checks the corrected Task-3 opponent and bomb-interaction features."""

import numpy as np
from .features import (
    is_dead_end, opponent_trapped, proposed_bomb_analysis,
    state_to_features, FEATURE_NAMES,
)

OPP_TRAPPED_IDX = FEATURE_NAMES.index('OPP_TRAPPED')
OPP_DEADEND_IDX = FEATURE_NAMES.index('OPP_NEAR_DEADEND')
SAFE_HIT_IDX = FEATURE_NAMES.index('SAFE_BOMB_HITS_OPPONENT')
SAFE_TRAP_IDX = FEATURE_NAMES.index('SAFE_BOMB_TRAPS_OPPONENT')
SAFE_CRATE_IDX = FEATURE_NAMES.index('SAFE_BOMB_HITS_CRATE')


def _state(field, self_pos, others=(), coins=(), bombs=(), bomb_possible=True):
    return {
        'field': field,
        'bombs': list(bombs),
        'explosion_map': np.zeros_like(field),
        'coins': list(coins),
        'self': ('me', 0, bomb_possible, self_pos),
        'others': [('opp' + str(i), 0, True, pos) for i, pos in enumerate(others)],
        'round': 1, 'step': 1, 'user_input': None,
    }


def _open_board():
    field = np.full((9, 9), -1, dtype=np.int8)
    field[1:8, 1:8] = 0
    return field


def _trapped_board():
    field = _open_board()
    # Opponent pocket at (1,1); its only exit is (1,2), where OUR proposed
    # bomb is placed. This specifically tests bomb-at-self geometry.
    field[2, 1] = -1
    field[2, 2] = -1
    field[1, 3] = -1
    return field


def check_actual_bomb_geometry():
    field = _trapped_board()
    trapped = _state(field, self_pos=(1, 2), others=[(1, 1)])
    ok = True

    if not is_dead_end(field, 1, 1):
        print('FAIL: (1,1) should be a dead end')
        ok = False
    if not opponent_trapped(trapped, (1, 1)):
        print('FAIL: actual bomb at self=(1,2) should trap opponent at (1,1)')
        ok = False

    # Same opponent in open space, still inside our blast, can escape.
    open_state = _state(_open_board(), self_pos=(5, 3), others=[(5, 5)])
    if opponent_trapped(open_state, (5, 5)):
        print('FAIL: opponent in open space should escape our proposed bomb')
        ok = False

    if ok:
        print('OK: opponent_trapped uses our actual proposed bomb and time-aware escape.')
    return ok


def check_feature_wiring():
    ok = True
    trapped_state = _state(_trapped_board(), self_pos=(1, 2), others=[(1, 1)])
    feats = state_to_features(trapped_state)
    if feats[OPP_TRAPPED_IDX] != 1.0 or feats[OPP_DEADEND_IDX] != 1.0:
        print('FAIL: OPP_TRAPPED/OPP_NEAR_DEADEND not wired to corrected target')
        ok = False

    open_state = _state(_open_board(), self_pos=(5, 3), others=[(5, 5)])
    open_feats = state_to_features(open_state)
    if open_feats[SAFE_HIT_IDX] != 1.0:
        print('FAIL: expected SAFE_BOMB_HITS_OPPONENT=1 on an escapable aligned bomb')
        ok = False
    if open_feats[SAFE_TRAP_IDX] != 0.0:
        print('FAIL: open opponent should not set SAFE_BOMB_TRAPS_OPPONENT')
        ok = False

    crate_field = _open_board()
    crate_field[5, 5] = 1
    crate_state = _state(crate_field, self_pos=(5, 3))
    crate_feats = state_to_features(crate_state)
    if crate_feats[SAFE_CRATE_IDX] != 1.0:
        print('FAIL: expected SAFE_BOMB_HITS_CRATE=1 for safe crate-clearing bomb')
        ok = False

    if ok:
        print('OK: corrected opponent + explicit safe-bomb interaction features are wired through.')
    return ok


if __name__ == '__main__':
    ok1 = check_actual_bomb_geometry()
    ok2 = check_feature_wiring()
    if ok1 and ok2:
        print('\nAll opponent-feature checks passed.')
    else:
        raise SystemExit('\nSome opponent-feature checks failed.')
