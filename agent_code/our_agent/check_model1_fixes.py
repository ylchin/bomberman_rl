"""Regression checks for the Model-1 fixes that do not require a full game run."""

import numpy as np
import events as e
from .features import FEATURE_DIM, FEATURE_NAMES, state_to_features
from .rewards import (
    detect_custom_events,
    BOMB_NEXT_TO_CRATE,
    USELESS_BOMB,
    BOMB_WITH_NO_ESCAPE,
    SAFE_BOMB_THREATENS_OPPONENT,
    SAFE_BOMB_TRAPS_OPPONENT,
)

BOMB_EVENTS = {
    BOMB_NEXT_TO_CRATE,
    USELESS_BOMB,
    BOMB_WITH_NO_ESCAPE,
    SAFE_BOMB_THREATENS_OPPONENT,
    SAFE_BOMB_TRAPS_OPPONENT,
}


def _open_state(bomb_possible=True, bombs=()):
    field = np.full((9, 9), -1, dtype=np.int8)
    field[1:8, 1:8] = 0
    return {
        "field": field,
        "bombs": list(bombs),
        "explosion_map": np.zeros_like(field),
        "coins": [],
        "self": ("me", 0, bomb_possible, (5, 3)),
        "others": [("opp", 0, True, (5, 5))],
        "round": 1,
        "step": 1,
        "user_input": None,
    }


def check_feature_dimension():
    state = _open_state()
    feats = state_to_features(state)
    ok = len(feats) == FEATURE_DIM == 44 and len(FEATURE_NAMES) == 44
    print(
        "OK" if ok else "FAIL",
        f": FEATURE_DIM={FEATURE_DIM}, len(features)={len(feats)}",
    )
    return ok


def check_invalid_bomb_gets_no_bomb_shaping():
    old = _open_state(bomb_possible=False)
    new = _open_state(bomb_possible=False)
    custom = detect_custom_events(old, "BOMB", new, [e.INVALID_ACTION])
    leaked = BOMB_EVENTS.intersection(custom)
    ok = not leaked
    print(
        "OK: invalid BOMB gets no bomb-quality reward."
        if ok
        else f"FAIL: invalid BOMB leaked custom bomb events: {sorted(leaked)}"
    )
    return ok


def check_successful_bomb_gets_safe_threat_reward():
    old = _open_state(bomb_possible=True)
    # Framework's post-step state has the newly placed bomb at countdown 3.
    new = _open_state(bomb_possible=False, bombs=[((5, 3), 3)])
    custom = detect_custom_events(old, "BOMB", new, [e.BOMB_DROPPED])
    ok = SAFE_BOMB_THREATENS_OPPONENT in custom and BOMB_WITH_NO_ESCAPE not in custom
    print(
        "OK: successful escapable threat is rewarded."
        if ok
        else f"FAIL: unexpected events for successful bomb: {custom}"
    )
    return ok


if __name__ == "__main__":
    checks = [
        check_feature_dimension(),
        check_invalid_bomb_gets_no_bomb_shaping(),
        check_successful_bomb_gets_safe_threat_reward(),
    ]
    if all(checks):
        print("\nAll Model-1 regression checks passed.")
    else:
        raise SystemExit("\nAt least one Model-1 regression check failed.")
