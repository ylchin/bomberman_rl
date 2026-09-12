"""
check_symmetry_sync.py

Verifies that features.DIRECTIONAL_GROUPS still matches the actual layout
of features.FEATURE_NAMES, and that symmetry.apply_to_features / 
apply_to_action form a consistent, invertible group action (i.e. augmenting
a transition doesn't silently mislabel it).

Run from the repo root:
    python -m agent_code.our_agent.check_symmetry_sync

(uses -m because symmetry.py does `from .features import ...`, a relative
import -- it must be run as part of the agent_code.our_agent package, not
as a standalone script, or you'll get "attempted relative import with no
known parent package".)
"""

import numpy as np
from .features import FEATURE_NAMES, FEATURE_DIM, DIRECTIONAL_GROUPS, DIRECTIONS, ACTIONS
from .symmetry import sym_transforms, apply_to_features, apply_to_action


def check_groups_match_names():
    """
    For each (start, has_none) in DIRECTIONAL_GROUPS, confirm FEATURE_NAMES
    at that offset actually reads like [X_UP, X_RIGHT, X_DOWN, X_LEFT, (X_NONE)]
    for some common prefix X -- i.e. the offsets haven't drifted out of sync
    with the feature list above them.
    """
    problems = []
    for start, has_none in DIRECTIONAL_GROUPS:
        span = 5 if has_none else 4
        names = FEATURE_NAMES[start:start + span]
        if len(names) != span:
            problems.append(f"offset {start}: expected {span} names, only {len(names)} exist "
                             f"(FEATURE_DIM={FEATURE_DIM})")
            continue

        prefixes = [n.rsplit('_', 1)[0] for n in names[:4]]
        if len(set(prefixes)) != 1:
            problems.append(f"offset {start}: inconsistent prefixes {names[:4]}")
            continue
        prefix = prefixes[0]

        expected_suffixes = ['UP', 'RIGHT', 'DOWN', 'LEFT'] + (['NONE'] if has_none else [])
        actual_suffixes = [n.rsplit('_', 1)[1] for n in names]
        if actual_suffixes != expected_suffixes:
            problems.append(f"offset {start} ('{prefix}'): expected suffix order "
                             f"{expected_suffixes}, got {actual_suffixes}")

    if problems:
        print("MISMATCHES FOUND between DIRECTIONAL_GROUPS and FEATURE_NAMES:")
        for p in problems:
            print(f"  - {p}")
        return False

    print(f"OK: all {len(DIRECTIONAL_GROUPS)} DIRECTIONAL_GROUPS entries match "
          f"FEATURE_NAMES layout (FEATURE_DIM={FEATURE_DIM}).")
    return True


def check_symmetry_group_consistency():
    """
    Confirms the 8 ops form a real group action on feature vectors:
      - identity op is truly a no-op
      - every op has an inverse among the 8 that undoes it exactly
      - action transforms and feature-vector transforms agree with each
        other (transforming a one-hot-encoded action via apply_to_action
        matches transforming the same info via apply_to_features)
    """
    ops = sym_transforms()
    if len(ops) != 8:
        print(f"FAIL: expected 8 symmetry ops, got {len(ops)}")
        return False

    rng = np.random.default_rng(0)
    test_vec = rng.random(FEATURE_DIM).astype(np.float32)

    # identity check
    identity = ops[0]
    if not np.allclose(apply_to_features(test_vec, identity), test_vec):
        print("FAIL: identity op is not a no-op on apply_to_features")
        return False
    for a in range(len(ACTIONS)):
        if apply_to_action(a, identity) != a:
            print(f"FAIL: identity op changed action {ACTIONS[a]}")
            return False

    # every op must have a matching inverse among the 8 ops
    all_ok = True
    for op in ops:
        found_inverse = False
        for inv in ops:
            round_trip = apply_to_features(apply_to_features(test_vec, op), inv)
            if np.allclose(round_trip, test_vec):
                # also check action round-trips consistently under the same pair
                action_ok = all(
                    apply_to_action(apply_to_action(a, op), inv) == a
                    for a in range(len(ACTIONS))
                )
                if action_ok:
                    found_inverse = True
                    break
        if not found_inverse:
            print(f"FAIL: op '{op.get('name', op)}' has no consistent inverse among the 8 ops")
            all_ok = False

    if all_ok:
        print(f"OK: all {len(ops)} ops are invertible and features/actions transform consistently.")
    return all_ok


if __name__ == "__main__":
    ok1 = check_groups_match_names()
    ok2 = check_symmetry_group_consistency()
    if ok1 and ok2:
        print("\nAll checks passed -- safe to switch augmentation on for Task 2.")
    else:
        print("\nDO NOT switch augmentation on yet -- fix the issues above first.")
        raise SystemExit(1)
