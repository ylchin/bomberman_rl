"""
rewards.py

Turns the framework's event list (+ the game states around a step) into a
scalar training reward.

  1. reward_from_events(events)      base reward from predefined + custom events
  2. detect_custom_events(old, a, new)   our own events, appended before (1)
  3. potential_shaping(old, new, gamma)  potential-based shaping term F = gamma*phi(s') - phi(s)
"""

import math
import os

import events as e
from .features import (
    DIRECTIONS,
    DIRECTION_VECTORS,
    danger_map,
    bfs_direction,
    crate_approach_targets,
    select_opponent_target,
    proposed_bomb_analysis,
    MAX_DIST_NORM,
    _escape_route_from,
)

# ---------------------------------------------------------------------------
# Custom event names. Keep them here so train.py and rewards.py agree.
# ---------------------------------------------------------------------------
MOVED_TOWARD_COIN = "MOVED_TOWARD_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"
MOVED_TOWARD_SAFETY = "MOVED_TOWARD_SAFETY"
STAYED_IN_DANGER = "STAYED_IN_DANGER"
MOVED_INTO_DANGER = "MOVED_INTO_DANGER"
ESCAPED_DANGER = "ESCAPED_DANGER"
BOMB_NEXT_TO_CRATE = "BOMB_NEXT_TO_CRATE"
USELESS_BOMB = "USELESS_BOMB"
BOMB_WITH_NO_ESCAPE = "BOMB_WITH_NO_ESCAPE"
SAFE_BOMB_THREATENS_OPPONENT = "SAFE_BOMB_THREATENS_OPPONENT"
SAFE_BOMB_TRAPS_OPPONENT = "SAFE_BOMB_TRAPS_OPPONENT"

# ---------------------------------------------------------------------------
# Reward table. Every value is a hyperparameter -- tune via experiments/.
# Keep the true game rewards (COIN_COLLECTED=1, KILLED_OPPONENT=5) roughly at
# their real scale so shaped and true return stay comparable.
# ---------------------------------------------------------------------------
ESCAPE_REWARD_SCALE = float(os.environ.get("AGENT_ESCAPE_REWARD_SCALE", "1"))
if not math.isfinite(ESCAPE_REWARD_SCALE) or ESCAPE_REWARD_SCALE < 0:
    raise ValueError("AGENT_ESCAPE_REWARD_SCALE must be finite and non-negative")

# Experimental training-only correction: reward every shortest coin move,
# including directions that lose the BFS traversal-order tie-break.
COIN_TIE_REWARD = os.environ.get("AGENT_COIN_TIE_REWARD", "0") == "1"

GAME_REWARDS = {
    # --- real objective (matches settings.py REWARD_COIN / REWARD_KILL) ---
    e.COIN_COLLECTED: 1.0,
    e.KILLED_OPPONENT: 5.0,
    # --- staying alive ---
    e.KILLED_SELF: -5.0,
    e.GOT_KILLED: -5.0,
    e.SURVIVED_ROUND: 0.5,
    # --- progress toward opening the board ---
    e.CRATE_DESTROYED: 0.10,
    e.COIN_FOUND: 0.05,
    # --- anti-dithering / anti-noop ---
    e.INVALID_ACTION: -0.10,
    e.WAITED: -0.02,
    # --- custom (see detect_custom_events) ---
    MOVED_TOWARD_COIN: 0.06,
    MOVED_AWAY_FROM_COIN: -0.07,  # slightly harsher than the reward: no oscillation gain
    MOVED_TOWARD_SAFETY: 0.30 * ESCAPE_REWARD_SCALE,
    MOVED_INTO_DANGER: -0.35,
    STAYED_IN_DANGER: -0.25,
    ESCAPED_DANGER: 0.30 * ESCAPE_REWARD_SCALE,
    BOMB_NEXT_TO_CRATE: 0.05,
    USELESS_BOMB: -0.10,
    BOMB_WITH_NO_ESCAPE: -0.55,
    # Small, separate aggression shaping. Keep these modest relative to the true
    # +5 kill reward; tune one lever at a time if Task 3 needs more aggression.
    SAFE_BOMB_THREATENS_OPPONENT: 0.10,
    SAFE_BOMB_TRAPS_OPPONENT: 0.20,
    # Tried doubling these (0.60/0.45/0.80) + n_step=5 + symmetry together on
    # 2026-09-17: official eval coins dropped 28.1 -> 9.7, training curve
    # plateaued at ep~3000/8000. Reverted. If retrying, change ONE of these
    # three things at a time so a regression is attributable.
    #
    # 2026-09-17, second finding: with -0.40 unchanged, more training time
    # alone (3000 -> 6000 rounds, nothing else changed) raised coins
    # 35.55->37.90 but self-kill 6%->11% -- self-kill is a structural issue,
    # not an undertraining issue.
    #
    # 2026-09-17, third: -0.40 -> -0.55 alone (n_step=3, no symmetry, 3000
    # rounds, otherwise = baseline): coins 35.55->37.42, self-kill 6%->4%.
    # Both moved the right way together -- confirmed single-variable win.
    #
    # 2026-09-17, fourth: pushed -0.55 -> -0.70. Self-kill kept improving
    # (4%->1%) but coins collapsed 37.42->21.98 (variance 11.9->18.6) --
    # model got too bomb-shy to clear crates. Overshoot. REVERTED to -0.55,
    # the best coins/safety point found on this single lever. Neither -0.55
    # nor -0.70 clears both exit-criteria thresholds simultaneously (<2%
    # self-kill AND >=40/50 coins) -- this axis alone won't get there; the
    # remaining self-kill gap likely needs a feature/execution fix (better
    # escape routing), not just a bigger penalty. -0.55 is current best.
}


def reward_from_events(events, logger=None):
    """Sum the table over the (predefined + custom) events for this step."""
    total = sum(GAME_REWARDS.get(ev, 0.0) for ev in events)
    if logger is not None:
        logger.debug(f"reward {total:+.3f} from {events}")
    return float(total)


# ---------------------------------------------------------------------------
# Custom event detection
# ---------------------------------------------------------------------------
def _agent_pos(game_state):
    return game_state["self"][3]


def _moved_action(self_action, old_state, new_state):
    """The direction actually moved this step, or None (blocked / waited / bombed)."""
    if self_action not in DIRECTIONS:
        return None
    ox, oy = _agent_pos(old_state)
    nx, ny = _agent_pos(new_state)
    if (nx - ox, ny - oy) == DIRECTION_VECTORS[self_action]:
        return self_action
    return None


def detect_custom_events(old_state, self_action, new_state, framework_events=()):
    """
    Return extra event strings for old -> new. Bomb-quality rewards are emitted
    only when the framework confirms BOMB_DROPPED, so an unavailable/invalid
    BOMB request can never receive bomb-placement shaping.
    """
    if old_state is None or new_state is None or self_action is None:
        return []

    ev = []
    old_pos = _agent_pos(old_state)
    new_pos = _agent_pos(new_state)
    old_danger = danger_map(old_state)
    new_danger = danger_map(new_state)
    in_danger_before = old_danger[old_pos] > 0
    in_danger_after = new_danger[new_pos] > 0
    bomb_info = (
        proposed_bomb_analysis(old_state)
        if self_action == "BOMB" and e.BOMB_DROPPED in framework_events
        else None
    )

    # --- danger handling dominates: if we were in a blast path, only judge escape ---
    if in_danger_before:
        opp_positions = {pos for (_, _, _, pos) in old_state["others"]}
        bomb_positions = {pos for pos, _ in old_state["bombs"]}
        blocked = opp_positions | bomb_positions
        can_escape, safe_dir, _ = _escape_route_from(
            old_state, old_pos, blocked=blocked
        )
        moved = _moved_action(self_action, old_state, new_state)

        if not in_danger_after:
            ev.append(ESCAPED_DANGER)
        elif moved is not None and moved == safe_dir:
            ev.append(MOVED_TOWARD_SAFETY)
        elif can_escape and safe_dir is None and self_action == "WAIT":
            ev.append(MOVED_TOWARD_SAFETY)
        else:
            ev.append(STAYED_IN_DANGER)
    else:
        # --- moved into a fresh blast path (e.g. walked next to a ticking bomb) ---
        # A successful bomb necessarily puts us in its blast path. When the
        # placement leaves an escape route, judge it using bomb quality below.
        # Keep the danger penalty for unsafe placements and failed requests.
        safe_placement = bomb_info is not None and bomb_info["can_escape"]
        if in_danger_after and not safe_placement:
            ev.append(MOVED_INTO_DANGER)

        # --- coin seeking (only when safe and coins are visible) ---
        coins = old_state["coins"]
        if coins:
            coin_dir, coin_dist = bfs_direction(old_state, coins)
            moved = _moved_action(self_action, old_state, new_state)
            if moved is not None and coin_dir is not None:
                toward_coin = moved == coin_dir
                if COIN_TIE_REWARD and not toward_coin:
                    # Compare distances on the SAME board to the SAME coins.
                    # Collection or another bomb clearing a crate this turn
                    # must not change whether our move made progress.
                    after_move = dict(old_state)
                    after_move["self"] = (*old_state["self"][:3], new_pos)
                    _, remaining = bfs_direction(after_move, coins)
                    toward_coin = remaining == coin_dist - 1
                ev.append(
                    MOVED_TOWARD_COIN if toward_coin else MOVED_AWAY_FROM_COIN
                )

    # --- bomb quality, judged only after a SUCCESSFUL placement ---
    if bomb_info is not None:
        info = bomb_info

        if info["hits_crate"]:
            ev.append(BOMB_NEXT_TO_CRATE)
        if not info["hits_crate"] and not info["hits_opponent"]:
            ev.append(USELESS_BOMB)

        if not info["can_escape"]:
            ev.append(BOMB_WITH_NO_ESCAPE)
        elif info["hits_opponent"]:
            ev.append(SAFE_BOMB_THREATENS_OPPONENT)
            if info["target_trapped"]:
                ev.append(SAFE_BOMB_TRAPS_OPPONENT)

    return ev


# ---------------------------------------------------------------------------
# Potential-based shaping  (Lecture 36)  --  F = gamma*Phi(s') - Phi(s)
# ---------------------------------------------------------------------------
def potential(game_state):
    """
    Phi(s): higher is better. Afunction of state (never of the action).

    Design:
      + closeness to the nearest reachable coin        (weight 1.0)
      + closeness to a tile from which a crate can be bombed, only when no
        coin is currently visible                       (weight 0.5)
      + closeness to the same BFS-selected opponent used by Model 1
      - being in a blast path, scaled by how soon it detonates

    Bomb/trap opportunity is intentionally NOT placed in the potential: after
    dropping a bomb, BOMB_POSSIBLE becomes false, so an action-opportunity
    potential could accidentally punish the very bomb action it is meant to
    encourage. Successful safe threats are shaped as events instead.
    """
    if game_state is None:
        return 0.0

    phi = 0.0
    pos = _agent_pos(game_state)
    field = game_state["field"]

    coins = game_state["coins"]
    if coins:
        _, dist = bfs_direction(game_state, coins)
        if dist is not None:
            phi += 1.0 - min(dist, MAX_DIST_NORM) / MAX_DIST_NORM  # in [0, 1]
    else:
        crate_tiles = crate_approach_targets(field)
        if crate_tiles:
            _, dist = bfs_direction(game_state, crate_tiles)
            if dist is not None:
                phi += 0.5 * (1.0 - min(dist, MAX_DIST_NORM) / MAX_DIST_NORM)

    others = game_state["others"]
    if others:
        _, _, opp_dist = select_opponent_target(game_state)
        if opp_dist is not None:
            phi += 1.0 - min(opp_dist, MAX_DIST_NORM) / MAX_DIST_NORM

    dmap = danger_map(game_state)
    d = dmap[pos]
    if d > 0:
        phi -= 1.0 + 1.0 / d  # -2 when about to explode, ~-1.2 when far off

    return float(phi)


def potential_shaping(old_state, new_state, gamma):
    """The additive shaping reward for this transition."""
    return gamma * potential(new_state) - potential(old_state)
