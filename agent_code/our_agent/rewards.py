"""
rewards.py

Turns the framework's event list (+ the game states around a step) into a
scalar training reward. 

  1. reward_from_events(events)      base reward from predefined + custom events
  2. detect_custom_events(old, a, new)   our own events, appended before (1)
  3. potential_shaping(old, new, gamma)  potential-based shaping term F = gamma*phi(s') - phi(s)
"""

import events as e
from .features import (
    DIRECTIONS, DIRECTION_VECTORS,
    danger_map, get_blast_coords, bfs_direction, crate_approach_targets,
    MAX_DIST_NORM,
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

# ---------------------------------------------------------------------------
# Reward table. Every value is a hyperparameter -- tune via experiments/.
# Keep the true game rewards (COIN_COLLECTED=1, KILLED_OPPONENT=5) roughly at
# their real scale so shaped and true return stay comparable.
# ---------------------------------------------------------------------------
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
    MOVED_AWAY_FROM_COIN: -0.07,     # slightly harsher than the reward: no oscillation gain
    MOVED_TOWARD_SAFETY: 0.30,
    MOVED_INTO_DANGER: -0.35,
    STAYED_IN_DANGER: -0.25,
    ESCAPED_DANGER: 0.30,
    BOMB_NEXT_TO_CRATE: 0.05,
    USELESS_BOMB: -0.10,
    BOMB_WITH_NO_ESCAPE: -0.40,
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


def detect_custom_events(old_state, self_action, new_state):
    """
    Return a list of extra event strings for the transition old -> new.
    Cheap-ish: at most a couple of BFS calls. Only runs during training.
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
    field = old_state["field"]

    # --- danger handling dominates: if we were in a blast path, only judge escape ---
    if in_danger_before:
        safe_tiles = [
            (x, y)
            for x in range(field.shape[0])
            for y in range(field.shape[1])
            if field[x, y] == 0 and old_danger[x, y] == 0
        ]
        safe_dir, _ = bfs_direction(old_state, safe_tiles, avoid_danger=True, danger=old_danger)
        moved = _moved_action(self_action, old_state, new_state)

        if not in_danger_after:
            ev.append(ESCAPED_DANGER)
        elif moved is not None and moved == safe_dir:
            ev.append(MOVED_TOWARD_SAFETY)
        else:
            ev.append(STAYED_IN_DANGER)
    else:
        # --- moved into a fresh blast path (e.g. walked next to a ticking bomb) ---
        if in_danger_after:
            ev.append(MOVED_INTO_DANGER)

        # --- coin seeking (only when safe and coins are visible) ---
        coins = old_state["coins"]
        if coins:
            coin_dir, _ = bfs_direction(old_state, coins)
            moved = _moved_action(self_action, old_state, new_state)
            if moved is not None and coin_dir is not None:
                ev.append(MOVED_TOWARD_COIN if moved == coin_dir else MOVED_AWAY_FROM_COIN)

    # --- bomb quality, judged at drop time ---
    if self_action == "BOMB":
        blast = set(get_blast_coords(old_pos, field))
        hits_crate = any(field[x, y] == 1 for (x, y) in blast)
        opp_positions = {pos for (_, _, _, pos) in old_state["others"]}
        hits_opp = bool(blast & opp_positions)
        ev.append(BOMB_NEXT_TO_CRATE if (hits_crate or hits_opp) else USELESS_BOMB)

        # can we still reach safety after dropping here?
        hypothetical = {**old_state, "bombs": list(old_state["bombs"]) + [(old_pos, 3)]}
        post_danger = danger_map(hypothetical)
        safe_tiles = [
            (x, y)
            for x in range(field.shape[0])
            for y in range(field.shape[1])
            if field[x, y] == 0 and post_danger[x, y] == 0
        ]
        sdir, sdist = bfs_direction(old_state, safe_tiles, avoid_danger=True, danger=post_danger)
        if sdir is None or (sdist is not None and sdist > 3):
            ev.append(BOMB_WITH_NO_ESCAPE)

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
      - being in a blast path, scaled by how soon it detonates
    Extend with an opponent term for Tasks 3-4.
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

    dmap = danger_map(game_state)
    d = dmap[pos]
    if d > 0:
        phi -= (1.0 + 1.0 / d)  # -2 when about to explode, ~-1.2 when far off

    return float(phi)


def potential_shaping(old_state, new_state, gamma):
    """The additive shaping reward for this transition."""
    return gamma * potential(new_state) - potential(old_state)
