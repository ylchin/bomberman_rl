"""Short-horizon survival search matching this framework's bomb update order.

Other agents are held at their observed positions. Future bomb placements and
opponent moves are unknown, so a route is a prediction, not a guarantee.
"""

from collections import deque

import numpy as np

from .features import ACTIONS, BOMB_TIMER, DIRECTION_VECTORS, EXPLOSION_LINGER, get_blast_coords


def _opponent_arrival_times(state):
    """Earliest arrival through currently open tiles, including simultaneous moves."""
    field = state["field"]
    arrival = np.full(field.shape, np.inf)
    bombs = {pos for pos, _ in state["bombs"]}
    queue = deque()
    for opponent in state["others"]:
        pos = opponent[3]
        arrival[pos] = 0
        queue.append(pos)
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTION_VECTORS.values():
            nxt = (x + dx, y + dy)
            nx, ny = nxt
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue
            if field[nxt] != 0 or nxt in bombs or np.isfinite(arrival[nxt]):
                continue
            arrival[nxt] = arrival[x, y] + 1
            queue.append(nxt)
    return arrival


def escape_route(state, start, blocked=(), placement_turn=False, first_action=None, deadline_margin=0,
                 avoid_opponent_collisions=False):
    """Return (survives, first move or None for WAIT, moves to refuge).

    `placement_turn` forces the agent to stay put for the first update: placing a
    new bomb consumes an action. Callers include that bomb with its initial timer.
    `first_action` forces the first move (including WAIT), for action screening.
    `deadline_margin` expands blast intervals earlier without clearing obstacles
    or fire earlier. It reserves time for an opponent obstructing the route.
    `avoid_opponent_collisions` rejects moves to tiles an opponent can reach by
    that turn. Used before committing to a new bomb; it assumes opponents can
    wait, but cannot predict new bombs or paths opened by future crate removal.
    """
    field = state["field"]
    opponent_arrival = _opponent_arrival_times(state) if avoid_opponent_collisions else None
    bombs = [(tuple(pos), int(timer) + 1) for pos, timer in state["bombs"]]
    existing_fire = np.asarray(state.get("explosion_map", np.zeros_like(field)))
    horizon = max(
        [1, int(existing_fire.max())]
        + [detonation + EXPLOSION_LINGER for _, detonation in bombs]
    )
    lethal = np.zeros((horizon + 1, *field.shape), dtype=bool)
    opens_at = np.where(field == 0, 0, horizon + 1)
    occupied_until = {}
    for t in range(1, horizon + 1):
        lethal[t] |= existing_fire >= t
    for pos, detonation in bombs:
        occupied_until[pos] = max(occupied_until.get(pos, 0), detonation)
        for x, y in get_blast_coords(pos, field):
            lethal[max(1, detonation - deadline_margin) : detonation + EXPLOSION_LINGER + 1, x, y] = True
            if field[x, y] == 1:
                # Moves occur before crates are destroyed in this step.
                opens_at[x, y] = min(opens_at[x, y], detonation + 1)

    # Bomb occupancy is time-dependent; callers may include it in blocked.
    blocked = set(blocked) - set(occupied_until) - {start}
    start_time = int(placement_turn)
    if placement_turn and lethal[1, start[0], start[1]]:
        return False, None, None
    queue = deque([(start, start_time, None)])
    visited = {(start, start_time)}
    moves = [(name, delta) for name, delta in DIRECTION_VECTORS.items()] + [
        (None, (0, 0))
    ]
    while queue:
        pos, t, first = queue.popleft()
        if (first_action is None or t > start_time) and not lethal[t + 1 :, pos[0], pos[1]].any():
            return True, first, t - start_time
        if t == horizon:
            continue
        nt = t + 1
        for name, (dx, dy) in moves:
            if t == start_time and first_action is not None and (name or "WAIT") != first_action:
                continue
            nxt = (pos[0] + dx, pos[1] + dy)
            x, y = nxt
            if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]):
                continue
            if opens_at[x, y] > nt or nxt in blocked or lethal[nt, x, y]:
                continue
            # Agents choose simultaneously and execute in random order. An
            # enemy can win a race to an empty tile and invalidate our move.
            # Waiting keeps an occupied tile, so is not a collision risk.
            if nxt != pos and opponent_arrival is not None and opponent_arrival[nxt] <= nt:
                continue
            # Standing on a just-placed bomb is allowed; re-entering it is not.
            if nxt != pos and occupied_until.get(nxt, 0) >= nt:
                continue
            if (nxt, nt) in visited:
                continue
            visited.add((nxt, nt))
            queue.append((nxt, nt, name if t == start_time else first))
    return False, None, None


def survival_actions(state, deadline_margin=1, bomb_collision_guard=True):
    """Actions leaving a route through all currently known explosions.

    New bombs optionally require an escape avoiding tiles opponents can reach
    first. Existing-bomb escapes retain the usual search, so the guard never
    removes a last-chance movement route. All-false means no route was found;
    the learned policy then falls back to its legal actions rather than freezing.
    This screen restricts choices, but Q values still rank the retained actions.
    """
    start = state['self'][3]
    opponents = {a[3] for a in state['others']}
    allowed = np.zeros(len(ACTIONS), dtype=bool)
    for i, action in enumerate(ACTIONS[:-1]):
        allowed[i] = escape_route(state, start, blocked=opponents, first_action=action,
                                  deadline_margin=deadline_margin)[0]
    if state['self'][2]:
        proposed = dict(state)
        proposed['bombs'] = list(state['bombs']) + [(start, BOMB_TIMER)]
        allowed[-1] = escape_route(proposed, start, blocked=opponents, placement_turn=True,
                                   deadline_margin=deadline_margin,
                                   avoid_opponent_collisions=bomb_collision_guard)[0]
    # If no conservative route exists, still use an exact-timing escape before
    # falling back to legal actions. Do not throw away the only surviving move.
    if deadline_margin and not allowed.any():
        return survival_actions(state, deadline_margin=0, bomb_collision_guard=bomb_collision_guard)
    return allowed
