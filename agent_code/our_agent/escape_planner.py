"""Short-horizon survival search matching this framework's bomb update order.

Observed opponent positions block routes; optional collision guards also
consider tiles opponents could reach as known bombs and crates disappear.
Future bomb placements and moves are unknown, so routes are not guarantees.
"""

from collections import deque
from heapq import heappop, heappush

import numpy as np

from .features import ACTIONS, BOMB_TIMER, DIRECTION_VECTORS, EXPLOSION_LINGER, get_blast_coords


def _opponent_arrival_times(state, opens_at, occupied_until, horizon):
    """Optimistic arrival bounds using the escape search's obstacle schedule.

    Opponents may wait for a bomb or crate to disappear. Fire and other agents
    are deliberately ignored: this can predict an arrival earlier than a real
    safe route permits, making collision screening conservative. Only arrivals
    within the escape horizon matter. Stone walls never open.
    """
    field = state["field"]
    arrival = np.full(field.shape, np.inf)
    queue = []
    for opponent in state["others"]:
        pos = tuple(opponent[3])
        arrival[pos] = 0
        heappush(queue, (0, pos))
    while queue:
        t, (x, y) = heappop(queue)
        if t != arrival[x, y]:
            continue
        for dx, dy in DIRECTION_VECTORS.values():
            nxt = (x + dx, y + dy)
            nx, ny = nxt
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue
            if field[nxt] == -1:
                continue
            # Moves precede bomb updates, so a detonating bomb blocks entry
            # through its detonation turn. Crate release times already include
            # the same one-turn offset in opens_at.
            nt = max(t + 1, int(opens_at[nxt]), occupied_until.get(nxt, 0) + 1)
            if nt > horizon or nt >= arrival[nxt]:
                continue
            arrival[nxt] = nt
            heappush(queue, (nt, nxt))
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
    that turn. It includes paths opened by known explosions and assumes
    opponents can wait, even through fire, but cannot predict new bombs.
    """
    field = state["field"]
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

    opponent_arrival = (
        _opponent_arrival_times(state, opens_at, occupied_until, horizon)
        if avoid_opponent_collisions else None
    )

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


def optimistic_fallback_actions(state, action_mask, legal):
    """Prefer possible escapes only when the normal legal safety screen is empty.

    The first action must be legal under observed occupancy. Later moves may
    use opponent-occupied tiles, assuming opponents clear them. Known bombs,
    crates and fire retain exact timing, including a proposed bomb's placement
    turn. This is an emergency preference, not a certified safe route.
    """
    if np.any(np.asarray(action_mask, dtype=bool) & legal):
        return action_mask
    possible = np.zeros(len(ACTIONS), dtype=bool)
    start = state["self"][3]
    for i, action in enumerate(ACTIONS):
        if not legal[i]:
            continue
        if action == "BOMB":
            proposed = dict(state, bombs=[*state["bombs"], (start, BOMB_TIMER)])
            possible[i] = escape_route(proposed, start, placement_turn=True)[0]
        else:
            possible[i] = escape_route(state, start, first_action=action)[0]
    # Preserve the existing legal-action fallback when even optimism finds no
    # route. Do not replace a hopeless screen with an arbitrary new preference.
    return possible if possible.any() else action_mask


def survival_actions(state, deadline_margin=1, bomb_collision_guard=True,
                     escape_collision_guard=False):
    """Actions leaving a route through all currently known explosions.

    New bombs optionally require an escape avoiding tiles opponents can reach
    first. The optional escape guard prefers collision-safe movement routes
    while already in a blast path, but retains the original choices if none
    are found. All-false means no route was found;
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
        return survival_actions(state, deadline_margin=0,
                                bomb_collision_guard=bomb_collision_guard,
                                escape_collision_guard=escape_collision_guard)
    in_blast_path = escape_collision_guard and bool(opponents) and any(
        start in get_blast_coords(pos, state['field']) for pos, _ in state['bombs']
    )
    if in_blast_path:
        robust_moves = np.zeros(len(ACTIONS) - 1, dtype=bool)
        for i, action in enumerate(ACTIONS[:-1]):
            if allowed[i]:
                robust_moves[i] = escape_route(
                    state, start, blocked=opponents, first_action=action,
                    deadline_margin=deadline_margin,
                    avoid_opponent_collisions=True,
                )[0]
        # Do not turn a difficult escape into an empty mask (which would make
        # LinearQ fall back to arbitrary legal actions). Keep the old search
        # when every route is contested. Bomb screening remains independent.
        if robust_moves.any():
            allowed[:-1] = robust_moves
    return allowed
