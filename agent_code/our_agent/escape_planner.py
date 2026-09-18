"""Short-horizon survival search matching this framework's bomb update order.

Other agents are held at their observed positions. Future bomb placements and
opponent moves are unknown, so a route is a prediction, not a guarantee.
"""

from collections import deque

import numpy as np

from .features import DIRECTION_VECTORS, EXPLOSION_LINGER, get_blast_coords


def escape_route(state, start, blocked=(), placement_turn=False):
    """Return (survives, first move or None for WAIT, moves to refuge).

    `placement_turn` forces the agent to stay put for the first update: placing a
    new bomb consumes an action. Callers include that bomb with its initial timer.
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
            lethal[detonation : detonation + EXPLOSION_LINGER + 1, x, y] = True
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
        if not lethal[t + 1 :, pos[0], pos[1]].any():
            return True, first, t - start_time
        if t == horizon:
            continue
        nt = t + 1
        for name, (dx, dy) in moves:
            nxt = (pos[0] + dx, pos[1] + dy)
            x, y = nxt
            if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]):
                continue
            if opens_at[x, y] > nt or nxt in blocked or lethal[nt, x, y]:
                continue
            # Standing on a just-placed bomb is allowed; re-entering it is not.
            if nxt != pos and occupied_until.get(nxt, 0) >= nt:
                continue
            if (nxt, nt) in visited:
                continue
            visited.add((nxt, nt))
            queue.append((nxt, nt, name if t == start_time else first))
    return False, None, None
