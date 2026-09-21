"""Clear coin paths and an optional evaluation-time preference over WAIT."""

from collections import deque

import numpy as np

from .features import ACTIONS, DIRECTION_VECTORS, danger_map

COIN_PREFERENCE_MAX_DISTANCE = 6


def safe_coin_actions(state):
    """All shortest first moves through currently free tiles outside danger."""
    start = state["self"][3]
    danger = danger_map(state)
    if not state["coins"] or danger[start] > 0:
        return [], None
    walkable = (state["field"] == 0) & (danger == 0)
    for pos in [a[3] for a in state["others"]] + [p for p, _ in state["bombs"]]:
        walkable[pos] = False
    distances = np.full(walkable.shape, -1, dtype=int)
    queue = deque()
    for coin in state["coins"]:
        if walkable[coin] and distances[coin] < 0:
            distances[coin] = 0
            queue.append(coin)
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTION_VECTORS.values():
            nxt = (x + dx, y + dy)
            if not (0 <= nxt[0] < walkable.shape[0] and 0 <= nxt[1] < walkable.shape[1]):
                continue
            if walkable[nxt] and distances[nxt] < 0:
                distances[nxt] = distances[x, y] + 1
                queue.append(nxt)
    distance = int(distances[start])
    if distance <= 0:
        return [], None
    actions = []
    for action, (dx, dy) in DIRECTION_VECTORS.items():
        nxt = (start[0] + dx, start[1] + dy)
        if (0 <= nxt[0] < walkable.shape[0] and 0 <= nxt[1] < walkable.shape[1]
                and walkable[nxt] and distances[nxt] == distance - 1):
            actions.append(ACTIONS.index(action))
    return actions, distance


def prefer_nearby_coin(state, phi, action_id, model, action_mask, rng):
    """Replace WAIT with the highest-Q screened nearby coin move, when eligible.

    No bombs or active fire may exist anywhere. The existing safety mask must
    explicitly permit a legal coin move; an empty/missing screen is never
    relaxed. All equally short first moves compete by Q with normal tie-breaking.
    Opponents can still create new danger after this decision.
    """
    if action_id != ACTIONS.index("WAIT") or action_mask is None:
        return action_id
    if state["bombs"] or np.any(state["explosion_map"] > 0):
        return action_id
    coin_ids, distance = safe_coin_actions(state)
    if not coin_ids or distance > COIN_PREFERENCE_MAX_DISTANCE:
        return action_id
    screened = model.legal_actions(phi) & np.asarray(action_mask, dtype=bool)
    coin_mask = np.zeros(len(ACTIONS), dtype=bool)
    coin_mask[coin_ids] = screened[coin_ids]
    if not coin_mask.any():
        return action_id
    return model.act(phi, epsilon=0.0, rng=rng, action_mask=coin_mask)
