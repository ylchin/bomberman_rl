"""
features.py 

Turns the raw `game_state` dict (as documented in the assignment PDF, Sec. 5)
into inputs a model can consume. Two consumers:

  - state_to_features(game_state) -> flat vector      for Model 1 (linear/forest)
  - state_to_channels(game_state) -> (C, W, H) stack   for Model 2 (CNN / DQN)

Both are built on top of `danger_map`, which is also imported by Yi Ling Chin's
rewards.py (so bomb-danger logic lives in exactly one place).

ACTIONS ORDER IS LAW. Every index below (in ACTIONS, in one-hot blocks, in
symmetry.py) assumes this exact order. Do not reorder without updating
symmetry.py and every trained checkpoint.
"""

from collections import deque
import numpy as np

# ---------------------------------------------------------------------------
# Fixed action order. index = action id, used everywhere (features, symmetry,
# model outputs). WAIT and BOMB are not directional and are never permuted
# by symmetry transforms.
# ---------------------------------------------------------------------------
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# (dx, dy) in game_state['field'] coordinates (x, y), matching the framework's
# image-coordinate convention noted in the assignment PDF.
DIRECTION_VECTORS = {
    'UP':    (0, -1),
    'RIGHT': (1, 0),
    'DOWN':  (0, 1),
    'LEFT':  (-1, 0),
}
DIRECTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']  # the 4 directional actions, in order

BOMB_POWER = 3       # blast radius in tiles, each cardinal direction (settings.py default)
BOMB_TIMER = 4       # steps from drop to detonation (countdown starts at BOMB_TIMER - 1)
EXPLOSION_LINGER = 1 # extra steps the fire remains lethal after detonating

MAX_DIST_NORM = 20.0   # cap used to normalize BFS distances into [0, 1]
MAX_DANGER_NORM = float(BOMB_TIMER + EXPLOSION_LINGER)  # cap for danger-step normalization


# ---------------------------------------------------------------------------
# Feature vector layout (D = 32). Keep this list and the code that fills it
# in sync — symmetry.py's DIRECTIONAL_GROUPS depends on these exact offsets.
# ---------------------------------------------------------------------------
FEATURE_NAMES = (
    ['WALL_UP', 'WALL_RIGHT', 'WALL_DOWN', 'WALL_LEFT']       # 0-3   blocked neighbor?
    + ['BOMB_POSSIBLE']                                          # 4     can I drop a bomb now
    + ['IN_DANGER']                                              # 5     standing in a future/-current blast
    + ['DANGER_STEPS']                                           # 6     normalized steps until lethal (0=safe)
    + ['COIN_DIR_UP', 'COIN_DIR_RIGHT', 'COIN_DIR_DOWN', 'COIN_DIR_LEFT', 'COIN_DIR_NONE']   # 7-11
    + ['COIN_DIST']                                              # 12
    + ['CRATE_DIR_UP', 'CRATE_DIR_RIGHT', 'CRATE_DIR_DOWN', 'CRATE_DIR_LEFT', 'CRATE_DIR_NONE']  # 13-17
    + ['CRATE_DIST']                                             # 18
    + ['SAFE_DIR_UP', 'SAFE_DIR_RIGHT', 'SAFE_DIR_DOWN', 'SAFE_DIR_LEFT', 'SAFE_DIR_NONE']    # 19-23
    + ['OPP_NEARBY']                                             # 24    opponent within trap distance
    + ['OPP_DIR_UP', 'OPP_DIR_RIGHT', 'OPP_DIR_DOWN', 'OPP_DIR_LEFT', 'OPP_DIR_NONE']  # 25-29
    + ['OPP_DIST']                                               # 30
    + ['WOULD_HIT_OPPONENT']                                     # 31    bombing now would hit an opponent
)
FEATURE_DIM = len(FEATURE_NAMES)  # 32

# Directional one-hot blocks, used by symmetry.py to know which slices to
# permute under rotation/flip. Each tuple = (start_index, has_none_slot).
# The first 4 entries starting at `start_index` are [UP, RIGHT, DOWN, LEFT];
# if has_none_slot, a 5th "NONE" entry follows and is left untouched.
DIRECTIONAL_GROUPS = [
    (0, False),   # WALL_*
    (7, True),    # COIN_DIR_*
    (13, True),   # CRATE_DIR_*
    (19, True),   # SAFE_DIR_*
    (25, True),   # OPP_DIR_*
]

# Channel stack layout for state_to_channels — order is fixed, document it here.
CHANNEL_NAMES = ['walls', 'crates', 'coins', 'self', 'others', 'bomb_danger', 'explosion']
N_CHANNELS = len(CHANNEL_NAMES)  # 7


# ---------------------------------------------------------------------------
# Bomb blast geometry — single source of truth, reused by danger_map and by
# rewards.py (import this, don't reimplement blast logic elsewhere).
# ---------------------------------------------------------------------------
def get_blast_coords(bomb_xy, field):
    """
    Tiles covered by a bomb's explosion if it detonates at bomb_xy right now.
    Stops at stone walls (-1). Passes through (and destroys) crates, per the
    assignment: "explosion destroys crates ... will stop at stone walls."
    Does not wrap around corners (cardinal directions only).
    """
    x, y = bomb_xy
    w, h = field.shape
    coords = [(x, y)]
    for dx, dy in DIRECTION_VECTORS.values():
        for i in range(1, BOMB_POWER + 1):
            nx, ny = x + dx * i, y + dy * i
            if not (0 <= nx < w and 0 <= ny < h):
                break
            if field[nx, ny] == -1:  # stone wall stops the blast
                break
            coords.append((nx, ny))
    return coords


def danger_map(game_state):
    """
    (W, H) int array. Convention:
        0   -> tile is safe (not covered by any current bomb/explosion)
        N>0 -> tile becomes/remains lethal in N steps from *now*
               (N=1 means "lethal on the very next step, move now")
    Combines:
      - each active bomb's blast coverage, valued at (countdown + 1)
      - currently active explosions (game_state['explosion_map']), valued at 1
    Takes the min across sources per tile (soonest danger wins).
    Used by features.py (IN_DANGER / DANGER_STEPS / SAFE_DIR) and by
    Yi Ling Chin's rewards.py for shaping bomb-avoidance reward.
    """
    field = game_state['field']
    w, h = field.shape
    dmap = np.zeros((w, h), dtype=np.int32)

    for (bx, by), countdown in game_state['bombs']:
        steps_to_blast = countdown + 1  # countdown==0 means "explodes next step"
        for (x, y) in get_blast_coords((bx, by), field):
            if dmap[x, y] == 0 or steps_to_blast < dmap[x, y]:
                dmap[x, y] = steps_to_blast

    explosion_map = game_state.get('explosion_map')
    if explosion_map is not None:
        live = explosion_map > 0
        # currently on fire = lethal right now (1 step)
        dmap[live] = np.where((dmap[live] == 0) | (dmap[live] > 1), 1, dmap[live])

    return dmap


# ---------------------------------------------------------------------------
# BFS pathfinding — generic shortest-path-to-nearest-target on the free-tile
# graph. Returns (direction_name_or_None, distance_or_None).
# ---------------------------------------------------------------------------
def _walkable(field, x, y):
    w, h = field.shape
    return 0 <= x < w and 0 <= y < h and field[x, y] == 0


def bfs_direction(game_state, targets, avoid_danger=False, danger=None):
    """
    BFS from the agent's position to the nearest tile in `targets`
    (a set/list of (x, y) coords). Returns (first_step_direction, distance).
    If no target is reachable, returns (None, None).

    If avoid_danger=True, tiles with danger[x,y] == 1 (lethal next step) are
    treated as walls — used for SAFE_DIR routing so we don't path through
    something about to explode. Pass a precomputed `danger` map to avoid
    recomputing it multiple times per act() call.
    """
    _, _, _, (sx, sy) = game_state['self']
    field = game_state['field']
    target_set = set(targets)
    if (sx, sy) in target_set:
        return None, 0  # already there

    visited = {(sx, sy)}
    # queue entries: (x, y, first_step_direction)
    queue = deque()
    for d in DIRECTIONS:
        dx, dy = DIRECTION_VECTORS[d]
        nx, ny = sx + dx, sy + dy
        if _walkable(field, nx, ny) and not (avoid_danger and danger is not None and danger[nx, ny] == 1):
            visited.add((nx, ny))
            queue.append((nx, ny, d, 1))

    while queue:
        x, y, first_dir, dist = queue.popleft()
        if (x, y) in target_set:
            return first_dir, dist
        for dx, dy in DIRECTION_VECTORS.values():
            nx, ny = x + dx, y + dy
            if (nx, ny) in visited:
                continue
            if not _walkable(field, nx, ny):
                continue
            if avoid_danger and danger is not None and danger[nx, ny] == 1:
                continue
            visited.add((nx, ny))
            queue.append((nx, ny, first_dir, dist + 1))

    return None, None


def _one_hot_direction(direction):
    """4-slot one-hot for UP/RIGHT/DOWN/LEFT plus a trailing NONE slot (5 total)."""
    vec = [0, 0, 0, 0, 0]
    if direction is None:
        vec[4] = 1
    else:
        vec[DIRECTIONS.index(direction)] = 1
    return vec


# ---------------------------------------------------------------------------
# Main entry point 1: flat feature vector for Model 1 (linear / boosted-tree)
# ---------------------------------------------------------------------------
def state_to_features(game_state):
    """
    game_state -> np.ndarray of shape (FEATURE_DIM,), dtype float32.
    Returns None only if game_state is None (e.g. terminal call pattern some
    training loops use) — callers must handle that case explicitly.
    """
    if game_state is None:
        return None

    field = game_state['field']
    _, _, bomb_possible, (sx, sy) = game_state['self']
    dmap = danger_map(game_state)

    feats = np.zeros(FEATURE_DIM, dtype=np.float32)

    # --- WALL_* (0-3): is the neighbor tile blocked (wall or crate)? ---
    for i, d in enumerate(DIRECTIONS):
        dx, dy = DIRECTION_VECTORS[d]
        nx, ny = sx + dx, sy + dy
        w, h = field.shape
        blocked = not (0 <= nx < w and 0 <= ny < h) or field[nx, ny] != 0
        feats[i] = float(blocked)

    # --- BOMB_POSSIBLE (4) ---
    feats[4] = float(bomb_possible)

    # --- IN_DANGER / DANGER_STEPS (5-6) ---
    my_danger = dmap[sx, sy]
    feats[5] = float(my_danger > 0)
    feats[6] = min(my_danger, MAX_DANGER_NORM) / MAX_DANGER_NORM

    # --- COIN_DIR / COIN_DIST (7-12) ---
    coins = game_state['coins']
    coin_dir, coin_dist = bfs_direction(game_state, coins) if coins else (None, None)
    feats[7:12] = _one_hot_direction(coin_dir)
    feats[12] = 0.0 if coin_dist is None else min(coin_dist, MAX_DIST_NORM) / MAX_DIST_NORM

    # --- CRATE_DIR / CRATE_DIST (13-18) ---
    crate_coords = list(zip(*np.where(field == 1)))
    crate_dir, crate_dist = bfs_direction(game_state, crate_coords) if crate_coords else (None, None)
    feats[13:18] = _one_hot_direction(crate_dir)
    feats[18] = 0.0 if crate_dist is None else min(crate_dist, MAX_DIST_NORM) / MAX_DIST_NORM

    # --- SAFE_DIR (19-23): nearest tile with danger==0, avoiding walk-through-fire ---
    if my_danger > 0:
        w, h = field.shape
        safe_tiles = [(x, y) for x in range(w) for y in range(h)
                      if field[x, y] == 0 and dmap[x, y] == 0]
        safe_dir, _ = bfs_direction(game_state, safe_tiles, avoid_danger=True, danger=dmap)
    else:
        safe_dir = None
    feats[19:24] = _one_hot_direction(safe_dir)

    # --- OPP_NEARBY / OPP_DIR / OPP_DIST / WOULD_HIT_OPPONENT (24-31) ---
    others = game_state['others']
    opp_coords = [pos for (_, _, _, pos) in others]
    if opp_coords:
        opp_dir, opp_dist = bfs_direction(game_state, opp_coords)
        manhattan_min = min(abs(sx - ox) + abs(sy - oy) for ox, oy in opp_coords)
        feats[24] = float(manhattan_min <= (2 * BOMB_POWER + 1))
        feats[25:30] = _one_hot_direction(opp_dir)
        feats[30] = 0.0 if opp_dist is None else min(opp_dist, MAX_DIST_NORM) / MAX_DIST_NORM
        blast_if_bombed_here = set(get_blast_coords((sx, sy), field))
        feats[31] = float(bool(blast_if_bombed_here & set(opp_coords)))
    # else: leave OPP_* block as zeros / NONE-implicit (index 29 stays 0 too —
    # acceptable since OPP_NEARBY=0 already signals "no opponents visible")

    return feats


# ---------------------------------------------------------------------------
# Main entry point 2: channel stack for Model 2 (CNN / DQN)
# ---------------------------------------------------------------------------
def state_to_channels(game_state):
    """
    game_state -> np.ndarray of shape (N_CHANNELS, W, H), dtype float32.
    Channel order (see CHANNEL_NAMES): walls, crates, coins, self, others,
    bomb_danger (normalized), explosion (normalized).
    Returns None only if game_state is None.
    """
    if game_state is None:
        return None

    field = game_state['field']
    w, h = field.shape
    channels = np.zeros((N_CHANNELS, w, h), dtype=np.float32)

    channels[0] = (field == -1).astype(np.float32)   # walls
    channels[1] = (field == 1).astype(np.float32)     # crates

    for (cx, cy) in game_state['coins']:
        channels[2, cx, cy] = 1.0

    _, _, _, (sx, sy) = game_state['self']
    channels[3, sx, sy] = 1.0

    for (_, _, _, (ox, oy)) in game_state['others']:
        channels[4, ox, oy] = 1.0

    dmap = danger_map(game_state)
    channels[5] = np.minimum(dmap, MAX_DANGER_NORM).astype(np.float32) / MAX_DANGER_NORM

    explosion_map = game_state.get('explosion_map')
    if explosion_map is not None:
        channels[6] = np.clip(explosion_map, 0, None).astype(np.float32)
        max_val = channels[6].max()
        if max_val > 0:
            channels[6] = channels[6] / max_val

    return channels
