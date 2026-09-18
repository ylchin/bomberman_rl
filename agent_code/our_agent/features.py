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

Legacy 32/34-dim linear checkpoints can be used only as explicit initialization
checkpoints; q_linear.load pads the appended columns with zeros. ***
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
# Feature vector layout (D = 44). Keep this list and the code that fills it
# in sync. Symmetry only permutes directional groups; the appended interaction
# features are scalar/invariant, so adding them does not affect Model 2.
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
    + ['OPP_TRAPPED']                                            # 32    selected opponent cannot escape OUR proposed bomb
    + ['OPP_NEAR_DEADEND']                                       # 33    selected opponent is standing in a dead end
    + ['SAFE_BOMB_HITS_OPPONENT']                                 # 34    bomb now hits >=1 opponent and we can escape
    + ['SAFE_BOMB_TRAPS_OPPONENT']                                # 35    bomb now traps the selected opponent and we can escape
    + ['SAFE_BOMB_HITS_CRATE']                                       # 36    bomb now hits >=1 crate and we can escape
    + [
    'ATTACK_DIR_UP',
    'ATTACK_DIR_RIGHT',
    'ATTACK_DIR_DOWN',
    'ATTACK_DIR_LEFT',
    'ATTACK_DIR_NONE',
    ]
    + ['ATTACK_DIST']
    + ['AT_SAFE_ATTACK_POS']                               
)
FEATURE_DIM = len(FEATURE_NAMES)  # 44

# Directional one-hot blocks, used by symmetry.py to know which slices to
# permute under rotation/flip. Each tuple = (start_index, has_none_slot).
# The first 4 entries starting at `start_index` are [UP, RIGHT, DOWN, LEFT];
# if has_none_slot, a 5th "NONE" entry follows and is left untouched.
# Appended opponent/bomb interaction features are scalars, so they need no
# directional-group entry.
DIRECTIONAL_GROUPS = [
    (0, False),   # WALL_*
    (7, True),    # COIN_DIR_*
    (13, True),   # CRATE_DIR_*
    (19, True),   # SAFE_DIR_*
    (25, True),   # OPP_DIR_*
    (37, True),   # ATTACK_DIR_*
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


def bfs_direction_and_target(game_state, targets, avoid_danger=False, danger=None, blocked=None):
    """
    Time-aware BFS from the agent to the nearest reachable target. Returns
    (first_step_direction, distance, target_coord). The returned target is the
    exact target that the direction/distance describe, preventing opponent
    features from accidentally referring to different opponents.
    """
    _, _, _, (sx, sy) = game_state['self']
    field = game_state['field']
    target_set = set(targets)
    blocked = set(blocked or ())

    def _unsafe_at(x, y, depth):
        return not _walkable(field, x, y) or (x, y) in blocked or (
            avoid_danger and danger is not None and danger[x, y] != 0 and danger[x, y] <= depth
        )

    if (sx, sy) in target_set:
        return None, 0, (sx, sy)

    visited = {(sx, sy)}
    queue = deque()
    for d in DIRECTIONS:
        dx, dy = DIRECTION_VECTORS[d]
        nx, ny = sx + dx, sy + dy
        if not _unsafe_at(nx, ny, 1):
            visited.add((nx, ny))
            queue.append((nx, ny, d, 1))

    while queue:
        x, y, first_dir, dist = queue.popleft()
        if (x, y) in target_set:
            return first_dir, dist, (x, y)
        for dx, dy in DIRECTION_VECTORS.values():
            nx, ny = x + dx, y + dy
            if (nx, ny) in visited or _unsafe_at(nx, ny, dist + 1):
                continue
            visited.add((nx, ny))
            queue.append((nx, ny, first_dir, dist + 1))

    return None, None, None


def bfs_direction(game_state, targets, avoid_danger=False, danger=None, blocked=None):
    """Backward-compatible two-value wrapper around bfs_direction_and_target()."""
    direction, distance, _ = bfs_direction_and_target(
        game_state, targets, avoid_danger=avoid_danger, danger=danger, blocked=blocked
    )
    return direction, distance


def _one_hot_direction(direction):
    """4-slot one-hot for UP/RIGHT/DOWN/LEFT plus a trailing NONE slot (5 total)."""
    vec = [0, 0, 0, 0, 0]
    if direction is None:
        vec[4] = 1
    else:
        vec[DIRECTIONS.index(direction)] = 1
    return vec


def crate_approach_targets(field):
    """
    Free tiles adjacent to at least one crate -- i.e. tiles you can stand on to
    bomb a crate. `bfs_direction` only reaches walkable tiles, so BFS-ing to
    crate coords directly never matches (crates are field==1, not walkable).
    Also imported by rewards.py so the definition lives in exactly one place.
    """
    w, h = field.shape
    targets = set()
    for cx, cy in np.argwhere(field == 1):
        for dx, dy in DIRECTION_VECTORS.values():
            nx, ny = int(cx + dx), int(cy + dy)
            if 0 <= nx < w and 0 <= ny < h and field[nx, ny] == 0:
                targets.add((nx, ny))
    return targets


# ---------------------------------------------------------------------------
# NEW: opponent trapped / dead-end detection, for Tasks 3-4 aggression.
# ---------------------------------------------------------------------------
def local_degree(field, x, y):
    """Number of walkable neighbors of a walkable tile (0-4). A tile with
    degree <=1 is a dead end (one way in, same way out)."""
    w, h = field.shape
    if not _walkable(field, x, y):
        return 0
    degree = 0
    for dx, dy in DIRECTION_VECTORS.values():
        nx, ny = x + dx, y + dy
        if _walkable(field, nx, ny):
            degree += 1
    return degree


def is_dead_end(field, x, y):
    """True if (x, y) is a walkable pocket with at most one walkable exit."""
    return _walkable(field, x, y) and local_degree(field, x, y) <= 1


def _escape_route_from(game_state, start_pos, danger=None, blocked=None, placement_turn=False):
    """Search positions and arrival times through all known explosion intervals.

    `danger` is retained for call compatibility; the full bomb schedule is used
    because a minimum-countdown map loses later blasts and fire expiration.
    """
    from .escape_planner import escape_route
    return escape_route(game_state, start_pos, blocked or (), placement_turn)


def select_opponent_target(game_state):
    """Return one consistent opponent target plus BFS direction/distance to it."""
    opp_coords = [pos for (_, _, _, pos) in game_state['others']]
    if not opp_coords:
        return None, None, None
    direction, distance, target = bfs_direction_and_target(game_state, opp_coords)
    if target is None:
        sx, sy = game_state['self'][3]
        target = min(opp_coords, key=lambda p: abs(sx - p[0]) + abs(sy - p[1]))
    return target, direction, distance


def proposed_bomb_analysis(game_state, target_opp=None):
    """
    Analyse the bomb the agent would ACTUALLY place now (at its own tile).
    Existing bombs are included in the danger schedule. The result is shared by
    Model-1 features and reward shaping so they cannot silently disagree.
    """
    field = game_state['field']
    _, _, bomb_possible, self_pos = game_state['self']
    opp_coords = [pos for (_, _, _, pos) in game_state['others']]
    if target_opp is None and opp_coords:
        target_opp, _, _ = select_opponent_target(game_state)

    blast = set(get_blast_coords(self_pos, field))
    hits_crate = any(field[x, y] == 1 for x, y in blast)
    hit_opponents = set(opp_coords) & blast

    result = {
        'can_drop': bool(bomb_possible),
        'blast': blast,
        'hits_crate': hits_crate,
        'hit_opponents': hit_opponents,
        'hits_opponent': bool(hit_opponents),
        'target': target_opp,
        'target_hit': target_opp in blast if target_opp is not None else False,
        'can_escape': False,
        'escape_dir': None,
        'escape_dist': None,
        'target_trapped': False,
    }
    if not bomb_possible:
        return result

    hypothetical = dict(game_state)
    hypothetical['bombs'] = list(game_state['bombs']) + [(self_pos, BOMB_TIMER)]
    bomb_positions = {pos for pos, _ in hypothetical['bombs']}
    opp_positions = set(opp_coords)

    can_escape, escape_dir, escape_dist = _escape_route_from(
        hypothetical, self_pos, blocked=bomb_positions | opp_positions,
        placement_turn=True
    )
    result['can_escape'] = can_escape
    result['escape_dir'] = escape_dir
    result['escape_dist'] = escape_dist

    if result['target_hit']:
        target_can_escape, _, _ = _escape_route_from(
            hypothetical, target_opp, blocked=bomb_positions
        )
        result['target_trapped'] = not target_can_escape

    return result

def safe_attack_positions(game_state, target_opp):
    """
    Return currently free tiles from which:

      1. a bomb would hit target_opp, and
      2. the agent would have an escape route after dropping it.

    This is deliberately small/cheap: only tiles within bomb range of the
    selected opponent are considered.
    """
    if target_opp is None:
        return set()

    field = game_state["field"]

    occupied = (
        {pos for pos, _ in game_state["bombs"]}
        | {pos for (_, _, _, pos) in game_state["others"]}
    )

    tx, ty = target_opp

    candidates = set()

    # Candidate bombing positions can only lie in the same row/column
    # within BOMB_POWER tiles.
    for dx, dy in DIRECTION_VECTORS.values():

        for distance in range(1, BOMB_POWER + 1):
            x = tx + dx * distance
            y = ty + dy * distance

            if not (
                0 <= x < field.shape[0]
                and 0 <= y < field.shape[1]
            ):
                break

            # Stone wall terminates blast visibility.
            if field[x, y] == -1:
                break

            # Must be somewhere the agent could stand.
            if field[x, y] != 0:
                continue

            pos = (x, y)

            if pos in occupied:
                continue

            # Defensive check using the actual framework blast geometry.
            if target_opp not in set(
                get_blast_coords(pos, field)
            ):
                continue

            # Pretend we were standing at this candidate and could bomb.
            name, score, _, _ = game_state["self"]

            hypothetical = dict(game_state)
            hypothetical["self"] = (
                name,
                score,
                True,
                pos,
            )

            info = proposed_bomb_analysis(
                hypothetical,
                target_opp=target_opp,
            )

            if (
                info["target_hit"]
                and info["can_escape"]
            ):
                candidates.add(pos)

    return candidates


def opponent_trapped(game_state, opp_pos):
    """True only if OUR proposed bomb hits this opponent and they cannot escape it."""
    return bool(proposed_bomb_analysis(game_state, target_opp=opp_pos)['target_trapped'])


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

    # --- WALL_* (0-3): blocked by terrain, a bomb, or another agent. ---
    occupied = {pos for pos, _ in game_state['bombs']} | {a[3] for a in game_state['others']}
    for i, d in enumerate(DIRECTIONS):
        dx, dy = DIRECTION_VECTORS[d]
        nx, ny = sx + dx, sy + dy
        w, h = field.shape
        blocked = not (0 <= nx < w and 0 <= ny < h) or field[nx, ny] != 0 or (nx, ny) in occupied
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

    # --- CRATE_DIR / CRATE_DIST (13-18): route to a tile you can bomb a crate from ---
    crate_targets = crate_approach_targets(field)
    crate_dir, crate_dist = bfs_direction(game_state, crate_targets) if crate_targets else (None, None)
    feats[13:18] = _one_hot_direction(crate_dir)
    feats[18] = 0.0 if crate_dist is None else min(crate_dist, MAX_DIST_NORM) / MAX_DIST_NORM

    # --- SAFE_DIR (19-23): nearest tile with danger==0, avoiding walk-through-fire ---
    if my_danger > 0:
        opp_positions = {pos for (_, _, _, pos) in game_state['others']}
        bomb_positions = {pos for pos, _ in game_state['bombs']}
        blocked = opp_positions | bomb_positions
        _, safe_dir, _ = _escape_route_from(game_state, (sx, sy), blocked=blocked)
    else:
        safe_dir = None
    feats[19:24] = _one_hot_direction(safe_dir)

    # --- Opponent target features (24-33): all describe ONE selected target. ---
    others = game_state['others']
    opp_coords = [pos for (_, _, _, pos) in others]
    bomb_info = proposed_bomb_analysis(game_state)
    if opp_coords:
        target_opp, opp_dir, opp_dist = select_opponent_target(game_state)
        if opp_dist is not None:
            nearby_dist = opp_dist
        else:
            nearby_dist = abs(sx - target_opp[0]) + abs(sy - target_opp[1])
        feats[24] = float(nearby_dist <= (2 * BOMB_POWER + 1))
        feats[25:30] = _one_hot_direction(opp_dir)
        feats[30] = 0.0 if opp_dist is None else min(opp_dist, MAX_DIST_NORM) / MAX_DIST_NORM
        feats[31] = float(bomb_info['hits_opponent'])
        feats[32] = float(bomb_info['target_trapped'])
        feats[33] = float(is_dead_end(field, *target_opp))
    else:
        feats[29] = 1.0  # OPP_DIR_NONE

    # --- Explicit Model-1 interaction features (34-36). ---
    # A linear Q function cannot create these AND conditions on its own.
    feats[34] = float(bomb_info['can_drop'] and bomb_info['can_escape'] and bomb_info['hits_opponent'])
    feats[35] = float(bomb_info['can_drop'] and bomb_info['can_escape'] and bomb_info['target_trapped'])
    feats[36] = float(bomb_info['can_drop'] and bomb_info['can_escape'] and bomb_info['hits_crate'])

    # -------------------------------------------------------------

    # Only pursue attack positions when:
    #   1. an opponent exists,
    #   2. we can currently place a bomb, and
    #   3. we are currently safe.
    #
    # Otherwise the attack features are disabled so that navigation,
    # coin collection and escape behaviour can dominate.
    
    attack_enabled = (
        bool(opp_coords)
        and bool(bomb_possible)
        and my_danger == 0
    )

    if attack_enabled:
        attack_tiles = safe_attack_positions(
            game_state,
            target_opp,
        )

        if attack_tiles:
            bomb_positions = {
                pos for pos, _ in game_state["bombs"]
            }

            opp_positions = {
                pos for (_, _, _, pos)
                in game_state["others"]
            }

            blocked = bomb_positions | opp_positions

            attack_dir, attack_dist, _ = (
                bfs_direction_and_target(
                    game_state,
                    attack_tiles,
                    avoid_danger=True,
                    danger=dmap,
                    blocked=blocked,
                )
            )

            feats[37:42] = _one_hot_direction(
                attack_dir
            )

            feats[42] = (
                0.0
                if attack_dist is None
                else min(
                    attack_dist,
                    MAX_DIST_NORM,
                ) / MAX_DIST_NORM
            )

            feats[43] = float(
                attack_dist == 0
                and bomb_possible
            )


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

    NOTE: unaffected by the OPP_TRAPPED/OPP_NEAR_DEADEND addition -- those
    are scalar features for Model 1 only. If Model 2 wants this signal too,
    it would need a new spatial channel; not added here since it wasn't
    requested and changing N_CHANNELS has the same "retrain from scratch"
    cost as changing FEATURE_DIM did -- flag to Person B before doing it.
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
