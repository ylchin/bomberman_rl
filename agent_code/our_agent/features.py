"""
turns the raw `game_state` dict into inputs a model can consume. 

two consumers:
- state_to_features(game_state) -> flat vector (for model 1)
- state_to_channels(game_state) -> (C, W, H) stack (for model 2)
"""

from collections import deque
import numpy as np


#fixed action order. index = action id, used everywhere (features, symmetry, model outputs)
#WAIT and BOMB are not directional and are never permuted by symmetry transforms

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

DIRECTION_VECTORS = {
    'UP':    (0, -1),
    'RIGHT': (1, 0),
    'DOWN':  (0, 1),
    'LEFT':  (-1, 0),
}
DIRECTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']  #4 directional actions in order

BOMB_POWER = 3  #blast radius in tiles, each cardinal direction 
BOMB_TIMER = 4  #steps from drop to detonation (countdown starts at BOMB_TIMER - 1)
EXPLOSION_LINGER = 1  #extra steps the fire remains lethal after detonating

MAX_DIST_NORM = 20.0  #cap used to normalize BFS distances into [0, 1]
MAX_DANGER_NORM = float(BOMB_TIMER + EXPLOSION_LINGER)  #cap for danger-step normalization


# Feature vector layout (D = 34)

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
    + ['OPP_TRAPPED']                                            # 32    nearest opp has no bomb-escape route
    + ['OPP_NEAR_DEADEND']                                       # 33    nearest opp is standing in a dead end
)
FEATURE_DIM = len(FEATURE_NAMES)  # 34

#directional one-hot blocks, used by symmetry.py to know which slices to permute under rotation/flip
#each tuple = (start_index, has_none_slot)
#first 4 entries starting at `start_index` are [UP, RIGHT, DOWN, LEFT]
#if has_none_slot, a 5th "NONE" entry follows and is left untouched

DIRECTIONAL_GROUPS = [
    (0, False),   # WALL_*
    (7, True),    # COIN_DIR_*
    (13, True),   # CRATE_DIR_*
    (19, True),   # SAFE_DIR_*
    (25, True),   # OPP_DIR_*
]

#channel stack layout for state_to_channels — order is fixed
CHANNEL_NAMES = ['walls', 'crates', 'coins', 'self', 'others', 'bomb_danger', 'explosion']
N_CHANNELS = len(CHANNEL_NAMES)  # 7


# Bomb blast geometry 
def get_blast_coords(bomb_xy, field):
    #tiles covered by a bomb's explosion if it detonates at bomb_xy right now
    #stops at stone walls (-1). passes through (and destroys) crates
    #does not wrap around corners (cardinal directions only)
  
    x, y = bomb_xy
    w, h = field.shape
    coords = [(x, y)]
    for dx, dy in DIRECTION_VECTORS.values():
        for i in range(1, BOMB_POWER + 1):
            nx, ny = x + dx * i, y + dy * i
            if not (0 <= nx < w and 0 <= ny < h):
                break
            if field[nx, ny] == -1:  #stone wall stops the blast
                break
            coords.append((nx, ny))
    return coords


def danger_map(game_state):
    #(W, H) int array. convention:
    #0 -> tile is safe (not covered by any current bomb/explosion)
    #N>0 -> tile becomes/remains lethal in N steps from no2 (N=1 means lethal on the very next step, move now)
    #combines: each active bomb's blast coverage, valued at (countdown + 1)
    # & currently active explosions (game_state['explosion_map']), valued at 1
    #takes the min across sources per tile (soonest danger wins)
    
    field = game_state['field']
    w, h = field.shape
    dmap = np.zeros((w, h), dtype=np.int32)

    for (bx, by), countdown in game_state['bombs']:
        steps_to_blast = countdown + 1  #countdown==0 means "explodes next step"
        for (x, y) in get_blast_coords((bx, by), field):
            if dmap[x, y] == 0 or steps_to_blast < dmap[x, y]:
                dmap[x, y] = steps_to_blast

    explosion_map = game_state.get('explosion_map')
    if explosion_map is not None:
        live = explosion_map > 0
        # currently on fire = lethal right now (1 step)
        dmap[live] = np.where((dmap[live] == 0) | (dmap[live] > 1), 1, dmap[live])

    return dmap


# BFS pathfinding — generic shortest-path-to-nearest-target on the free-tile graph. returns (direction_name_or_None, distance_or_None).

def _walkable(field, x, y):
    w, h = field.shape
    return 0 <= x < w and 0 <= y < h and field[x, y] == 0


def bfs_direction(game_state, targets, avoid_danger=False, danger=None):
    #BFS from the agent's position to the nearest tile in targets (a set/list of (x, y) coords)
    #returns (first_step_direction, distance)
    #if no target is reachable, returns (None, None)

    #If avoid_danger=True, tiles with danger[x,y] == 1 (lethal next step) are treated as walls — 
    #used for SAFE_DIR routing so we don't path through something about to explode
    #pass a precomputed danger map to avoid recomputing it multiple times per act() call.
    
    _, _, _, (sx, sy) = game_state['self']
    field = game_state['field']
    target_set = set(targets)
    if (sx, sy) in target_set:
        return None, 0  

    visited = {(sx, sy)}
    #queue entries: (x, y, first_step_direction)
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
    #4-slot one-hot for UP/RIGHT/DOWN/LEFT plus a trailing NONE slot (5 total)
    vec = [0, 0, 0, 0, 0]
    if direction is None:
        vec[4] = 1
    else:
        vec[DIRECTIONS.index(direction)] = 1
    return vec


def crate_approach_targets(field):
    #free tiles adjacent to at least one crate - tiles you can stand on to bomb a crate 
    #bfs_direction only reaches walkable tiles, so BFS-ing to crate coords directly never matches (crates are field==1, not walkable)
    
    w, h = field.shape
    targets = set()
    for cx, cy in np.argwhere(field == 1):
        for dx, dy in DIRECTION_VECTORS.values():
            nx, ny = int(cx + dx), int(cy + dy)
            if 0 <= nx < w and 0 <= ny < h and field[nx, ny] == 0:
                targets.add((nx, ny))
    return targets


#opponent trapped / dead-end detection

def local_degree(field, x, y):
    #number of walkable neighbors of a walkable tile (0-4). tile with degree <=1 is a dead end (one way in, same way out)
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
    #True if (x, y) is a walkable pocket with at most one walkable exit
    return _walkable(field, x, y) and local_degree(field, x, y) <= 1


def opponent_trapped(field, opp_pos, bomb_timer=BOMB_TIMER):
    #True if assuming a bomb detonates at the opponent's current tile (worst-case: they've just been cornered and bombed there), 
    #the opponent cannot reach any tile outside that blast within bomb_timer steps

    blast = set(get_blast_coords(opp_pos, field))
    visited = {opp_pos: 0}
    queue = deque([opp_pos])

    while queue:
        (x, y) = queue.popleft()
        dist = visited[(x, y)]
        if (x, y) not in blast:
            return False  #found a safe tile reachable in time -> not trapped
        if dist >= bomb_timer:
            continue  # out of time to search further from here
        for dx, dy in DIRECTION_VECTORS.values():
            nx, ny = x + dx, y + dy
            if _walkable(field, nx, ny) and (nx, ny) not in visited:
                visited[(nx, ny)] = dist + 1
                queue.append((nx, ny))

    return True  #exhausted every reachable tile within the time window, all lethal


#flat feature vector for Model 1 

def state_to_features(game_state):
    #game_state -> np.ndarray of shape (FEATURE_DIM,), dtype float32
    #returns None only if game_state is None (e.g. terminal call pattern some training loops use) — callers must handle that case explicitly
    
    if game_state is None:
        return None

    field = game_state['field']
    _, _, bomb_possible, (sx, sy) = game_state['self']
    dmap = danger_map(game_state)

    feats = np.zeros(FEATURE_DIM, dtype=np.float32)

    # WALL_* (0-3): is the neighbor tile blocked (wall or crate)
    for i, d in enumerate(DIRECTIONS):
        dx, dy = DIRECTION_VECTORS[d]
        nx, ny = sx + dx, sy + dy
        w, h = field.shape
        blocked = not (0 <= nx < w and 0 <= ny < h) or field[nx, ny] != 0
        feats[i] = float(blocked)

    # BOMB_POSSIBLE (4)
    feats[4] = float(bomb_possible)

    # IN_DANGER / DANGER_STEPS (5-6)
    my_danger = dmap[sx, sy]
    feats[5] = float(my_danger > 0)
    feats[6] = min(my_danger, MAX_DANGER_NORM) / MAX_DANGER_NORM

    # COIN_DIR / COIN_DIST (7-12)
    coins = game_state['coins']
    coin_dir, coin_dist = bfs_direction(game_state, coins) if coins else (None, None)
    feats[7:12] = _one_hot_direction(coin_dir)
    feats[12] = 0.0 if coin_dist is None else min(coin_dist, MAX_DIST_NORM) / MAX_DIST_NORM

    # CRATE_DIR / CRATE_DIST (13-18): route to a tile you can bomb a crate from
    crate_targets = crate_approach_targets(field)
    crate_dir, crate_dist = bfs_direction(game_state, crate_targets) if crate_targets else (None, None)
    feats[13:18] = _one_hot_direction(crate_dir)
    feats[18] = 0.0 if crate_dist is None else min(crate_dist, MAX_DIST_NORM) / MAX_DIST_NORM

    # SAFE_DIR (19-23): nearest tile with danger==0, avoiding walk-through-fire
    if my_danger > 0:
        w, h = field.shape
        safe_tiles = [(x, y) for x in range(w) for y in range(h)
                      if field[x, y] == 0 and dmap[x, y] == 0]
        safe_dir, _ = bfs_direction(game_state, safe_tiles, avoid_danger=True, danger=dmap)
    else:
        safe_dir = None
    feats[19:24] = _one_hot_direction(safe_dir)

    # OPP_NEARBY / OPP_DIR / OPP_DIST / WOULD_HIT_OPPONENT (24-31)
    # OPP_TRAPPED / OPP_NEAR_DEADEND (32-33)
    others = game_state['others']
    opp_coords = [pos for (_, _, _, pos) in others]
    if opp_coords:
        opp_dir, opp_dist = bfs_direction(game_state, opp_coords)
        #nearest opponent by Manhattan distance, also used as the target
        #for the two new trapped/dead-end features below, so they describe the SAME opponent that OPP_DIR/OPP_DIST point at
        nearest_opp = min(opp_coords, key=lambda p: abs(sx - p[0]) + abs(sy - p[1]))
        manhattan_min = abs(sx - nearest_opp[0]) + abs(sy - nearest_opp[1])
        feats[24] = float(manhattan_min <= (2 * BOMB_POWER + 1))
        feats[25:30] = _one_hot_direction(opp_dir)
        feats[30] = 0.0 if opp_dist is None else min(opp_dist, MAX_DIST_NORM) / MAX_DIST_NORM
        blast_if_bombed_here = set(get_blast_coords((sx, sy), field))
        feats[31] = float(bool(blast_if_bombed_here & set(opp_coords)))

        feats[32] = float(opponent_trapped(field, nearest_opp))
        feats[33] = float(is_dead_end(field, *nearest_opp))
    #else: leave OPP_* and OPP_TRAPPED/OPP_NEAR_DEADEND as zeros / NONE-implicit

    return feats


#channel stack for Model 2
def state_to_channels(game_state):
    #game_state -> np.ndarray of shape (N_CHANNELS, W, H), dtype float32
    #channel order (see CHANNEL_NAMES): walls, crates, coins, self, others, bomb_danger (normalised), explosion (normalised)
    #returns None only if game_state is None
    
    if game_state is None:
        return None

    field = game_state['field']
    w, h = field.shape
    channels = np.zeros((N_CHANNELS, w, h), dtype=np.float32)

    channels[0] = (field == -1).astype(np.float32) #walls
    channels[1] = (field == 1).astype(np.float32) #crates

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
