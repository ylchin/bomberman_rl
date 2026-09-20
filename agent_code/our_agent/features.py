"""
features.py

Turns the raw `game_state` dict (as documented in the assignment PDF, Sec. 5)
into inputs a model can consume. Two consumers:

  - state_to_features(game_state) -> flat vector       for Model 1 (linear/forest)
  - state_to_channels(game_state) -> (C, W, H) stack  for Model 2 (CNN / DQN)

Both are built on top of `danger_map`, which is also imported by Yi Ling Chin's
rewards.py (so bomb-danger logic lives in exactly one place).

ACTIONS ORDER IS LAW. Every index below (in ACTIONS, in one-hot blocks, in
symmetry.py) assumes this exact order. Do not reorder without updating
symmetry.py and every trained checkpoint.

Legacy 32/34-dim linear checkpoints can be used only as explicit initialization
checkpoints; q_linear.load pads the appended columns with zeros.
"""

from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# Fixed action order. index = action id, used everywhere (features, symmetry,
# model outputs). WAIT and BOMB are not directional and are never permuted
# by symmetry transforms.
# ---------------------------------------------------------------------------
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


DIRECTION_VECTORS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
}

DIRECTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]


BOMB_POWER = 3
BOMB_TIMER = 4
EXPLOSION_LINGER = 1

MAX_DIST_NORM = 20.0
MAX_DANGER_NORM = float(BOMB_TIMER + EXPLOSION_LINGER)


# ---------------------------------------------------------------------------
# Feature vector layout (D = 37).
# ---------------------------------------------------------------------------
FEATURE_NAMES = (
    ["WALL_UP", "WALL_RIGHT", "WALL_DOWN", "WALL_LEFT"]
    + ["BOMB_POSSIBLE"]
    + ["IN_DANGER"]
    + ["DANGER_STEPS"]
    + [
        "COIN_DIR_UP",
        "COIN_DIR_RIGHT",
        "COIN_DIR_DOWN",
        "COIN_DIR_LEFT",
        "COIN_DIR_NONE",
    ]
    + ["COIN_DIST"]
    + [
        "CRATE_DIR_UP",
        "CRATE_DIR_RIGHT",
        "CRATE_DIR_DOWN",
        "CRATE_DIR_LEFT",
        "CRATE_DIR_NONE",
    ]
    + ["CRATE_DIST"]
    + [
        "SAFE_DIR_UP",
        "SAFE_DIR_RIGHT",
        "SAFE_DIR_DOWN",
        "SAFE_DIR_LEFT",
        "SAFE_DIR_NONE",
    ]
    + ["OPP_NEARBY"]
    + [
        "OPP_DIR_UP",
        "OPP_DIR_RIGHT",
        "OPP_DIR_DOWN",
        "OPP_DIR_LEFT",
        "OPP_DIR_NONE",
    ]
    + ["OPP_DIST"]
    + ["WOULD_HIT_OPPONENT"]
    + ["OPP_TRAPPED"]
    + ["OPP_NEAR_DEADEND"]
    + ["SAFE_BOMB_HITS_OPPONENT"]
    + ["SAFE_BOMB_TRAPS_OPPONENT"]
    + ["SAFE_BOMB_HITS_CRATE"]
)

FEATURE_DIM = len(FEATURE_NAMES)  # 37


# Directional one-hot blocks used by symmetry.py.
DIRECTIONAL_GROUPS = [
    (0, False),  # WALL_*
    (7, True),  # COIN_DIR_*
    (13, True),  # CRATE_DIR_*
    (19, True),  # SAFE_DIR_*
    (25, True),  # OPP_DIR_*
]


# ---------------------------------------------------------------------------
# Model 2 channel layout.
# ---------------------------------------------------------------------------
CHANNEL_NAMES = [
    "walls",
    "crates",
    "coins",
    "self",
    "others",
    "bomb_danger",
    "explosion",
]

N_CHANNELS = len(CHANNEL_NAMES)  # 7


# ---------------------------------------------------------------------------
# Bomb blast geometry.
# ---------------------------------------------------------------------------
def get_blast_coords(bomb_xy, field):
    """
    Tiles covered by a bomb's explosion if it detonates at bomb_xy right now.

    Stops at stone walls (-1).
    """
    x, y = bomb_xy
    w, h = field.shape

    coords = [(x, y)]

    for dx, dy in DIRECTION_VECTORS.values():
        for i in range(1, BOMB_POWER + 1):

            nx = x + dx * i
            ny = y + dy * i

            if not (0 <= nx < w and 0 <= ny < h):
                break

            if field[nx, ny] == -1:
                break

            coords.append((nx, ny))

    return coords


# ---------------------------------------------------------------------------
# Danger map.
# ---------------------------------------------------------------------------
def danger_map(game_state):
    """
    (W, H) int array.

    Convention:

        0   -> currently safe
        N>0 -> becomes/remains lethal in N steps
    """
    field = game_state["field"]

    w, h = field.shape

    dmap = np.zeros(
        (w, h),
        dtype=np.int32,
    )

    for (bx, by), countdown in game_state["bombs"]:

        steps_to_blast = countdown + 1

        for x, y in get_blast_coords(
            (bx, by),
            field,
        ):

            if dmap[x, y] == 0 or steps_to_blast < dmap[x, y]:
                dmap[x, y] = steps_to_blast

    explosion_map = game_state.get("explosion_map")

    if explosion_map is not None:

        live = explosion_map > 0

        dmap[live] = np.where(
            (dmap[live] == 0) | (dmap[live] > 1),
            1,
            dmap[live],
        )

    return dmap


# ---------------------------------------------------------------------------
# Generic BFS helpers.
# ---------------------------------------------------------------------------
def _walkable(field, x, y):
    w, h = field.shape

    return 0 <= x < w and 0 <= y < h and field[x, y] == 0


def bfs_direction_and_target(
    game_state,
    targets,
    avoid_danger=False,
    danger=None,
    blocked=None,
):
    """
    BFS from the agent to the nearest reachable target.

    Returns:
        first_step_direction,
        distance,
        target_coord
    """
    _, _, _, (sx, sy) = game_state["self"]

    field = game_state["field"]

    target_set = set(targets)

    blocked = set(blocked or ())

    def _unsafe_at(x, y, depth):

        return (
            not _walkable(field, x, y)
            or (x, y) in blocked
            or (
                avoid_danger
                and danger is not None
                and danger[x, y] != 0
                and danger[x, y] <= depth
            )
        )

    if (sx, sy) in target_set:

        return (
            None,
            0,
            (sx, sy),
        )

    visited = {(sx, sy)}

    queue = deque()

    for direction in DIRECTIONS:

        dx, dy = DIRECTION_VECTORS[direction]

        nx = sx + dx
        ny = sy + dy

        if not _unsafe_at(
            nx,
            ny,
            1,
        ):

            visited.add((nx, ny))

            queue.append(
                (
                    nx,
                    ny,
                    direction,
                    1,
                )
            )

    while queue:

        (
            x,
            y,
            first_dir,
            dist,
        ) = queue.popleft()

        if (x, y) in target_set:

            return (
                first_dir,
                dist,
                (x, y),
            )

        for dx, dy in DIRECTION_VECTORS.values():

            nx = x + dx
            ny = y + dy

            if (nx, ny) in visited or _unsafe_at(
                nx,
                ny,
                dist + 1,
            ):
                continue

            visited.add((nx, ny))

            queue.append(
                (
                    nx,
                    ny,
                    first_dir,
                    dist + 1,
                )
            )

    return (
        None,
        None,
        None,
    )


def bfs_direction(
    game_state,
    targets,
    avoid_danger=False,
    danger=None,
    blocked=None,
):
    """
    Backwards-compatible wrapper returning only
    direction and distance.
    """

    direction, distance, _ = bfs_direction_and_target(
        game_state,
        targets,
        avoid_danger=avoid_danger,
        danger=danger,
        blocked=blocked,
    )

    return (
        direction,
        distance,
    )


def _one_hot_direction(direction):
    """
    UP / RIGHT / DOWN / LEFT / NONE.
    """

    vec = [
        0,
        0,
        0,
        0,
        0,
    ]

    if direction is None:

        vec[4] = 1

    else:

        vec[DIRECTIONS.index(direction)] = 1

    return vec


# ---------------------------------------------------------------------------
# Crate helpers.
# ---------------------------------------------------------------------------
def crate_approach_targets(field):
    """
    Free tiles adjacent to at least one crate.
    """

    w, h = field.shape

    targets = set()

    for cx, cy in np.argwhere(field == 1):

        for dx, dy in DIRECTION_VECTORS.values():

            nx = int(cx + dx)

            ny = int(cy + dy)

            if 0 <= nx < w and 0 <= ny < h and field[nx, ny] == 0:
                targets.add((nx, ny))

    return targets


# ---------------------------------------------------------------------------
# Opponent trapped / dead-end helpers.
# ---------------------------------------------------------------------------
def local_degree(field, x, y):
    """
    Number of walkable neighbours of a walkable tile.
    """

    if not _walkable(
        field,
        x,
        y,
    ):
        return 0

    degree = 0

    for dx, dy in DIRECTION_VECTORS.values():

        nx = x + dx
        ny = y + dy

        if _walkable(
            field,
            nx,
            ny,
        ):
            degree += 1

    return degree


def is_dead_end(field, x, y):
    """
    True when the tile has at most one walkable exit.
    """

    return (
        _walkable(
            field,
            x,
            y,
        )
        and local_degree(
            field,
            x,
            y,
        )
        <= 1
    )


# ---------------------------------------------------------------------------
# Full time-aware escape search.
# ---------------------------------------------------------------------------
def _escape_route_from(
    game_state,
    start_pos,
    danger=None,
    blocked=None,
    placement_turn=False,
):
    """
    Search positions and arrival times through
    all known explosion intervals.
    """

    from .escape_planner import escape_route

    return escape_route(
        game_state,
        start_pos,
        blocked or (),
        placement_turn,
    )


# ---------------------------------------------------------------------------
# Opponent target selection.
# ---------------------------------------------------------------------------
def select_opponent_target(game_state):
    """
    Return one consistent opponent target plus
    BFS direction and distance to it.
    """

    opp_coords = [
        pos
        for (
            _,
            _,
            _,
            pos,
        ) in game_state["others"]
    ]

    if not opp_coords:

        return (
            None,
            None,
            None,
        )

    (
        direction,
        distance,
        target,
    ) = bfs_direction_and_target(
        game_state,
        opp_coords,
    )

    if target is None:

        sx, sy = game_state["self"][3]

        target = min(
            opp_coords,
            key=lambda p: (abs(sx - p[0]) + abs(sy - p[1])),
        )

    return (
        target,
        direction,
        distance,
    )

#flat feature vector for Model 1 

# ---------------------------------------------------------------------------
# Hypothetical bomb analysis.
# ---------------------------------------------------------------------------
def proposed_bomb_analysis(
    game_state,
    target_opp=None,
):
    """
    Analyse the bomb the agent would ACTUALLY
    place at its current tile.
    """

    field = game_state["field"]

    (
        _,
        _,
        bomb_possible,
        self_pos,
    ) = game_state["self"]

    opp_coords = [
        pos
        for (
            _,
            _,
            _,
            pos,
        ) in game_state["others"]
    ]

    if target_opp is None and opp_coords:

        (
            target_opp,
            _,
            _,
        ) = select_opponent_target(game_state)

    blast = set(
        get_blast_coords(
            self_pos,
            field,
        )
    )

    hits_crate = any(field[x, y] == 1 for x, y in blast)

    hit_opponents = set(opp_coords) & blast

    result = {
        "can_drop": bool(bomb_possible),
        "blast": blast,
        "hits_crate": hits_crate,
        "hit_opponents": hit_opponents,
        "hits_opponent": bool(hit_opponents),
        "target": target_opp,
        "target_hit": (target_opp in blast if target_opp is not None else False),
        "can_escape": False,
        "escape_dir": None,
        "escape_dist": None,
        "target_trapped": False,
    }

    if not bomb_possible:

        return result

    hypothetical = dict(game_state)

    hypothetical["bombs"] = list(game_state["bombs"]) + [
        (
            self_pos,
            BOMB_TIMER,
        )
    ]

    bomb_positions = {pos for pos, _ in hypothetical["bombs"]}

    opp_positions = set(opp_coords)

    (
        can_escape,
        escape_dir,
        escape_dist,
    ) = _escape_route_from(
        hypothetical,
        self_pos,
        blocked=(bomb_positions | opp_positions),
        placement_turn=True,
    )

    result["can_escape"] = can_escape

    result["escape_dir"] = escape_dir

    result["escape_dist"] = escape_dist

    if result["target_hit"]:

        (
            target_can_escape,
            _,
            _,
        ) = _escape_route_from(
            hypothetical,
            target_opp,
            blocked=bomb_positions,
        )

        result["target_trapped"] = not target_can_escape

    return result


def opponent_trapped(
    game_state,
    opp_pos,
):
    """
    True only if OUR proposed bomb hits this
    opponent and they cannot escape it.
    """

    return bool(
        proposed_bomb_analysis(
            game_state,
            target_opp=opp_pos,
        )["target_trapped"]
    )


# ---------------------------------------------------------------------------
# Main entry point 1:
# flat feature vector for Model 1.
# ---------------------------------------------------------------------------
def state_to_features(game_state):
    """
    game_state -> np.ndarray of shape
    (FEATURE_DIM,), dtype float32.
    """

    if game_state is None:
        return None

    field = game_state["field"]

    (
        _,
        _,
        bomb_possible,
        (sx, sy),
    ) = game_state["self"]

    dmap = danger_map(game_state)

    feats = np.zeros(
        FEATURE_DIM,
        dtype=np.float32,
    )

    # ---------------------------------------------------------
    # WALL_* (0-3)
    # ---------------------------------------------------------
    occupied = {pos for pos, _ in game_state["bombs"]} | {
        a[3] for a in game_state["others"]
    }

    for i, direction in enumerate(DIRECTIONS):

        dx, dy = DIRECTION_VECTORS[direction]

        nx = sx + dx
        ny = sy + dy

        w, h = field.shape

        blocked = (
            not (0 <= nx < w and 0 <= ny < h)
            or field[
                nx,
                ny,
            ]
            != 0
            or (
                nx,
                ny,
            )
            in occupied
        )

        feats[i] = float(blocked)

    # ---------------------------------------------------------
    # BOMB_POSSIBLE (4)
    # ---------------------------------------------------------
    feats[4] = float(bomb_possible)

    # ---------------------------------------------------------
    # IN_DANGER / DANGER_STEPS (5-6)
    # ---------------------------------------------------------
    my_danger = dmap[
        sx,
        sy,
    ]

    feats[5] = float(my_danger > 0)

    feats[6] = (
        min(
            my_danger,
            MAX_DANGER_NORM,
        )
        / MAX_DANGER_NORM
    )

    # ---------------------------------------------------------
    # COIN_DIR / COIN_DIST (7-12)
    # ---------------------------------------------------------
    coins = game_state["coins"]

    if coins:

        (
            coin_dir,
            coin_dist,
        ) = bfs_direction(
            game_state,
            coins,
        )

    else:

        coin_dir = None
        coin_dist = None

    feats[7:12] = _one_hot_direction(coin_dir)

    feats[12] = (
        0.0
        if coin_dist is None
        else (
            min(
                coin_dist,
                MAX_DIST_NORM,
            )
            / MAX_DIST_NORM
        )
    )

    # ---------------------------------------------------------
    # CRATE_DIR / CRATE_DIST (13-18)
    # ---------------------------------------------------------
    crate_targets = crate_approach_targets(field)

    if crate_targets:

        (
            crate_dir,
            crate_dist,
        ) = bfs_direction(
            game_state,
            crate_targets,
        )

    else:

        crate_dir = None
        crate_dist = None

    feats[13:18] = _one_hot_direction(crate_dir)

    feats[18] = (
        0.0
        if crate_dist is None
        else (
            min(
                crate_dist,
                MAX_DIST_NORM,
            )
            / MAX_DIST_NORM
        )
    )

    # ---------------------------------------------------------
    # SAFE_DIR (19-23)
    # ---------------------------------------------------------
    if my_danger > 0:

        opp_positions = {
            pos
            for (
                _,
                _,
                _,
                pos,
            ) in game_state["others"]
        }

        bomb_positions = {pos for pos, _ in game_state["bombs"]}

        blocked = opp_positions | bomb_positions

        (
            _,
            safe_dir,
            _,
        ) = _escape_route_from(
            game_state,
            (sx, sy),
            blocked=blocked,
        )

    else:

        safe_dir = None

    feats[19:24] = _one_hot_direction(safe_dir)

    # ---------------------------------------------------------
    # Opponent target features (24-33)
    # ---------------------------------------------------------
    others = game_state["others"]

    opp_coords = [
        pos
        for (
            _,
            _,
            _,
            pos,
        ) in others
    ]

    bomb_info = proposed_bomb_analysis(game_state)

    if opp_coords:

        (
            target_opp,
            opp_dir,
            opp_dist,
        ) = select_opponent_target(game_state)

        if opp_dist is not None:

            nearby_dist = opp_dist

        else:

            nearby_dist = abs(sx - target_opp[0]) + abs(sy - target_opp[1])

        feats[24] = float(nearby_dist <= (2 * BOMB_POWER + 1))

        feats[25:30] = _one_hot_direction(opp_dir)

        feats[30] = (
            0.0
            if opp_dist is None
            else (
                min(
                    opp_dist,
                    MAX_DIST_NORM,
                )
                / MAX_DIST_NORM
            )
        )

        feats[31] = float(bomb_info["hits_opponent"])

        feats[32] = float(bomb_info["target_trapped"])

        feats[33] = float(
            is_dead_end(
                field,
                *target_opp,
            )
        )

    else:

        feats[29] = 1.0

    # ---------------------------------------------------------
    # Explicit Model-1 interaction features (34-36)
    # ---------------------------------------------------------

    # Bomb is safe and would hit an opponent.
    feats[34] = float(
        bomb_info["can_drop"] and bomb_info["can_escape"] and bomb_info["hits_opponent"]
    )

    # Bomb is safe and would trap selected opponent.
    feats[35] = float(
        bomb_info["can_drop"]
        and bomb_info["can_escape"]
        and bomb_info["target_trapped"]
    )

    # Bomb is safe and would hit at least one crate.
    feats[36] = float(
        bomb_info["can_drop"] and bomb_info["can_escape"] and bomb_info["hits_crate"]
    )

    return feats


# ---------------------------------------------------------------------------
# Main entry point 2:
# channel stack for Model 2.
# ---------------------------------------------------------------------------
def state_to_channels(game_state):
    """
    game_state -> np.ndarray of shape
    (N_CHANNELS, W, H), dtype float32.
    """

    if game_state is None:

        return None

    field = game_state["field"]

    w, h = field.shape

    channels = np.zeros(
        (
            N_CHANNELS,
            w,
            h,
        ),
        dtype=np.float32,
    )

    # walls
    channels[0] = (field == -1).astype(np.float32)

    # crates
    channels[1] = (field == 1).astype(np.float32)

    # coins
    for cx, cy in game_state["coins"]:

        channels[
            2,
            cx,
            cy,
        ] = 1.0

    # self
    (
        _,
        _,
        _,
        (sx, sy),
    ) = game_state["self"]

    channels[
        3,
        sx,
        sy,
    ] = 1.0

    # opponents
    for (
        _,
        _,
        _,
        (ox, oy),
    ) in game_state["others"]:

        channels[
            4,
            ox,
            oy,
        ] = 1.0

    # bomb danger
    dmap = danger_map(game_state)

    channels[5] = (
        np.minimum(
            dmap,
            MAX_DANGER_NORM,
        ).astype(np.float32)
        / MAX_DANGER_NORM
    )

    # active explosions
    explosion_map = game_state.get("explosion_map")

    if explosion_map is not None:

        channels[6] = np.clip(
            explosion_map,
            0,
            None,
        ).astype(np.float32)

        max_val = channels[6].max()

        if max_val > 0:

            channels[6] = channels[6] / max_val

    return channels
