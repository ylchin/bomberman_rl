"""
symmetry.py 

The Bomberman board has 8 symmetries (4 rotations x optional mirror = the
dihedral group D4). Any transition (state, action, reward, next_state) can be
turned into 8 equally-valid training examples by applying the same transform
to the state/features AND to the action label consistently.

Two ways to use this, both supported:

  1. RECOMMENDED — transform the raw game_state coordinates *before* running
     features.state_to_features / state_to_channels on it. This is the most
     robust option since it can never get out of sync with feature layout
     changes: apply_to_state() + re-extract features.

  2. FASTER — transform an already-computed feature vector directly via
     apply_to_features(), using the DIRECTIONAL_GROUPS metadata exported by
     features.py. Only permutes the directional one-hot blocks; use this once
     features.py is stable and you want to avoid recomputing BFS 8x per
     transition.

Either way, apply_to_action() must be used to transform the label action to
match, or your model will learn "go right" for boards that were flipped to
actually require "go left".
"""

import numpy as np
from features import ACTIONS, DIRECTIONS, DIRECTION_VECTORS, DIRECTIONAL_GROUPS

# The 8 ops, as (flip_first: bool, n_rotations_cw: int).
# flip = mirror across the vertical axis (x -> W-1-x) applied BEFORE rotating.
def sym_transforms():
    """Returns the list of 8 op identifiers. Pass these to the apply_* functions."""
    return [(flip, k) for flip in (False, True) for k in range(4)]


# --- linear part, shared by coordinate and direction-vector transforms -----
def _flip_vec(dx, dy):
    return (-dx, dy)

def _rot90_vec(dx, dy):
    return (dy, -dx)

def _transform_vec(dx, dy, op):
    flip, k = op
    if flip:
        dx, dy = _flip_vec(dx, dy)
    for _ in range(k):
        dx, dy = _rot90_vec(dx, dy)
    return dx, dy


def apply_to_coord(x, y, op, width, height):
    """
    Transform a single board coordinate under op. width/height are the
    field's dimensions (board is square in this game, but kept general).
    """
    flip, k = op
    w, h = width, height
    if flip:
        x = w - 1 - x
    for _ in range(k):
        x, y, w, h = y, w - 1 - x, h, w  # rotate 90 CW, swapping w/h each time
    return x, y


def apply_to_action(action_id_or_name, op):
    """
    Transform an action under op. Non-directional actions (WAIT, BOMB) are
    unchanged. Accepts either an int id (index into ACTIONS) or the action
    name string; returns the same type it was given.
    """
    is_name = isinstance(action_id_or_name, str)
    name = action_id_or_name if is_name else ACTIONS[action_id_or_name]

    if name not in DIRECTIONS:
        return action_id_or_name  # WAIT / BOMB pass through unchanged

    dx, dy = DIRECTION_VECTORS[name]
    ndx, ndy = _transform_vec(dx, dy, op)
    new_name = next(d for d in DIRECTIONS if DIRECTION_VECTORS[d] == (ndx, ndy))

    return new_name if is_name else ACTIONS.index(new_name)


def apply_to_features(vec, op):
    """
    Permute a state_to_features() vector under op. Only touches the
    directional one-hot blocks listed in features.DIRECTIONAL_GROUPS;
    everything else (scalars, distances) is copied unchanged, since those
    are rotation/flip-invariant magnitudes.
    """
    out = np.array(vec, dtype=vec.dtype if hasattr(vec, 'dtype') else np.float32)

    # Build the direction permutation implied by op once, reuse for every group.
    perm = [DIRECTIONS.index(apply_to_action(d, op)) for d in DIRECTIONS]

    for start, has_none in DIRECTIONAL_GROUPS:
        block = vec[start:start + 4]
        new_block = [0] * 4
        for old_i, new_i in enumerate(perm):
            new_block[new_i] = block[old_i]
        out[start:start + 4] = new_block
        # NONE slot (start+4), if present, is unaffected by rotation/flip.
        if has_none:
            out[start + 4] = vec[start + 4]

    return out


def apply_to_state(game_state, op):
    """
    Transform an entire raw game_state dict under op — the robust path.
    Returns a NEW dict; does not mutate the input. Recompute features from
    this via features.state_to_features(transformed_state) /
    state_to_channels(transformed_state).

    Note: 'round', 'step', 'user_input' pass through unchanged; only spatial
    fields are transformed.
    """
    field = game_state['field']
    w, h = field.shape

    new_field = np.zeros_like(field)
    for x in range(w):
        for y in range(h):
            nx, ny = apply_to_coord(x, y, op, w, h)
            new_field[nx, ny] = field[x, y]

    new_explosion_map = None
    if game_state.get('explosion_map') is not None:
        em = game_state['explosion_map']
        new_explosion_map = np.zeros_like(em)
        for x in range(w):
            for y in range(h):
                nx, ny = apply_to_coord(x, y, op, w, h)
                new_explosion_map[nx, ny] = em[x, y]

    def transform_agent_tuple(agent):
        name, score, can_bomb, (x, y) = agent
        return (name, score, can_bomb, apply_to_coord(x, y, op, w, h))

    new_state = dict(game_state)  # shallow copy; overwrite spatial fields below
    new_state['field'] = new_field
    new_state['bombs'] = [(apply_to_coord(x, y, op, w, h), t) for (x, y), t in game_state['bombs']]
    new_state['explosion_map'] = new_explosion_map
    new_state['coins'] = [apply_to_coord(x, y, op, w, h) for (x, y) in game_state['coins']]
    new_state['self'] = transform_agent_tuple(game_state['self'])
    new_state['others'] = [transform_agent_tuple(a) for a in game_state['others']]

    return new_state


def augment_transition(state, action, reward, next_state, use_state_transform=True,
                        feature_fn=None):
    """
    Convenience wrapper: given one (state, action, reward, next_state)
    transition, returns a list of 8 augmented transitions.

    If use_state_transform=True (recommended), `state`/`next_state` are raw
    game_state dicts and `feature_fn` (e.g. features.state_to_features) is
    called on each transformed state to produce the final training input.

    If use_state_transform=False, `state`/`next_state` are already feature
    vectors and apply_to_features() is used directly instead (faster, but
    only valid for the flat-vector case, not channel stacks — channel stacks
    should always go through apply_to_state + state_to_channels).
    """
    out = []
    for op in sym_transforms():
        new_action = apply_to_action(action, op)
        if use_state_transform:
            new_state = feature_fn(apply_to_state(state, op)) if state is not None else None
            new_next_state = feature_fn(apply_to_state(next_state, op)) if next_state is not None else None
        else:
            new_state = apply_to_features(state, op) if state is not None else None
            new_next_state = apply_to_features(next_state, op) if next_state is not None else None
        out.append((new_state, new_action, reward, new_next_state))
    return out
