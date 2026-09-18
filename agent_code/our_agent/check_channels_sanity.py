"""
check_channels_sanity.py

Sanity-checks features.state_to_channels() against a hand-built synthetic
game_state before Person B builds the CNN on top of it. Confirms:
  - output shape is (N_CHANNELS, board_width, board_height)
  - dtype is float32
  - each channel's values fall in the range that channel is documented to have
  - each channel actually lines up with what CHANNEL_NAMES says it is
    (walls channel really matches field==-1, coins channel really matches
    the coin list, etc.) -- catches silent transposition / off-by-one bugs
    that a shape-only check would miss.

Run from the repo root:
    python -m agent_code.our_agent.check_channels_sanity
"""

import numpy as np
from .features import state_to_channels, CHANNEL_NAMES, N_CHANNELS, danger_map


def build_synthetic_game_state(width=17, height=17):
    """
    A small hand-built game_state matching the schema in the assignment PDF
    Sec.5. Not a real game -- just enough structure (walls, a few crates,
    an active bomb, an explosion, coins, self, one opponent) to exercise
    every channel.
    """
    field = np.zeros((width, height), dtype=np.int8)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    crate_positions = [(3, 3), (3, 4), (5, 5)]
    for (x, y) in crate_positions:
        field[x, y] = 1

    self_pos = (2, 2)
    other_pos = (7, 7)
    coin_positions = [(4, 4), (9, 9)]
    bomb_pos = (10, 10)
    bomb_countdown = 2
    explosion_pos = (11, 11)

    explosion_map = np.zeros((width, height), dtype=np.int32)
    explosion_map[explosion_pos] = 1  # currently on fire

    game_state = {
        'round': 1,
        'step': 1,
        'field': field,
        'bombs': [(bomb_pos, bomb_countdown)],
        'explosion_map': explosion_map,
        'coins': coin_positions,
        'self': ('our_agent', 0, True, self_pos),
        'others': [('opponent_0', 0, True, other_pos)],
        'user_input': None,
    }
    return game_state, {
        'field': field, 'self_pos': self_pos, 'other_pos': other_pos,
        'coin_positions': coin_positions, 'bomb_pos': bomb_pos,
        'explosion_pos': explosion_pos,
    }


def check_shape_and_dtype(channels, field):
    w, h = field.shape
    ok = True
    if channels.shape != (N_CHANNELS, w, h):
        print(f"FAIL: shape is {channels.shape}, expected {(N_CHANNELS, w, h)}")
        ok = False
    if channels.dtype != np.float32:
        print(f"FAIL: dtype is {channels.dtype}, expected float32")
        ok = False
    if ok:
        print(f"OK: shape {channels.shape}, dtype {channels.dtype}")
    return ok


def check_value_ranges(channels):
    ok = True
    binary_channels = {'walls', 'crates', 'coins', 'self', 'others'}
    for i, name in enumerate(CHANNEL_NAMES):
        ch = channels[i]
        lo, hi = ch.min(), ch.max()
        if name in binary_channels:
            bad_vals = ch[(ch != 0) & (ch != 1)]
            if bad_vals.size > 0:
                print(f"FAIL: channel '{name}' (binary) has non-0/1 values, "
                      f"e.g. {bad_vals[0]}")
                ok = False
        else:  # bomb_danger, explosion -- normalized, expect [0, 1]
            if lo < 0.0 or hi > 1.0 + 1e-6:
                print(f"FAIL: channel '{name}' out of [0,1] range: min={lo}, max={hi}")
                ok = False
    if ok:
        print("OK: all channel value ranges are within spec.")
    return ok


def check_channel_alignment(channels, ref):
    """Confirms each channel actually matches the ground truth we built,
    at the ground-truth coordinates -- not just 'has some 1s somewhere'."""
    ok = True
    idx = {name: i for i, name in enumerate(CHANNEL_NAMES)}

    walls_expected = (ref['field'] == -1)
    if not np.array_equal(channels[idx['walls']] > 0, walls_expected):
        print("FAIL: 'walls' channel does not match field==-1 positions")
        ok = False

    crates_expected = (ref['field'] == 1)
    if not np.array_equal(channels[idx['crates']] > 0, crates_expected):
        print("FAIL: 'crates' channel does not match field==1 positions")
        ok = False

    for (cx, cy) in ref['coin_positions']:
        if channels[idx['coins'], cx, cy] != 1.0:
            print(f"FAIL: 'coins' channel missing a 1 at coin position {(cx, cy)}")
            ok = False
    if channels[idx['coins']].sum() != len(ref['coin_positions']):
        print(f"FAIL: 'coins' channel has {channels[idx['coins']].sum():.0f} set pixels, "
              f"expected exactly {len(ref['coin_positions'])}")
        ok = False

    sx, sy = ref['self_pos']
    if channels[idx['self'], sx, sy] != 1.0 or channels[idx['self']].sum() != 1.0:
        print(f"FAIL: 'self' channel should have exactly one 1, at {ref['self_pos']}")
        ok = False

    ox, oy = ref['other_pos']
    if channels[idx['others'], ox, oy] != 1.0 or channels[idx['others']].sum() != 1.0:
        print(f"FAIL: 'others' channel should have exactly one 1, at {ref['other_pos']}")
        ok = False

    bx, by = ref['bomb_pos']
    if channels[idx['bomb_danger'], bx, by] <= 0:
        print(f"FAIL: 'bomb_danger' channel is 0 at the active bomb's own tile {ref['bomb_pos']}")
        ok = False

    ex, ey = ref['explosion_pos']
    if channels[idx['explosion'], ex, ey] <= 0:
        print(f"FAIL: 'explosion' channel is 0 at the currently-on-fire tile {ref['explosion_pos']}")
        ok = False

    if ok:
        print("OK: every channel's active pixels line up with the ground-truth positions.")
    return ok


if __name__ == "__main__":
    game_state, ref = build_synthetic_game_state()
    channels = state_to_channels(game_state)

    ok1 = check_shape_and_dtype(channels, ref['field'])
    ok2 = check_value_ranges(channels)
    ok3 = check_channel_alignment(channels, ref)

    if ok1 and ok2 and ok3:
        print("\nAll checks passed -- state_to_channels() is safe to build the CNN against.")
    else:
        print("\nDO NOT build the CNN against this yet -- fix the issues above first.")
        raise SystemExit(1)
