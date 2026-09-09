# bomberman_rl

Setup for a project/competition amongst students to train a winning Reinforcement Learning agent for the classic game Bomberman.

```python
# features.py  — Person A implements, you consume
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']   # fixed order, index = action id

def state_to_features(game_state: dict) -> np.ndarray
    # 1-D float array, fixed length D (~25–40). For linear + forest models.
    # Returns None only when game_state is None.

def state_to_channels(game_state: dict) -> np.ndarray
    # shape (C, 17, 17), C ~ 6–7. For the CNN. Channels:
    # walls, crates, coins, self, others, bomb-danger-map, explosion-map

def danger_map(game_state: dict) -> np.ndarray
    # (17,17) int: steps until this tile becomes lethal (0 = safe). Used by rewards too.

# symmetry (Person A) — you call these when augmenting a transition
def sym_transforms() -> list            # the 8 (rot × flip) ops
def apply_to_features(vec, op) -> np.ndarray
def apply_to_action(action_id, op) -> int

# evaluate.py (Person A) — you read its CSV
# columns: seed, round, score, coins, kills, self_kill(bool), steps_survived, invalid_actions
```
