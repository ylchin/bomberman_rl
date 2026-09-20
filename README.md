# bomberman_rl

Setup for a project/competition amongst students to train a winning Reinforcement Learning agent for the classic game Bomberman.

## Model 2: CNN / Double DQN

Select `AGENT_MODEL=net` to use Model 2. The default remains `linear`.
From the repository root in PowerShell:

```powershell
$env:AGENT_MODEL = 'net'
$env:AGENT_PRESET = 'task1'
$env:AGENT_RUN = 'cnn_task1_seed1'
$env:AGENT_SEED = '1'
.venv/Scripts/python.exe main.py play --agents our_agent --scenario coin-heaven --train 1 --no-gui --n-rounds 3000 --seed 1
.venv/Scripts/python.exe evaluate.py --agent our_agent --task 1 --n-rounds 100 --out experiments/cnn_task1_seed1_eval.csv
```

Keep the same environment variables for evaluation. Checkpoints are saved to
`agent_code/our_agent/weights/q_net_<AGENT_RUN>.pt`; logs use
`training_net_<AGENT_RUN>.csv`. Evaluation prefers the best checkpoint, then
latest, then `weights/q_net.pt`. Best selection uses trailing 100-round training
coins, so evaluate both candidates before selecting a competitive model.
Use a distinct run name for each fresh experiment.

The network consumes the existing seven spatial channels and produces six action
values. Training uses n-step Double DQN targets, a frozen target network synced
every 500 optimizer updates, Adam, Huber loss, gradient clipping, and the shared
reward and replay pipeline. Spatial symmetry augmentation is supported via
`TRAIN['use_symmetry']`. Replay holds 10,000 transitions (about 162 MB for the two
float32 state arrays); batch size is 64. Settings are in
`agent_code/our_agent/config.py`. CPU with one PyTorch thread is the default;
set `AGENT_DEVICE=cuda` only with a CUDA-capable PyTorch installation.

For Task 2 use `AGENT_PRESET=task2` and `--scenario loot-crate`; for Task 3 use
`AGENT_PRESET=task3`, `--scenario classic`, and
`--agents our_agent peaceful_agent coin_collector_agent`.
To transfer a CNN checkpoint, set `AGENT_INIT_CHECKPOINT` to its path relative to
`agent_code/our_agent` and choose a new `AGENT_RUN`. This transfers online weights
and restarts the optimizer, target synchronization counter, replay, and schedules.
Linear checkpoints cannot initialize a CNN. The existing `TRAIN['resume']` option
also restores CNN optimizer/target state, but episode schedules and replay still
restart; it is not an exact interrupted-run continuation.

Run checks with `.venv/Scripts/python.exe -m unittest discover -s tests -v`.
The implementation has smoke-test coverage; training quality needs a full run
and held-out evaluation.

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
