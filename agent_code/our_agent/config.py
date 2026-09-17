"""
config.py

One place for every knob. train.py and callbacks.py both read MODEL and TRAIN.

Per-task tuning lives in PRESETS below; select one without editing the file:

    AGENT_PRESET=task2 uv run python main.py play --agents our_agent \
        --scenario loot-crate --train 1 --no-gui --n-rounds 3000

The chosen preset name is recorded in the training log so any run is reproducible.
"""

import os

# Select Model 1 (linear) or Model 2 (convolutional Double DQN).
MODEL = os.environ.get("AGENT_MODEL", "linear").strip()
if MODEL not in ("linear", "net"):
    raise ValueError("AGENT_MODEL must be linear or net")

# path (relative to the agent dir -- cwd is set there by the framework) where
# act() loads weights from when self.train is False.
WEIGHTS_FILE = {
    "linear": "weights/q_linear.pkl",
    "net":    "weights/q_net.pt",
    "forest": "weights/q_forest.pkl",
}

TRAIN = dict(
    # --- RL ---
    gamma=0.95,
    n_step=3,                # k in the k-step TD target; 1 = plain 1-step
    rule="q",                # "q" (Q-learning) or "sarsa"
    alpha=0.02,              # linear-model learning rate (start)
    alpha_end=0.005,         # linear-annealed to this over eps_decay_episodes; None = constant
    td_clip=8.0,             # clip |TD error| in the gradient step; None = off

    # --- exploration: linear anneal of epsilon, then hold ---
    eps_start=1.0,
    eps_end=0.01,            # low floor: coin-heaven needs little exploration once solved
    eps_decay_episodes=500,
    softmax_beta=None,       # set a float to use softmax instead of eps-greedy

    # --- replay ---
    buffer_capacity=100_000,
    batch_size=256,
    learn_every=4,           # learn once per this many env steps
    learn_iters_end=8,       # extra learn steps at end_of_round
    priority_alpha=0.0,      # 0 = uniform; try 0.5-0.7 once it trains

    # --- data augmentation ---
    use_symmetry=False,      # 8x transitions per step via symmetry.py (turn on later)

    # --- bookkeeping ---
    resume=False,            # True = keep training the existing checkpoint
    init_checkpoint=os.environ.get("AGENT_INIT_CHECKPOINT", "").strip(),
    seed=int(os.environ["AGENT_SEED"]) if "AGENT_SEED" in os.environ else None,
    save_every=25,           # checkpoint every N episodes (also always at the end)
    log_csv="training_log.csv",
    preset="task1",          # overwritten below when AGENT_PRESET is set
)

# ---------------------------------------------------------------------------
# Per-task overrides. The bare TRAIN dict above is the tuned Task 1 config.
# ---------------------------------------------------------------------------
PRESETS = {
    "task1": {},
    "task2": dict(               # loot-crate: bomb crates, find coins, never suicide
        gamma=0.97,             # longer horizon: bomb -> crate -> coin -> collect chains
        n_step=3,
        alpha=0.02, alpha_end=0.004,
        eps_end=0.03,           # more residual exploration: bombing must keep being tried
        eps_decay_episodes=1500,
        buffer_capacity=200_000,
        save_every=50,
        # 2026-09-17: tried n_step=5 + use_symmetry=True + doubled escape
        # penalties together -- official eval coins 28.1 -> 9.7, reverted.
    ),
    "task3": dict(               # classic vs peaceful_agent + coin_collector_agent
        gamma=0.97,
        n_step=4,               # approach -> corner -> bomb -> escape -> kill is a longer chain
        alpha=0.02, alpha_end=0.004,
        eps_end=0.05,           # opponents move -- keep more exploration than task2
        eps_decay_episodes=2000,
        buffer_capacity=200_000,
        save_every=50,
        # needs FEATURE_DIM=34 (OPP_TRAPPED/OPP_NEAR_DEADEND) -- first preset
        # that does. AGENT_INIT_CHECKPOINT explicitly permits padding old 32-dim weights.
        #
        # 2026-09-17: 4000 rounds -> eval coins 0.75/9, 8% kill rate, 3%
        # self-kill. Doubled to 8000 rounds (everything else identical):
        # coins 1.09/9, 6% kill rate, but self-kill worsened to 6% -- and
        # the training curve plateaued by ep~4000-5000.
    ),
}

_preset = os.environ.get("AGENT_PRESET", "").strip()
if _preset:
    if _preset not in PRESETS:
        raise KeyError(f"AGENT_PRESET={_preset!r} unknown; choose from {list(PRESETS)}")
    TRAIN.update(PRESETS[_preset])
    TRAIN["preset"] = _preset

# keep each task's training log + checkpoint separate so parallel runs don't clash
_run_name = os.environ.get("AGENT_RUN", TRAIN["preset"]).strip()
if not _run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in _run_name):
    raise ValueError("AGENT_RUN must contain only letters, digits, underscores or hyphens")
if MODEL == "net":
    # Spatial replay needs substantially more memory than flat features.
    TRAIN.update(alpha=1e-4, alpha_end=1e-4, batch_size=64,
                 buffer_capacity=10_000, target_update_every=500,
                 grad_clip=10.0, torch_threads=1,
                 device=os.environ.get("AGENT_DEVICE", "cpu"))
_extension = "pt" if MODEL == "net" else "pkl"
_log_prefix = "training_net" if MODEL == "net" else "training"
TRAIN["log_csv"] = f"{_log_prefix}_{_run_name}.csv"
TRAIN["weights_out"] = f"weights/q_{MODEL}_{_run_name}.{_extension}"
TRAIN["best_weights_out"] = f"weights/q_{MODEL}_{_run_name}_best.{_extension}"

# What callbacks.act() loads in eval / tournament mode: prefer the best
# checkpoint (a training run can get WORSE after a bad hyperparameter change
# -- this bit us once already, don't silently ship the latest instead of the
# best), then the latest, then the stable path (copy your final model there
# before submitting:  cp weights/q_linear_task4_best.pkl weights/q_linear.pkl).
EVAL_WEIGHTS = [TRAIN["best_weights_out"], TRAIN["weights_out"], WEIGHTS_FILE[MODEL]]
