"""
config.py  --  Yi Ling Chin

One place for every knob. train.py and callbacks.py both read MODEL and TRAIN.
For experiments, copy this file / override the dict from an env var or a JSON
in experiments/ rather than editing in place.
"""

# which model the agent runs. "linear" is the only one wired up so far;
# "net" / "forest" land later and read the same TRAIN block where it applies.
MODEL = "linear"

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
    alpha=0.05,              # linear-model learning rate

    # --- exploration: linear anneal of epsilon, then hold ---
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_episodes=400,
    softmax_beta=None,       # set a float to use softmax instead of eps-greedy

    # --- replay ---
    buffer_capacity=100_000,
    batch_size=128,
    learn_every=4,           # learn once per this many env steps
    learn_iters_end=8,       # extra learn steps at end_of_round
    priority_alpha=0.0,      # 0 = uniform; try 0.5-0.7 once it trains

    # --- data augmentation ---
    use_symmetry=False,      # 8x transitions per step via symmetry.py (turn on later)

    # --- bookkeeping ---
    resume=False,            # True = keep training the existing checkpoint
    save_every=25,           # checkpoint every N episodes (also always at the end)
    log_csv="training_log.csv",
)
