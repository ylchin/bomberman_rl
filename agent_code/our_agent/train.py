"""
train.py  --  our_agent  (Yi Ling Chin)

Imported by the framework only when --train is set. Owns:

    setup_training(self)                              once, after setup()
    game_events_occurred(self, old, action, new, ev)  after every step but the last
    end_of_round(self, last, last_action, ev)         once, after the last step

Learning: transitions are collected per episode, turned into k-step TD targets
at end_of_round, and pushed to the replay buffer. A batch update runs every
`learn_every` steps (on data from previous episodes) plus a few extra at
end_of_round. See config.TRAIN for every knob.
"""

import csv
import os
import shutil
from collections import Counter, deque

import numpy as np

import events as e
from .features import ACTIONS, FEATURE_DIM, state_to_features
from .replay_buffer import ReplayBuffer
from .rewards import reward_from_events, detect_custom_events, potential_shaping
from .config import TRAIN, MODEL

_ACTION_ID = {name: i for i, name in enumerate(ACTIONS)}


# ---------------------------------------------------------------------------
def setup_training(self):
    cfg = TRAIN
    self.t = cfg  # shorthand

    if MODEL == "net":
        from .model_net import INPUT_DIM, encode_state, transform_input

        self.feature_dim = INPUT_DIM
        self.encode_state = encode_state
        self.transform_input = transform_input
    else:
        from .symmetry import apply_to_features

        self.feature_dim = FEATURE_DIM
        self.encode_state = state_to_features
        self.transform_input = apply_to_features

    self.buffer = ReplayBuffer(
        capacity=cfg["buffer_capacity"],
        feature_dim=self.feature_dim,
        rng=self.rng,
        alpha=cfg["priority_alpha"],
    )

    self.episode = 0
    self.step_count = 0  # global env steps, for learn_every
    self.traj = []  # [(phi, action_id, reward), ...] for the current episode
    self.ep_counts = Counter()  # event tally for the CSV row
    self.ep_reward = 0.0

    self.epsilon = _epsilon(self, 0)
    self.alpha = _alpha(self, 0)

    os.makedirs("weights", exist_ok=True)
    self._weights_path = cfg["weights_out"]
    self._best_weights_path = cfg["best_weights_out"]
    self._best_checkpoint_key = None
    self._recent_metrics = deque(
        maxlen=100
    )  # task-aware rolling proxy; final choice uses validation
    self._csv_path = cfg["log_csv"]
    _csv_header(self._csv_path)

    try:
        from . import symmetry as _sym

        self._sym = _sym
    except Exception as exc:  # pragma: no cover
        self.logger.warning(f"symmetry.py unavailable ({exc}); augmentation off")
        self._sym = None

    self.logger.info(
        f"training: preset={cfg.get('preset')} model={MODEL} rule={cfg['rule']} "
        f"k={cfg['n_step']} gamma={cfg['gamma']} alpha={cfg['alpha']}->{cfg.get('alpha_end')} "
        f"eps_end={cfg['eps_end']} buffer={cfg['buffer_capacity']}"
    )


# ---------------------------------------------------------------------------
def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None or self_action is None:
        return

    framework_events = list(events)
    events = framework_events + detect_custom_events(
        old_game_state, self_action, new_game_state, framework_events
    )
    reward = reward_from_events(events, self.logger)
    reward += potential_shaping(old_game_state, new_game_state, self.t["gamma"])

    phi = self.encode_state(old_game_state)
    self.traj.append((phi, _ACTION_ID[self_action], reward))
    self.ep_reward += reward
    self.ep_counts.update(events)

    self.step_count += 1
    if self.step_count % self.t["learn_every"] == 0:
        _learn(self, n_iters=1)


# ---------------------------------------------------------------------------
def end_of_round(self, last_game_state, last_action, events):
    events = list(events)
    died = e.KILLED_SELF in events or e.GOT_KILLED in events
    survived = e.SURVIVED_ROUND in events

    # The framework delivers a survivor's final step to game_events_occurred AND
    # again here (plus SURVIVED_ROUND). A dead agent's final step comes ONLY here.
    if died or not survived or not self.traj:
        reward = reward_from_events(events, self.logger)  # terminal: no shaping term
        if last_action is not None:
            phi = self.encode_state(last_game_state)
            self.traj.append((phi, _ACTION_ID[last_action], reward))
        self.ep_reward += reward
        self.ep_counts.update(events)
    else:
        # survivor: fold only the genuinely-new SURVIVED_ROUND into the last step
        bonus = reward_from_events([e.SURVIVED_ROUND])
        phi, a, r = self.traj[-1]
        self.traj[-1] = (phi, a, r + bonus)
        self.ep_reward += bonus
        self.ep_counts[e.SURVIVED_ROUND] += 1

    _flush_episode_to_buffer(self)
    _learn(self, n_iters=self.t["learn_iters_end"])

    self.episode += 1
    self.model.save(self._weights_path)  # always keep the latest
    _save_periodic_candidate(self)
    _update_best_checkpoint(self)

    _write_csv_row(self, last_game_state)

    # reset for next episode
    self.epsilon = _epsilon(self, self.episode)
    self.alpha = _alpha(self, self.episode)
    self.traj = []
    self.ep_counts = Counter()
    self.ep_reward = 0.0


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------
def _save_periodic_candidate(self):
    """Preserve validation candidates; `save_every` used to be configured but unused."""
    every = int(self.t.get("save_every", 0) or 0)
    if every <= 0 or self.episode % every != 0:
        return
    candidate_dir = self.t.get("candidate_dir", "weights/candidates")
    os.makedirs(candidate_dir, exist_ok=True)
    _, ext = os.path.splitext(self._weights_path)
    path = os.path.join(candidate_dir, f"ep_{self.episode:05d}{ext}")
    shutil.copyfile(self._weights_path, path)
    self.logger.info(f"saved validation candidate -> {path}")


def _update_best_checkpoint(self):
    """
    Keep a cheap training-time best checkpoint without pretending it is final
    model selection. Task 1/2 prioritise coins; Task 3/4 prioritise true game
    score (coins + 5*kills), then kills, then lower self-kill rate, then coins.

    Because epsilon is non-zero during training, use the periodic candidates +
    validate_checkpoints.py for the final choice on separate validation seeds.
    """
    metrics = (
        self.ep_counts[e.COIN_COLLECTED],
        self.ep_counts[e.KILLED_OPPONENT],
        int(self.ep_counts[e.KILLED_SELF] > 0),
    )
    self._recent_metrics.append(metrics)
    if len(self._recent_metrics) < self._recent_metrics.maxlen:
        return

    n = len(self._recent_metrics)
    avg_coins = sum(m[0] for m in self._recent_metrics) / n
    avg_kills = sum(m[1] for m in self._recent_metrics) / n
    self_kill_rate = sum(m[2] for m in self._recent_metrics) / n
    preset = self.t.get("preset", "task1")

    if preset in ("task3", "task4"):
        avg_score = avg_coins + 5.0 * avg_kills
        key = (avg_score, avg_kills, -self_kill_rate, avg_coins)
        label = (
            f"score={avg_score:.2f}, kills={avg_kills:.3f}, "
            f"self_kill={100*self_kill_rate:.1f}%, coins={avg_coins:.2f}"
        )
    elif preset == "task2":
        key = (avg_coins, -self_kill_rate)
        label = f"coins={avg_coins:.2f}, self_kill={100*self_kill_rate:.1f}%"
    else:
        key = (avg_coins,)
        label = f"coins={avg_coins:.2f}"

    if self._best_checkpoint_key is None or key > self._best_checkpoint_key:
        self._best_checkpoint_key = key
        shutil.copyfile(self._weights_path, self._best_weights_path)
        self.logger.info(
            f"new training-proxy best ({label}, last 100) -> {self._best_weights_path}"
        )


def _epsilon(self, episode):
    c = TRAIN
    frac = min(1.0, episode / max(1, c["eps_decay_episodes"]))
    return c["eps_end"] + (c["eps_start"] - c["eps_end"]) * (1.0 - frac)


def _alpha(self, episode):
    c = TRAIN
    a0, a1 = c["alpha"], c.get("alpha_end")
    if a1 is None:
        return a0
    frac = min(1.0, episode / max(1, c["eps_decay_episodes"]))
    return a0 + (a1 - a0) * frac


def _flush_episode_to_buffer(self):
    """Turn this episode's trajectory into k-step TD transitions."""
    traj, T = self.traj, len(self.traj)
    if T == 0:
        return
    gamma, k = self.t["gamma"], self.t["n_step"]
    zeros = np.zeros(self.feature_dim, dtype=np.float32)

    for t in range(T):
        R, discount = 0.0, 1.0
        for i in range(k):
            if t + i >= T:
                break
            R += discount * traj[t + i][2]
            discount *= gamma
        j = t + k
        if j < T:
            boot_phi, boot_a, done = traj[j][0], traj[j][1], 0.0
        else:
            boot_phi, boot_a, done = zeros, 0, 1.0

        phi_t, a_t, _ = traj[t]
        _push(self, phi_t, a_t, R, boot_phi, boot_a, done)


def _push(self, phi, a, R, boot_phi, boot_a, done):
    ep = self.episode
    self.buffer.push(phi, a, R, boot_phi, boot_a, done, ep)
    if self._sym is not None and self.t["use_symmetry"]:
        for op in self._sym.sym_transforms():
            if op == (False, 0):
                continue
            self.buffer.push(
                self.transform_input(phi, op),
                self._sym.apply_to_action(a, op),
                R,
                self.transform_input(boot_phi, op),
                self._sym.apply_to_action(boot_a, op),
                done,
                ep,
            )


def _learn(self, n_iters=1):
    cfg = self.t
    if len(self.buffer) < cfg["batch_size"]:
        return
    for _ in range(n_iters):
        batch, idx, w = self.buffer.sample(cfg["batch_size"])
        v_next = self.model.bootstrap_value(
            batch["next_phi"],
            batch["done"],
            next_actions=batch["next_action"] if cfg["rule"] == "sarsa" else None,
            rule=cfg["rule"],
        )
        targets = batch["reward"] + (cfg["gamma"] ** cfg["n_step"]) * v_next
        td_err = self.model.update(
            batch["phi"],
            batch["action"],
            targets,
            self.alpha,
            weights=w,
            td_clip=cfg.get("td_clip"),
        )
        if cfg["priority_alpha"] > 0.0:
            self.buffer.update_priorities(idx, td_err)


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------
_CSV_FIELDS = [
    "episode",
    "steps",
    "total_reward",
    "coins",
    "crates",
    "invalid",
    "killed_self",
    "killed_opponents",
    "survived",
    "epsilon",
    "alpha",
    "buffer",
]


def _csv_header(path):
    if not os.path.isfile(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(_CSV_FIELDS)


def _write_csv_row(self, last_game_state):
    c = self.ep_counts
    steps = last_game_state["step"] if last_game_state else 0
    row = [
        self.episode,
        steps,
        round(self.ep_reward, 3),
        c[e.COIN_COLLECTED],
        c[e.CRATE_DESTROYED],
        c[e.INVALID_ACTION],
        int(c[e.KILLED_SELF] > 0),
        c[e.KILLED_OPPONENT],
        int(c[e.SURVIVED_ROUND] > 0),
        round(self.epsilon, 3),
        round(self.alpha, 4),
        len(self.buffer),
    ]
    with open(self._csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)
    self.logger.info(
        f"ep {self.episode}: steps={steps} reward={self.ep_reward:.1f} "
        f"coins={c[e.COIN_COLLECTED]} invalid={c[e.INVALID_ACTION]} "
        f"self_kill={int(c[e.KILLED_SELF] > 0)} eps={self.epsilon:.2f}"
    )
