"""
replay_buffer.py 

Experience replay for the value-based models. 

Design follows lecture:
  - sample a batch to decorrelate the online updates
  - never put two transitions from the *same episode* in one batch
  - optionally prefer transitions with large |TD error| (prioritised replay)

Start with alpha=0 (uniform + the episode rule). Turn on prioritisation once
the linear model trains at all, by passing alpha>0 and calling
update_priorities() after each learn step.
"""

from collections import namedtuple

import numpy as np

Transition = namedtuple(
    "Transition",
    ["phi", "action", "reward", "next_phi", "next_action", "done", "episode"],
)


class ReplayBuffer:
    def __init__(self, capacity, feature_dim, rng=None, alpha=0.0, eps=1e-3):
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.rng = rng or np.random.default_rng()
        self.alpha = float(alpha)          # 0 -> uniform, >0 -> prioritised
        self.eps = float(eps)              # floor added to every priority

        self.phi = np.zeros((self.capacity, feature_dim), dtype=np.float32)
        self.next_phi = np.zeros((self.capacity, feature_dim), dtype=np.float32)
        self.action = np.zeros(self.capacity, dtype=np.int64)
        self.next_action = np.zeros(self.capacity, dtype=np.int64)   # for SARSA bootstrap
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.done = np.zeros(self.capacity, dtype=np.float32)
        self.episode = np.zeros(self.capacity, dtype=np.int64)
        self.priority = np.zeros(self.capacity, dtype=np.float64)

        self._size = 0
        self._next = 0
        self._max_priority = 1.0

    def __len__(self):
        return self._size

    def push(self, phi, action, reward, next_phi, next_action, done, episode):
        i = self._next
        self.phi[i] = phi
        # terminal transition: next_phi is None -> store zeros, done flag masks it
        self.next_phi[i] = 0.0 if next_phi is None else next_phi
        self.action[i] = action
        self.next_action[i] = 0 if next_action is None else next_action
        self.reward[i] = reward
        self.done[i] = float(done)
        self.episode[i] = episode
        self.priority[i] = self._max_priority       # new samples: max priority

        self._next = (self._next + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Returns (batch_dict, indices, is_weights).
        batch_dict has keys phi, action, reward, next_phi, done  (all np arrays).
        `indices` is what you pass back to update_priorities().
        `is_weights` are importance-sampling weights (all 1.0 when alpha==0).
        """
        n = self._size
        batch_size = min(batch_size, n)

        if self.alpha > 0.0:
            p = self.priority[:n] ** self.alpha
            p /= p.sum()
            idx = self.rng.choice(n, size=batch_size, replace=False, p=p)
            # IS weights, normalised to <=1 (beta fixed at 1.0 -> full correction)
            w = (n * p[idx]) ** (-1.0)
            w /= w.max()
        else:
            idx = self._uniform_episode_spread(n, batch_size)
            w = np.ones(len(idx), dtype=np.float32)

        batch = {
            "phi": self.phi[idx],
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_phi": self.next_phi[idx],
            "next_action": self.next_action[idx],
            "done": self.done[idx],
        }
        return batch, idx, w

    def _uniform_episode_spread(self, n, batch_size):
        """
        Uniform sample that avoids two transitions from the same episode
        (Lecture 36 sec.7). O(batch_size) -- draws a random candidate pool
        rather than permuting the whole buffer.
        """
        pool = self.rng.integers(0, n, size=min(n, batch_size * 8))
        chosen, seen = [], set()
        for i in pool:
            ep = self.episode[i]
            if ep in seen:
                continue
            seen.add(ep)
            chosen.append(int(i))
            if len(chosen) == batch_size:
                break
        # early training: too few distinct episodes -> relax to fill the batch
        if len(chosen) < batch_size:
            chosen.extend(int(i) for i in self.rng.integers(0, n, size=batch_size - len(chosen)))
        return np.array(chosen, dtype=np.int64)

    def update_priorities(self, indices, td_errors):
        td = np.abs(np.asarray(td_errors, dtype=np.float64)) + self.eps
        self.priority[indices] = td
        self._max_priority = max(self._max_priority, td.max())
