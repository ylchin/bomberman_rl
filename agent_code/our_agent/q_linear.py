"""
q_linear.py

Model 1: linear action-value regression

    Q(s, a) = phi(s) . w_a          # one weight vector per action

Trained on k-step TD targets by semi-gradient descent:

    w_a  <-  w_a + alpha * (R_t - Q(s_t, a_t)) * phi(s_t)

with the target return

    R_t = sum_{i=0..k-1} gamma^i r_{t+i}  +  gamma^k * V(s_{t+k})

    V(s') = max_a' Q(s', a')     -> rule = "q"      (Q-learning)
    V(s') = Q(s', a'_taken)      -> rule = "sarsa"  (SARSA)

"""

import pickle

import numpy as np

from .features import ACTIONS, FEATURE_DIM

N_ACTIONS = len(ACTIONS)


class LinearQ:
    def __init__(self, feature_dim=FEATURE_DIM, n_actions=N_ACTIONS, seed=None):
        self.feature_dim = feature_dim
        self.n_actions = n_actions
        # init at zero: Q^(0)(s,a) = 0, and Q(terminal, .) = 0  (Lecture 36 sec.7)
        self.W = np.zeros((n_actions, feature_dim), dtype=np.float64)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ predict
    def q_values(self, phi):
        """phi: (D,) -> (n_actions,)"""
        return self.W @ phi

    def q_values_batch(self, Phi):
        """Phi: (N, D) -> (N, n_actions)"""
        return Phi @ self.W.T

    # ------------------------------------------------------------------ act
    @staticmethod
    def legal_actions(phi):
        """Use observed move occupancy and bomb availability, not tactical rules."""
        phi = np.asarray(phi)
        legal = np.ones((*phi.shape[:-1], N_ACTIONS), dtype=bool)
        legal[..., :4] = phi[..., :4] < 0.5
        legal[..., 5] = phi[..., 4] > 0.5
        return legal

    def act(self, phi, epsilon=0.0, beta=None, rng=None, action_mask=None):
        """
        Returns an action id in [0, n_actions).
          epsilon > 0  -> epsilon-greedy
          beta not None -> softmax(beta * Q) (overrides epsilon-greedy)
        """
        rng = rng or self._rng
        legal = self.legal_actions(phi)
        if action_mask is not None:
            legal &= np.asarray(action_mask, dtype=bool)
        candidates = np.flatnonzero(legal)
        if len(candidates) == 0:
            return ACTIONS.index("WAIT")
        if beta is not None:
            q = self.q_values(phi)[candidates]
            q = q - q.max()
            p = np.exp(beta * q)
            p /= p.sum()
            return int(rng.choice(candidates, p=p))
        if epsilon > 0.0 and rng.random() < epsilon:
            return int(rng.choice(candidates))
        q = self.q_values(phi)[candidates]
        # random tie-break among the argmax set
        best = np.flatnonzero(q == q.max())
        return int(rng.choice(candidates[best]))

    # ------------------------------------------------------------------ targets
    def bootstrap_value(self, next_Phi, done, next_actions=None, rule="q"):
        """
        V(s') for a batch, masked to 0 where done.
        next_Phi: (N, D), done: (N,), next_actions: (N,) needed for SARSA.
        """
        q_next = self.q_values_batch(next_Phi)           # (N, n_actions)
        if rule == "sarsa":
            if next_actions is None:
                raise ValueError("SARSA target needs next_actions")
            v = q_next[np.arange(len(q_next)), next_actions]
        else:
            v = np.where(self.legal_actions(next_Phi), q_next, -np.inf).max(axis=1)
        return v * (1.0 - np.asarray(done, dtype=np.float64))

    # ------------------------------------------------------------------ learn
    def update(self, Phi, actions, targets, alpha, weights=None, td_clip=None):
        """
        One semi-gradient TD step over a batch.
          Phi: (N, D)   actions: (N,)   targets: (N,)  (the R_t values)
          weights: (N,) importance-sampling weights (prioritised replay); default 1
          td_clip: if set, the TD error driving the gradient is clipped to
                   [-td_clip, td_clip] (Huber-style: bounds the step a rare
                   death spike (-10) can take). Priorities still see the raw error.
        Returns the per-sample (unclipped) TD error, for replay priorities.
        """
        Phi = np.asarray(Phi, dtype=np.float64)
        actions = np.asarray(actions)
        targets = np.asarray(targets, dtype=np.float64)
        w = np.ones(len(Phi)) if weights is None else np.asarray(weights, dtype=np.float64)

        q_pred = np.einsum("nd,nd->n", Phi, self.W[actions])   # Q(s_i, a_i)
        td_error = targets - q_pred
        step_error = td_error if td_clip is None else np.clip(td_error, -td_clip, td_clip)

        # accumulate per-action gradient:  dL/dw_a = -sum_i (err_i * weight_i) phi_i
        grad = np.zeros_like(self.W)
        np.add.at(grad, actions, (w * step_error)[:, None] * Phi)
        counts = np.bincount(actions, minlength=self.n_actions).astype(np.float64)
        counts[counts == 0] = 1.0
        self.W += alpha * grad / counts[:, None]

        return td_error

    # ------------------------------------------------------------------ io
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"W": self.W, "feature_dim": self.feature_dim,
                         "n_actions": self.n_actions, "actions": ACTIONS}, f)

    @classmethod
    def load(cls, path, *, allow_legacy_padding=False):
        with open(path, "rb") as f:
            data = pickle.load(f)
        if data.get("actions") != ACTIONS:
            raise ValueError("Checkpoint action order differs from features.ACTIONS")
        old_dim = data["feature_dim"]
        weights = np.asarray(data["W"], dtype=np.float64)
        if data["n_actions"] != N_ACTIONS or weights.shape != (N_ACTIONS, old_dim):
            raise ValueError("Checkpoint weight shape or action count is invalid")
        if not np.isfinite(weights).all():
            raise ValueError("Checkpoint contains non-finite weights")
        legacy_ok = allow_legacy_padding and old_dim in (32, 34) and old_dim < FEATURE_DIM
        if old_dim != FEATURE_DIM and not legacy_ok:
            raise ValueError(
                f"Checkpoint has {old_dim} features; expected {FEATURE_DIM}. "
                "Set AGENT_INIT_CHECKPOINT to explicitly pad a legacy 32/34-dim model."
            )
        m = cls()
        m.W[:, :old_dim] = weights
        return m
