"""
callbacks.py  --  our_agent 

Tournament entry point. Must stay importable with NO training-only deps
(train.py is only imported by the framework when --train is set).

    setup(self)            once, before the first round
    act(self, game_state)  -> one of features.ACTIONS, every step
"""

import os

import numpy as np

from .features import ACTIONS, state_to_features
from . import config

STUCK_AFTER = 2  # consecutive ineffective repeats of the same move before we force a change


def setup(self):
    """
    self.train is set by the framework. When training, train.setup_training()
    runs right after this and may replace self.model with a fresh one.
    """
    self.rng = np.random.default_rng()
    self.cfg = config
    self.model_kind = config.MODEL

    if self.model_kind != "linear":
        raise NotImplementedError(f"model kind {self.model_kind!r} not wired up yet")
    from .q_linear import LinearQ

    if self.train:
        resume_path = config.TRAIN["weights_out"]
        if config.TRAIN["resume"] and os.path.isfile(resume_path):
            self.logger.info(f"Resuming LinearQ from {resume_path}.")
            self.model = LinearQ.load(resume_path)
        else:
            self.logger.info("Fresh LinearQ model.")
            self.model = LinearQ()
    else:
        for path in config.EVAL_WEIGHTS:
            if os.path.isfile(path):
                self.logger.info(f"Loading LinearQ from {path}.")
                self.model = LinearQ.load(path)
                break
        else:
            raise FileNotFoundError(
                f"No checkpoint found (looked in {config.EVAL_WEIGHTS}) and not "
                f"training -- train an agent first, or set AGENT_PRESET to match."
            )

    # act() reads this; train.py keeps it up to date during training.
    self.epsilon = 0.0
    self._last_pos = None
    self._last_action_id = None
    self._repeat_count = 0


def act(self, game_state: dict) -> str:
    phi = state_to_features(game_state)
    if phi is None:
        return "WAIT"

    if game_state["step"] == 1:
        # new round -- don't compare across the round boundary
        self._last_pos = None
        self._repeat_count = 0

    if self.train:
        beta = config.TRAIN["softmax_beta"]
        action_id = self.model.act(phi, epsilon=self.epsilon, beta=beta, rng=self.rng)
    else:
        action_id = self.model.act(phi, epsilon=0.0, rng=self.rng)

    # Safety net: a greedy (eps=0) policy that picks an action the game rejects
    # (wall/crate/out of bounds) sees an unchanged game_state next step, so it
    # picks the exact same losing action again -- forever, silently, at score 0.
    # If we're about to repeat a move that already failed to move us last step,
    # break the loop with a random action instead of trusting the model again.
    pos = game_state["self"][3]
    ineffective_repeat = (
        pos == self._last_pos
        and action_id == self._last_action_id
        and ACTIONS[action_id] != "WAIT"
    )
    self._repeat_count = self._repeat_count + 1 if ineffective_repeat else 0
    if self._repeat_count >= STUCK_AFTER:
        stuck_on = ACTIONS[action_id]
        action_id = int(self.rng.integers(len(ACTIONS)))
        self.logger.warning(
            f"step {game_state['step']}: stuck at {pos} repeating {stuck_on!r} "
            f"with no effect; forcing random action {ACTIONS[action_id]!r} instead"
        )
        self._repeat_count = 0

    self._last_pos = pos
    self._last_action_id = action_id

    action = ACTIONS[action_id]
    self.logger.debug(f"step {game_state['step']}: {action} (eps={getattr(self, 'epsilon', 0):.2f})")
    return action
