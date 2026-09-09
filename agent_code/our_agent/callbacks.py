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


def act(self, game_state: dict) -> str:
    phi = state_to_features(game_state)
    if phi is None:
        return "WAIT"

    if self.train:
        beta = config.TRAIN["softmax_beta"]
        action_id = self.model.act(phi, epsilon=self.epsilon, beta=beta, rng=self.rng)
    else:
        action_id = self.model.act(phi, epsilon=0.0, rng=self.rng)

    action = ACTIONS[action_id]
    self.logger.debug(f"step {game_state['step']}: {action} (eps={getattr(self, 'epsilon', 0):.2f})")
    return action
