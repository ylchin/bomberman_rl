"""
callbacks.py  --  our_agent  (Yi Ling Chin)

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

    weights_path = config.WEIGHTS_FILE[config.MODEL]

    if self.model_kind == "linear":
        from .q_linear import LinearQ
        if self.train and not config.TRAIN["resume"]:
            self.logger.info("Fresh LinearQ model.")
            self.model = LinearQ()
        elif os.path.isfile(weights_path):
            self.logger.info(f"Loading LinearQ from {weights_path}.")
            self.model = LinearQ.load(weights_path)
        elif self.train:
            self.logger.info("No checkpoint to resume; starting LinearQ from scratch.")
            self.model = LinearQ()
        else:
            raise FileNotFoundError(
                f"{weights_path} missing and not in training mode -- train an agent first."
            )
    else:
        raise NotImplementedError(f"model kind {self.model_kind!r} not wired up yet")

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
