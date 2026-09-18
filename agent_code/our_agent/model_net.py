"""Model 2: convolutional Double DQN, using the shared n-step replay trainer."""

from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from settings import COLS, ROWS
from .features import ACTIONS, N_CHANNELS, state_to_channels

INPUT_SHAPE = (N_CHANNELS, COLS, ROWS)
INPUT_DIM = int(np.prod(INPUT_SHAPE))


def encode_state(game_state):
    channels = state_to_channels(game_state)
    if channels is None:
        return None
    if channels.shape != INPUT_SHAPE:
        raise ValueError(f"Expected board channels {INPUT_SHAPE}, got {channels.shape}")
    return channels.reshape(-1)


def transform_input(vector, op):
    """Match symmetry.apply_to_coord on arrays indexed [channel, x, y]."""
    channels = vector.reshape(INPUT_SHAPE)
    flip, rotations = op
    if flip:
        channels = np.flip(channels, axis=1)
    return np.rot90(channels, k=-rotations, axes=(1, 2)).copy().reshape(-1)


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * ((COLS + 3) // 4) * ((ROWS + 3) // 4), 128),
            nn.ReLU(), nn.Linear(128, len(ACTIONS)),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class NetQ:
    def __init__(self, seed=None, device="cpu", target_update_every=500,
                 grad_clip=10.0, torch_threads=1):
        if target_update_every < 1 or grad_clip <= 0 or torch_threads < 1:
            raise ValueError("Target interval, gradient clip and thread count must be positive")
        torch.set_num_threads(torch_threads)
        self.device = torch.device(device)
        # Keep initialization reproducible without changing the caller's torch RNG.
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.manual_seed(seed)
            self.online = QNetwork().to(self.device)
        self.target = deepcopy(self.online).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=1e-4)
        self.target_update_every = target_update_every
        self.grad_clip = grad_clip
        self.updates = 0
        self._rng = np.random.default_rng(seed)

    def _tensor(self, inputs):
        array = np.asarray(inputs, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim == 3:
            array = array[None, :]
        if array.ndim == 2 and array.shape[1] == INPUT_DIM:
            array = array.reshape(-1, *INPUT_SHAPE)
        if array.ndim != 4 or tuple(array.shape[1:]) != INPUT_SHAPE:
            raise ValueError(f"Expected flattened or spatial inputs with shape {INPUT_SHAPE}")
        return torch.as_tensor(np.ascontiguousarray(array), device=self.device)

    @torch.no_grad()
    def q_values_batch(self, inputs):
        self.online.eval()
        return self.online(self._tensor(inputs)).cpu().numpy()

    def q_values(self, inputs):
        return self.q_values_batch(inputs)[0]

    def act(self, phi, epsilon=0.0, beta=None, rng=None):
        rng = rng or self._rng
        if beta is None and rng.random() < epsilon:
            return int(rng.integers(len(ACTIONS)))
        q = self.q_values(phi)
        if beta is not None:
            probabilities = np.exp(beta * q - np.max(beta * q))
            return int(rng.choice(len(ACTIONS), p=probabilities / probabilities.sum()))
        return int(rng.choice(np.flatnonzero(q == q.max())))

    @torch.no_grad()
    def bootstrap_value(self, next_Phi, done, next_actions=None, rule="q"):
        self.online.eval()
        inputs = self._tensor(next_Phi)
        if rule == "q":
            # Online network chooses; frozen target network evaluates.
            actions = self.online(inputs).argmax(dim=1)
        elif rule == "sarsa" and next_actions is not None:
            actions = torch.as_tensor(next_actions, dtype=torch.long, device=self.device)
        else:
            raise ValueError("Expected rule='q', or rule='sarsa' with next_actions")
        values = self.target(inputs).gather(1, actions[:, None]).squeeze(1)
        values = values.cpu().numpy()
        return np.where(np.asarray(done, dtype=bool), 0.0, values)

    def update(self, Phi, actions, targets, alpha, weights=None, td_clip=None):
        self.online.train()
        for group in self.optimizer.param_groups:
            group["lr"] = alpha
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        targets = torch.as_tensor(targets, dtype=torch.float32, device=self.device)
        predictions = self.online(self._tensor(Phi)).gather(1, actions[:, None]).squeeze(1)
        errors = (targets - predictions).detach().cpu().numpy()
        loss = F.smooth_l1_loss(predictions, targets, reduction="none")
        if weights is not None:
            loss = loss * torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        self.optimizer.zero_grad(set_to_none=True)
        loss.mean().backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.target_update_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return errors

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(dict(version=1, actions=ACTIONS, input_shape=INPUT_SHAPE,
                        online=self.online.state_dict(), target=self.target.state_dict(),
                        optimizer=self.optimizer.state_dict(), updates=self.updates,
                        target_update_every=self.target_update_every,
                        grad_clip=self.grad_clip), temporary)
        # Windows scanners/readers can briefly deny replacement of an existing
        # checkpoint. Keep the valid old file and retry the atomic replacement.
        for attempt in range(20):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.25)

    @classmethod
    def load(cls, path, *, device="cpu", seed=None, training=False,
             target_update_every=500, grad_clip=10.0, torch_threads=1):
        data = torch.load(path, map_location=device, weights_only=True)
        if (data.get("version") != 1 or data.get("actions") != ACTIONS
                or tuple(data.get("input_shape", ())) != INPUT_SHAPE):
            raise ValueError("Incompatible CNN checkpoint version, actions or input shape")
        model = cls(seed=seed, device=device, target_update_every=target_update_every,
                    grad_clip=grad_clip, torch_threads=torch_threads)
        model.online.load_state_dict(data["online"])
        if not all(torch.isfinite(p).all() for p in model.online.parameters()):
            raise ValueError("Checkpoint contains non-finite weights")
        model.target.load_state_dict(data["target"] if training else data["online"])
        if training:
            model.optimizer.load_state_dict(data["optimizer"])
            model.updates = int(data["updates"])
            model.target_update_every = int(data["target_update_every"])
            model.grad_clip = float(data["grad_clip"])
        return model
