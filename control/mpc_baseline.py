"""Simple one-step MPC baseline using action grid search."""

from __future__ import annotations

from dataclasses import dataclass

import itertools
import numpy as np

from envs.reward_models import RewardWeights, reward
from models.coupled_system import CoupledSystem


@dataclass
class MPCConfig:
    grid: tuple[float, ...] = (0.0, 0.5, 1.0)


class OneStepMPC:
    def __init__(self, system: CoupledSystem, reward_weights: RewardWeights | None = None, config: MPCConfig | None = None):
        self.system = system
        self.reward_weights = reward_weights or RewardWeights()
        self.config = config or MPCConfig()

    def __call__(self, t: int, state: np.ndarray, rainfall: float) -> np.ndarray:
        best_u = None
        best_r = -np.inf
        for u_vals in itertools.product(self.config.grid, repeat=4):
            u = np.array(u_vals, dtype=float)
            next_state = self.system.step(state, u, rainfall)
            r = reward(state, next_state, u, self.reward_weights)
            if r > best_r:
                best_r = r
                best_u = u
        return best_u if best_u is not None else np.zeros(4, dtype=float)
