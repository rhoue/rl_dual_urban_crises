"""Gymnasium environment for the dual crisis digital twin."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("gymnasium is required for DualCrisisEnv") from exc

from envs.observation_models import ObservationParams, observe
from envs.reward_models import RewardWeights, reward
from models.coupled_system import CoupledParams, CoupledSystem, STATE_NAMES, default_state
from scenarios.policy_budgets import Budget


@dataclass
class EnvConfig:
    horizon: int = 60
    dt: float = 1.0
    action_high: float = 1.0
    seed: int | None = None


class DualCrisisEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        rainfall: np.ndarray,
        config: EnvConfig | None = None,
        system_params: CoupledParams | None = None,
        obs_params: ObservationParams | None = None,
        reward_weights: RewardWeights | None = None,
        budget: Budget | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self.system = CoupledSystem(system_params, dt=self.config.dt)
        self.obs_params = obs_params or ObservationParams()
        self.reward_weights = reward_weights or RewardWeights()
        self.budget = budget
        self.rainfall = np.asarray(rainfall, dtype=float)
        self.rng = np.random.default_rng(self.config.seed)
        self.state = default_state()
        obs_dim = len(self.obs_params.observed_indices)
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=0.0,
            high=self.config.action_high,
            shape=(4,),
            dtype=np.float32,
        )
        self.t = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.state = default_state()
        obs = observe(self.state, self.obs_params, self.rng)
        info = {"state": self.state.copy(), "state_names": STATE_NAMES}
        return obs.astype(np.float32), info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=float)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.budget is not None:
            action = self.budget.project(action)
        rainfall = float(self.rainfall[min(self.t, len(self.rainfall) - 1)])
        next_state = self.system.step(self.state, action, rainfall)
        r = reward(self.state, next_state, action, self.reward_weights)
        self.t += 1
        terminated = self.t >= self.config.horizon
        obs = observe(next_state, self.obs_params, self.rng)
        info = {"state": next_state.copy(), "action": action.copy(), "rainfall": rainfall}
        self.state = next_state
        return obs.astype(np.float32), r, terminated, False, info
