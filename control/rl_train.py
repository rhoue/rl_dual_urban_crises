"""Training utilities for RL baselines."""

from __future__ import annotations

import numpy as np

try:
    from stable_baselines3 import PPO
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("stable-baselines3 is required for rl_train") from exc

from envs.dualcrisis_env import DualCrisisEnv


def train_ppo(env: DualCrisisEnv, total_timesteps: int = 10000, seed: int | None = None) -> PPO:
    if seed is not None:
        env.reset(seed=seed)
        np.random.seed(seed)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    return model
