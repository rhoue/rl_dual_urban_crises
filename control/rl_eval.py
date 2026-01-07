"""Evaluation utilities for RL policies."""

from __future__ import annotations

import numpy as np

try:
    from stable_baselines3.common.base_class import BaseAlgorithm
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("stable-baselines3 is required for rl_eval") from exc

from envs.dualcrisis_env import DualCrisisEnv


def evaluate_policy(env: DualCrisisEnv, model: BaseAlgorithm, episodes: int = 5) -> dict:
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        returns.append(total_reward)
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
    }
