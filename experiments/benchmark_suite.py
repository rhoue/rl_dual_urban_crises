"""Benchmark suite for policy comparisons."""

from __future__ import annotations

import numpy as np

from control.mpc_baseline import OneStepMPC
from control.rule_based import RuleBasedPolicy
from envs.reward_models import RewardWeights, reward
from models.coupled_system import CoupledSystem, default_state
from scenarios.rainfall_generator import RainfallScenario


def _rollout(system: CoupledSystem, policy, rainfall: np.ndarray, reward_weights: RewardWeights) -> float:
    state = default_state()
    total = 0.0
    for t, r in enumerate(rainfall):
        if isinstance(policy, OneStepMPC):
            action = policy(t, state, float(r))
        else:
            action = policy(t, state)
        next_state = system.step(state, action, float(r))
        total += reward(state, next_state, action, reward_weights)
        state = next_state
    return float(total)


def run_benchmark(steps: int = 60) -> dict:
    rainfall = RainfallScenario(steps=steps).generate()
    system = CoupledSystem()
    weights = RewardWeights()
    rule_based = RuleBasedPolicy()
    mpc = OneStepMPC(system, weights)
    return {
        "rule_based_return": _rollout(system, rule_based, rainfall, weights),
        "mpc_return": _rollout(system, mpc, rainfall, weights),
    }
