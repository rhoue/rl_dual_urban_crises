"""Reward models for decision support."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.coupled_system import IDX


@dataclass
class RewardWeights:
    infected: float = 1.0
    contamination: float = 0.6
    deaths: float = 2.0
    econ_loss: float = 0.4
    control_l1: float = 0.1


def reward(state: np.ndarray, next_state: np.ndarray, action: np.ndarray, w: RewardWeights) -> float:
    d_deaths = next_state[IDX["deaths"]] - state[IDX["deaths"]]
    d_econ = next_state[IDX["econ_loss"]] - state[IDX["econ_loss"]]
    cost = (
        w.infected * next_state[IDX["infected"]]
        + w.contamination * next_state[IDX["contamination"]]
        + w.deaths * max(d_deaths, 0.0)
        + w.econ_loss * max(d_econ, 0.0)
        + w.control_l1 * float(np.sum(np.abs(action)))
    )
    return -float(cost)
