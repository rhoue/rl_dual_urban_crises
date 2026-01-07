"""Rule-based control policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.coupled_system import IDX


@dataclass
class RuleThresholds:
    flood: float = 1.0
    contamination: float = 0.8
    infected: float = 0.6
    displacement: float = 0.6


class RuleBasedPolicy:
    def __init__(self, thresholds: RuleThresholds | None = None, action_high: float = 1.0):
        self.thresholds = thresholds or RuleThresholds()
        self.action_high = action_high

    def __call__(self, t: int, state: np.ndarray) -> np.ndarray:
        u = np.zeros(4, dtype=float)
        if state[IDX["contamination"]] > self.thresholds.contamination:
            u[0] = self.action_high
        if state[IDX["displaced"]] > self.thresholds.displacement:
            u[1] = self.action_high
        if state[IDX["infected"]] > self.thresholds.infected:
            u[2] = self.action_high
        if state[IDX["damage"]] > self.thresholds.flood:
            u[3] = self.action_high
        return u
