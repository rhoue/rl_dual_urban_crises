"""Resource budget constraints for interventions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Budget:
    weights: np.ndarray
    max_budget: float

    def feasible(self, u: np.ndarray) -> bool:
        return float(np.dot(self.weights, u)) <= self.max_budget + 1e-9

    def project(self, u: np.ndarray) -> np.ndarray:
        cost = float(np.dot(self.weights, u))
        if cost <= self.max_budget:
            return u
        if cost <= 1e-9:
            return u
        scale = self.max_budget / cost
        return u * scale
