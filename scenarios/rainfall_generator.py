"""Rainfall scenario generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RainfallScenario:
    steps: int = 60
    baseline: float = 0.2
    pulse_prob: float = 0.1
    pulse_mean: float = 2.0
    pulse_std: float = 0.5
    rng_seed: int | None = None

    def generate(self) -> np.ndarray:
        rng = np.random.default_rng(self.rng_seed)
        rain = rng.normal(self.baseline, 0.05, size=self.steps)
        pulses = rng.random(self.steps) < self.pulse_prob
        pulse_vals = rng.normal(self.pulse_mean, self.pulse_std, size=self.steps)
        rain = np.clip(rain + pulses * np.maximum(pulse_vals, 0.0), 0.0, None)
        return rain


def deterministic_rainfall(steps: int, intensity: float, decay: float = 0.98) -> np.ndarray:
    rain = np.zeros(steps, dtype=float)
    value = intensity
    for t in range(steps):
        rain[t] = max(value, 0.0)
        value *= decay
    return rain
