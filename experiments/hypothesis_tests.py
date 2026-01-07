"""Hypothesis-driven structural checks."""

from __future__ import annotations

import numpy as np

from models.coupled_system import CoupledSystem, default_state, simulate
from scenarios.rainfall_generator import deterministic_rainfall


def _zero_policy(_: int, __: np.ndarray) -> np.ndarray:
    return np.zeros(4, dtype=float)


def test_flood_increases_infection(steps: int = 60) -> dict:
    system = CoupledSystem()
    low_rain = deterministic_rainfall(steps, intensity=0.3)
    high_rain = deterministic_rainfall(steps, intensity=1.2)
    x0 = default_state()
    low = simulate(system, x0, _zero_policy, low_rain)
    high = simulate(system, x0, _zero_policy, high_rain)
    return {
        "low_peak_infected": float(np.max(low[:, 4])),
        "high_peak_infected": float(np.max(high[:, 4])),
    }


def test_crowding_increases_infection(steps: int = 30) -> dict:
    system = CoupledSystem()
    rain = deterministic_rainfall(steps, intensity=0.5)
    x0 = default_state()
    x0_high_crowd = x0.copy()
    x0_high_crowd[11] = 2.0
    base = simulate(system, x0, _zero_policy, rain)
    high = simulate(system, x0_high_crowd, _zero_policy, rain)
    return {
        "base_peak_infected": float(np.max(base[:, 4])),
        "crowding_peak_infected": float(np.max(high[:, 4])),
    }
