"""Parameter sweep utilities."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from models.coupled_system import CoupledParams, CoupledSystem, default_state, simulate
from scenarios.rainfall_generator import RainfallScenario


def sweep_gamma8(values: list[float], steps: int = 60) -> dict:
    rainfall = RainfallScenario(steps=steps).generate()
    results = {}
    for value in values:
        params = CoupledParams()
        params.infrastructure = replace(params.infrastructure, gamma8=value)
        system = CoupledSystem(params)
        xs = simulate(system, default_state(), lambda t, x: np.zeros(4), rainfall)
        results[value] = {
            "peak_damage": float(np.max(xs[:, 7])),
            "peak_infected": float(np.max(xs[:, 4])),
        }
    return results
