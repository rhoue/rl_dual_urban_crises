"""Coupled system dynamics and integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from models.hydrology import HydrologyParams, hydrology_derivatives
from models.infrastructure import InfrastructureParams, infrastructure_derivatives
from models.displacement import DisplacementParams, displacement_derivatives
from models.contamination import ContaminationParams, contamination_derivatives
from models.epidemic import EpidemicParams, epidemic_derivatives
from models.economy_mortality import EconomyMortalityParams, economy_mortality_derivatives
from models.utils_params import COUPLED_DEFAULTS


STATE_NAMES = [
    "flood_depth",
    "flow",
    "displaced",
    "vulnerable",
    "infected",
    "contamination",
    "healthcare",
    "damage",
    "failures",
    "econ_loss",
    "deaths",
    "shelter_pressure",
    "sanitation",
    "personnel",
]
IDX = {name: i for i, name in enumerate(STATE_NAMES)}


@dataclass
class CoupledParams:
    hydrology: HydrologyParams = field(default_factory=HydrologyParams)
    infrastructure: InfrastructureParams = field(default_factory=InfrastructureParams)
    displacement: DisplacementParams = field(default_factory=DisplacementParams)
    contamination: ContaminationParams = field(default_factory=ContaminationParams)
    epidemic: EpidemicParams = field(default_factory=EpidemicParams)
    economy: EconomyMortalityParams = field(default_factory=EconomyMortalityParams)
    rho7: float = COUPLED_DEFAULTS["rho7"]
    v7: float = COUPLED_DEFAULTS["v7"]
    sigma4_x: float = COUPLED_DEFAULTS["sigma4_x"]
    rho14: float = COUPLED_DEFAULTS["rho14"]
    k14: float = COUPLED_DEFAULTS["k14"]
    sanitation_decay: float = COUPLED_DEFAULTS["sanitation_decay"]


def _healthcare_personnel_derivatives(x7: float, x9: float, x5: float, x14: float, x10: float, u_h: float, u_p: float, p: CoupledParams) -> tuple[float, float]:
    dx7 = p.rho7 * u_h - p.v7 * x9 * x7 + p.sigma4_x * x5 * (1.0 - x7)
    dx14 = p.rho14 * u_p - p.k14 * x10 * x14
    return dx7, dx14


def _sanitation_derivative(x13: float, u_s: float, decay: float) -> float:
    return u_s - decay * x13


class CoupledSystem:
    def __init__(self, params: CoupledParams | None = None, dt: float = 1.0):
        self.params = params or CoupledParams()
        self.dt = dt

    def derivative(self, state: np.ndarray, u: np.ndarray, rainfall: float) -> np.ndarray:
        x = state
        u_s, u_e, u_h, u_p = u
        p = self.params

        dx = np.zeros_like(x)
        dx[IDX["flood_depth"]], dx[IDX["flow"]] = hydrology_derivatives(
            x[IDX["flood_depth"]], x[IDX["flow"]], x[IDX["damage"]], rainfall, p.hydrology
        )
        dx[IDX["damage"]], dx[IDX["failures"]] = infrastructure_derivatives(
            x[IDX["flood_depth"]],
            x[IDX["flow"]],
            x[IDX["damage"]],
            x[IDX["failures"]],
            u_p,
            p.infrastructure,
        )
        dx[IDX["displaced"]], dx[IDX["shelter_pressure"]] = displacement_derivatives(
            x[IDX["flood_depth"]],
            x[IDX["damage"]],
            x[IDX["failures"]],
            x[IDX["displaced"]],
            x[IDX["shelter_pressure"]],
            u_e,
            p.displacement,
        )
        dx[IDX["contamination"]] = contamination_derivatives(
            x[IDX["flood_depth"]],
            x[IDX["flow"]],
            x[IDX["damage"]],
            x[IDX["shelter_pressure"]],
            x[IDX["contamination"]],
            u_s,
            p.contamination,
        )
        dx[IDX["vulnerable"]], dx[IDX["infected"]] = epidemic_derivatives(
            x[IDX["vulnerable"]],
            x[IDX["infected"]],
            x[IDX["contamination"]],
            x[IDX["shelter_pressure"]],
            x[IDX["healthcare"]],
            x[IDX["flood_depth"]],
            u_h,
            p.epidemic,
        )
        dx[IDX["healthcare"]], dx[IDX["personnel"]] = _healthcare_personnel_derivatives(
            x[IDX["healthcare"]],
            x[IDX["failures"]],
            x[IDX["infected"]],
            x[IDX["personnel"]],
            x[IDX["econ_loss"]],
            u_h,
            u_p,
            p,
        )
        dx[IDX["econ_loss"]], dx[IDX["deaths"]] = economy_mortality_derivatives(
            x[IDX["damage"]],
            x[IDX["failures"]],
            x[IDX["infected"]],
            x[IDX["displaced"]],
            x[IDX["shelter_pressure"]],
            x[IDX["healthcare"]],
            x[IDX["flood_depth"]],
            p.economy,
        )
        dx[IDX["sanitation"]] = _sanitation_derivative(
            x[IDX["sanitation"]], u_s, p.sanitation_decay
        )
        return dx

    def step(self, state: np.ndarray, u: np.ndarray, rainfall: float, method: str = "euler") -> np.ndarray:
        if method == "rk4":
            k1 = self.derivative(state, u, rainfall)
            k2 = self.derivative(state + 0.5 * self.dt * k1, u, rainfall)
            k3 = self.derivative(state + 0.5 * self.dt * k2, u, rainfall)
            k4 = self.derivative(state + self.dt * k3, u, rainfall)
            next_state = state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            next_state = state + self.dt * self.derivative(state, u, rainfall)
        return np.clip(next_state, 0.0, None)


def default_state() -> np.ndarray:
    x = np.zeros(len(STATE_NAMES), dtype=float)
    x[IDX["vulnerable"]] = 1.0
    return x


def simulate(system: CoupledSystem, x0: np.ndarray, policy: Callable[[int, np.ndarray], np.ndarray], rainfall: np.ndarray) -> np.ndarray:
    steps = len(rainfall)
    xs = np.zeros((steps + 1, len(STATE_NAMES)), dtype=float)
    xs[0] = x0
    x = x0
    for t in range(steps):
        u = policy(t, x)
        x = system.step(x, u, rainfall[t])
        xs[t + 1] = x
    return xs
