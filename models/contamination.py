"""Water contamination dynamics."""

from __future__ import annotations

from dataclasses import dataclass

from .utils_params import CONTAMINATION_DEFAULTS


@dataclass
class ContaminationParams:
    omega0: float = CONTAMINATION_DEFAULTS["omega0"]
    beta12: float = CONTAMINATION_DEFAULTS["beta12"]
    delta: float = CONTAMINATION_DEFAULTS["delta"]
    sigma_p: float = CONTAMINATION_DEFAULTS["sigma_p"]


def contamination_derivatives(x1: float, x2: float, x8: float, x12: float, x6: float, u_s: float, p: ContaminationParams) -> float:
    omega = p.omega0 * (x1 + 0.5 * x2 + x8)
    dx6 = omega + p.beta12 * x12 - p.delta * x6 - p.sigma_p * u_s * x6
    return dx6
