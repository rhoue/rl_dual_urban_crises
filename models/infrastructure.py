"""Infrastructure damage and cascading failures."""

from __future__ import annotations

from dataclasses import dataclass

from .utils_params import INFRASTRUCTURE_DEFAULTS


@dataclass
class InfrastructureParams:
    gamma8: float = INFRASTRUCTURE_DEFAULTS["gamma8"]
    varphi8: float = INFRASTRUCTURE_DEFAULTS["varphi8"]
    kappa9: float = INFRASTRUCTURE_DEFAULTS["kappa9"]
    rho9: float = INFRASTRUCTURE_DEFAULTS["rho9"]


def infrastructure_derivatives(x1: float, x2: float, x8: float, x9: float, u_p: float, p: InfrastructureParams) -> tuple[float, float]:
    damage_input = p.gamma8 * (x1 + 0.5 * x2)
    dx8 = damage_input - p.varphi8 * u_p * x8
    dx9 = p.kappa9 * x8 * (1.0 - x9) - p.rho9 * u_p * x9
    return dx8, dx9
