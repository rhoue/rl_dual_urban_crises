"""Economic loss and mortality accounting."""

from __future__ import annotations

from dataclasses import dataclass

from .utils_params import ECONOMY_DEFAULTS


@dataclass
class EconomyMortalityParams:
    psi_p: float = ECONOMY_DEFAULTS["psi_p"]
    c_i: float = ECONOMY_DEFAULTS["c_i"]
    c_d: float = ECONOMY_DEFAULTS["c_d"]
    mu_f: float = ECONOMY_DEFAULTS["mu_f"]
    mu_d: float = ECONOMY_DEFAULTS["mu_d"]
    mu_i: float = ECONOMY_DEFAULTS["mu_i"]


def economy_mortality_derivatives(x8: float, x9: float, x5: float, x3: float, x12: float, x7: float, x1: float, p: EconomyMortalityParams) -> tuple[float, float]:
    dx10 = p.psi_p * (x8 + x9) + p.c_i * x5 + p.c_d * x3
    dx11 = p.mu_f * x1 + p.mu_d * (x3 + x12) + p.mu_i * x5 / (1.0 + x7)
    return dx10, dx11
