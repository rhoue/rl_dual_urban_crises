"""Epidemic dynamics with contamination and crowding effects."""

from __future__ import annotations

from dataclasses import dataclass

from .utils_params import EPIDEMIC_DEFAULTS


@dataclass
class EpidemicParams:
    beta0: float = EPIDEMIC_DEFAULTS["beta0"]
    beta6: float = EPIDEMIC_DEFAULTS["beta6"]
    beta12: float = EPIDEMIC_DEFAULTS["beta12"]
    rho3: float = EPIDEMIC_DEFAULTS["rho3"]
    pi: float = EPIDEMIC_DEFAULTS["pi"]
    nu: float = EPIDEMIC_DEFAULTS["nu"]
    gamma: float = EPIDEMIC_DEFAULTS["gamma"]
    sigma4: float = EPIDEMIC_DEFAULTS["sigma4"]
    sigma4_u: float = EPIDEMIC_DEFAULTS["sigma4_u"]
    mu_i: float = EPIDEMIC_DEFAULTS["mu_i"]


def _crowding(x12: float) -> float:
    return x12 / (1.0 + x12)


def epidemic_derivatives(x4: float, x5: float, x6: float, x12: float, x7: float, x1: float, u_h: float, p: EpidemicParams) -> tuple[float, float]:
    lam = p.beta0 + p.beta6 * (p.rho3 * x1) * x6 + p.beta12 * _crowding(x12)
    dx4 = -lam * x4 + p.pi - p.nu * x4
    recovery = p.gamma + p.sigma4 * x7 + p.sigma4_u * u_h
    mortality = p.mu_i * x5 / (1.0 + x7)
    dx5 = lam * x4 - recovery * x5 - mortality
    return dx4, dx5
