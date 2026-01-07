"""Displacement and shelter pressure dynamics."""

from __future__ import annotations

from dataclasses import dataclass

from .utils_params import DISPLACEMENT_DEFAULTS


@dataclass
class DisplacementParams:
    p0: float = DISPLACEMENT_DEFAULTS["p0"]
    p1: float = DISPLACEMENT_DEFAULTS["p1"]
    r3: float = DISPLACEMENT_DEFAULTS["r3"]
    r3_u: float = DISPLACEMENT_DEFAULTS["r3_u"]
    ch: float = DISPLACEMENT_DEFAULTS["ch"]
    eta12: float = DISPLACEMENT_DEFAULTS["eta12"]


def displacement_derivatives(x1: float, x8: float, x9: float, x3: float, x12: float, u_e: float, p: DisplacementParams) -> tuple[float, float]:
    g3 = x1 + x8 + x9
    dx3 = p.p0 + p.p1 * g3 - p.r3 * x3 - p.r3_u * u_e * x3
    dx12 = (x3 / max(p.ch, 1e-6)) - p.eta12 * u_e * x12
    return dx3, dx12
