"""Hydrology dynamics for flood depth and flow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils_params import HYDROLOGY_DEFAULTS


@dataclass
class HydrologyParams:
    alpha1: float = HYDROLOGY_DEFAULTS["alpha1"]
    k1: float = HYDROLOGY_DEFAULTS["k1"]
    psi8_gain: float = HYDROLOGY_DEFAULTS["psi8_gain"]
    theta1: float = HYDROLOGY_DEFAULTS["theta1"]
    phi1: float = HYDROLOGY_DEFAULTS["phi1"]
    phi2: float = HYDROLOGY_DEFAULTS["phi2"]
    h1: float = HYDROLOGY_DEFAULTS["h1"]


def hydrology_derivatives(x1: float, x2: float, x8: float, rainfall: float, p: HydrologyParams) -> tuple[float, float]:
    psi8 = p.psi8_gain * x8
    dx1 = p.alpha1 * rainfall - p.k1 * x1 - psi8 * x1
    dx2 = -p.theta1 * x2 + p.phi1 * x1 - p.phi2 * x8 + p.h1 * x1 * x2
    return dx1, dx2
