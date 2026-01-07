"""Centralized default parameters for the dual-crisis project."""

from __future__ import annotations

UI_DEFAULTS = {
    "seed_global": 0,
    "steps": 80,
    "rainfall_type": "stochastic",
    "baseline": 0.2,
    "pulse_prob": 0.15,
    "pulse_mean": 2.0,
    "pulse_std": 0.5,
    "intensity": 0.8,
    "decay": 0.98,
    "action_high": 1.0,
    "hybrid_alpha": 0.5,
    "rl_model_path": "",
}

HYDROLOGY_DEFAULTS = {
    "alpha1": 0.8,
    "k1": 0.3,
    "psi8_gain": 0.2,
    "theta1": 0.4,
    "phi1": 0.6,
    "phi2": 0.3,
    "h1": 0.05,
}

INFRASTRUCTURE_DEFAULTS = {
    "gamma8": 0.4,
    "varphi8": 0.5,
    "kappa9": 0.6,
    "rho9": 0.3,
}

DISPLACEMENT_DEFAULTS = {
    "p0": 0.1,
    "p1": 0.6,
    "r3": 0.2,
    "r3_u": 0.4,
    "ch": 1.0,
    "eta12": 0.6,
}

CONTAMINATION_DEFAULTS = {
    "omega0": 0.5,
    "beta12": 0.4,
    "delta": 0.3,
    "sigma_p": 0.7,
}

EPIDEMIC_DEFAULTS = {
    "beta0": 0.02,
    "beta6": 0.6,
    "beta12": 0.4,
    "rho3": 0.5,
    "pi": 0.05,
    "nu": 0.02,
    "gamma": 0.3,
    "sigma4": 0.3,
    "sigma4_u": 0.4,
    "mu_i": 0.2,
}

ECONOMY_DEFAULTS = {
    "psi_p": 0.4,
    "c_i": 0.3,
    "c_d": 0.2,
    "mu_f": 0.2,
    "mu_d": 0.1,
    "mu_i": 0.4,
}

COUPLED_DEFAULTS = {
    "rho7": 0.6,
    "v7": 0.4,
    "sigma4_x": 0.2,
    "rho14": 0.5,
    "k14": 0.2,
    "sanitation_decay": 0.2,
}
