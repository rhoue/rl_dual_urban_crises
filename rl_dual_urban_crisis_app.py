"""Streamlit app for interactive dual-crisis simulation."""

from __future__ import annotations

import sys
import random
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import json

from control.mpc_baseline import OneStepMPC
from control.rule_based import RuleBasedPolicy
from models.coupled_system import CoupledParams, CoupledSystem, IDX, STATE_NAMES, default_state
from scenarios.rainfall_generator import RainfallScenario, deterministic_rainfall
from models.utils_params import (
    CONTAMINATION_DEFAULTS,
    COUPLED_DEFAULTS,
    DISPLACEMENT_DEFAULTS,
    ECONOMY_DEFAULTS,
    EPIDEMIC_DEFAULTS,
    HYDROLOGY_DEFAULTS,
    INFRASTRUCTURE_DEFAULTS,
    UI_DEFAULTS,
)
from utils_ui import help_text


st.set_page_config(page_title="Dual Crisis Digital Twin", layout="wide")

st.title("Dual Crisis Digital Twin")
st.caption("Interactive simulation for flood + epidemic dynamics with policy controls.")

with st.sidebar:
    st.header("Scenario")
    seed_global = st.number_input(
        "Global seed",
        value=UI_DEFAULTS["seed_global"],
        step=1,
        help=help_text("Global seed"),
    )
    steps = st.slider(
        "Horizon (steps)",
        min_value=20,
        max_value=200,
        value=UI_DEFAULTS["steps"],
        step=10,
        help=help_text("Horizon (steps)"),
    )
    rainfall_options = ["stochastic", "deterministic"]
    rainfall_type = st.selectbox(
        "Rainfall type",
        rainfall_options,
        index=rainfall_options.index(UI_DEFAULTS["rainfall_type"]),
        help=help_text("Rainfall type"),
    )
    if rainfall_type == "stochastic":
        baseline = st.slider(
            "Baseline rainfall",
            0.0,
            1.0,
            UI_DEFAULTS["baseline"],
            0.05,
            help=help_text("Baseline rainfall"),
        )
        pulse_prob = st.slider(
            "Pulse probability",
            0.0,
            0.5,
            UI_DEFAULTS["pulse_prob"],
            0.01,
            help=help_text("Pulse probability"),
        )
        pulse_mean = st.slider(
            "Pulse mean",
            0.5,
            4.0,
            UI_DEFAULTS["pulse_mean"],
            0.1,
            help=help_text("Pulse mean"),
        )
        pulse_std = st.slider(
            "Pulse std",
            0.1,
            2.0,
            UI_DEFAULTS["pulse_std"],
            0.1,
            help=help_text("Pulse std"),
        )
        seed = st.number_input(
            "Rainfall seed (optional)",
            value=int(seed_global),
            step=1,
            help=help_text("Rainfall seed (optional)"),
        )
    else:
        intensity = st.slider(
            "Initial intensity",
            0.1,
            3.0,
            UI_DEFAULTS["intensity"],
            0.1,
            help=help_text("Initial intensity"),
        )
        decay = st.slider(
            "Decay",
            0.90,
            1.00,
            UI_DEFAULTS["decay"],
            0.005,
            help=help_text("Decay"),
        )

    st.header("Policy")
    action_high = st.slider(
        "Action max",
        0.2,
        2.0,
        UI_DEFAULTS["action_high"],
        0.1,
        help=help_text("Action max"),
    )
    hybrid_alpha = st.slider(
        "Hybrid weight (MPC)",
        0.0,
        1.0,
        UI_DEFAULTS["hybrid_alpha"],
        0.05,
        help=help_text("Hybrid weight (MPC)"),
    )
    with st.expander("RL policy settings", expanded=True):
        rl_model_path = st.text_input(
            "PPO model path",
            value=UI_DEFAULTS["rl_model_path"],
            help=help_text("RL model path (optional)"),
        )
    with st.expander("Policy recommender settings", expanded=False):
        reco_horizon = st.slider(
            "Recommender horizon",
            min_value=5,
            max_value=80,
            value=min(UI_DEFAULTS["steps"], 30),
            step=5,
            help=help_text("Recommender horizon"),
        )
        reco_levels = st.multiselect(
            "Candidate action levels (fractions of max)",
            options=[0.0, 0.25, 0.5, 0.75, 1.0],
            default=[0.0, 0.5, 1.0],
            help=help_text("Candidate action levels (fractions of max)"),
        )
        w_mortality = st.slider(
            "Weight: mortality", 0.1, 5.0, 2.0, 0.1, help=help_text("Weight: mortality")
        )
        w_econ = st.slider(
            "Weight: economic loss",
            0.1,
            5.0,
            1.0,
            0.1,
            help=help_text("Weight: economic loss"),
        )
        w_services = st.slider(
            "Weight: service disruption",
            0.1,
            5.0,
            1.0,
            0.1,
            help=help_text("Weight: service disruption"),
        )
        w_action = st.slider(
            "Weight: action cost",
            0.0,
            5.0,
            0.5,
            0.1,
            help=help_text("Weight: action cost"),
        )
    with st.expander("Action cost configuration", expanded=False):
        cost_sanitation = st.slider(
            "Cost: sanitation (u_S)", 0.0, 5.0, 1.0, 0.1, help=help_text("Cost: sanitation (u_S)")
        )
        cost_evacuation = st.slider(
            "Cost: evacuation (u_E)", 0.0, 5.0, 1.0, 0.1, help=help_text("Cost: evacuation (u_E)")
        )
        cost_healthcare = st.slider(
            "Cost: healthcare (u_H)", 0.0, 5.0, 1.0, 0.1, help=help_text("Cost: healthcare (u_H)")
        )
        cost_personnel = st.slider(
            "Cost: personnel (u_P)", 0.0, 5.0, 1.0, 0.1, help=help_text("Cost: personnel (u_P)")
        )
    with st.expander("Constraint settings", expanded=False):
        budget_enabled = st.checkbox(
            "Enable budget constraint", value=True, help=help_text("Enable budget constraint")
        )
        budget_max = st.slider(
            "Budget max", 0.5, 5.0, 2.0, 0.1, help=help_text("Budget max")
        )
        b_s = st.slider(
            "Budget weight: sanitation",
            0.0,
            2.0,
            1.0,
            0.1,
            help=help_text("Budget weight: sanitation"),
        )
        b_e = st.slider(
            "Budget weight: evacuation",
            0.0,
            2.0,
            1.0,
            0.1,
            help=help_text("Budget weight: evacuation"),
        )
        b_h = st.slider(
            "Budget weight: healthcare",
            0.0,
            2.0,
            1.0,
            0.1,
            help=help_text("Budget weight: healthcare"),
        )
        b_p = st.slider(
            "Budget weight: personnel",
            0.0,
            2.0,
            1.0,
            0.1,
            help=help_text("Budget weight: personnel"),
        )
        safety_enabled = st.checkbox(
            "Enable safety constraints", value=False, help=help_text("Enable safety constraints")
        )
        max_contamination = st.slider(
            "Max contamination", 0.1, 5.0, 1.5, 0.1, help=help_text("Max contamination")
        )
        max_shelter = st.slider(
            "Max shelter pressure", 0.1, 5.0, 1.5, 0.1, help=help_text("Max shelter pressure")
        )
    with st.expander("Stress test settings", expanded=False):
        stress_ensembles = st.slider(
            "Ensembles", 3, 30, 10, 1, help=help_text("Ensembles")
        )
        stress_seed = st.number_input(
            "Stress test seed", value=int(seed_global), step=1, help=help_text("Stress test seed")
        )
        stress_pulse_prob = st.slider(
            "Pulse probability (stress)",
            0.05,
            0.5,
            0.2,
            0.01,
            help=help_text("Pulse probability (stress)"),
        )
        stress_pulse_mean = st.slider(
            "Pulse mean (stress)",
            0.5,
            4.0,
            2.2,
            0.1,
            help=help_text("Pulse mean (stress)"),
        )
        stress_pulse_std = st.slider(
            "Pulse std (stress)",
            0.1,
            2.0,
            0.6,
            0.1,
            help=help_text("Pulse std (stress)"),
        )
        fragility_gain = st.slider(
            "Fragility gain", 0.0, 2.0, 0.5, 0.1, help=help_text("Fragility gain")
        )
    with st.expander("Counterfactual settings", expanded=False):
        cf_base_policy = st.selectbox(
            "Baseline policy",
            options=["rule_based", "mpc", "hybrid"],
            index=0,
            help=help_text("Baseline policy"),
        )
        cf_lever = st.selectbox(
            "Lever to increase",
            options=["sanitation", "evacuation", "healthcare", "personnel"],
            index=0,
            help=help_text("Lever to increase"),
        )
        cf_increase_pct = st.slider(
            "Increase (%)", 5, 50, 10, 5, help=help_text("Increase (%)")
        )
        cf_horizon = st.slider(
            "Counterfactual horizon", 5, 80, 30, 5, help=help_text("Counterfactual horizon")
        )
    with st.expander("Explainability settings", expanded=False):
        expl_horizon = st.slider(
            "Explainability horizon", 5, 80, 30, 5, help=help_text("Explainability horizon")
        )
    with st.expander("Alerting settings", expanded=False):
        alert_enabled = st.checkbox(
            "Enable alerting", value=True, help=help_text("Enable alerting")
        )
        alert_infected = st.slider(
            "Infected threshold", 0.1, 5.0, 1.0, 0.1, help=help_text("Infected threshold")
        )
        alert_contamination = st.slider(
            "Contamination threshold",
            0.1,
            5.0,
            1.0,
            0.1,
            help=help_text("Contamination threshold"),
        )
        alert_displaced = st.slider(
            "Displaced threshold", 0.1, 5.0, 1.0, 0.1, help=help_text("Displaced threshold")
        )
        alert_horizon = st.slider(
            "Alert horizon", 5, 80, 30, 5, help=help_text("Alert horizon")
        )
    with st.expander("Policy analysis settings", expanded=False):
        pareto_max_points = st.slider("Pareto max points", 20, 200, 80, 10)
        cluster_k = st.slider("Policy clusters (k)", 2, 8, 4, 1)
        cluster_iters = st.slider("Cluster iterations", 5, 50, 20, 5)
        slice_bins = st.slider("Slice bins", 3, 12, 5, 1)
        if st.button("Run Pareto analysis (sidebar)", key="run_pareto_sidebar"):
            st.session_state["trigger_pareto"] = True
            st.success("Pareto analysis queued. Open the Pareto tab to view results.")
        if st.button("Run clustering (sidebar)", key="run_cluster_sidebar"):
            st.session_state["trigger_cluster"] = True
        if st.button("Run composite slice (sidebar)", key="run_slice_sidebar"):
            st.session_state["trigger_slice"] = True

    st.header("Hydrology")
    alpha1 = st.slider(
        "alpha1", 0.1, 2.0, HYDROLOGY_DEFAULTS["alpha1"], 0.05, help=help_text("alpha1")
    )
    k1 = st.slider(
        "k1", 0.05, 1.0, HYDROLOGY_DEFAULTS["k1"], 0.05, help=help_text("k1")
    )
    psi8_gain = st.slider(
        "psi8_gain",
        0.0,
        1.0,
        HYDROLOGY_DEFAULTS["psi8_gain"],
        0.05,
        help=help_text("psi8_gain"),
    )
    theta1 = st.slider(
        "theta1", 0.1, 1.0, HYDROLOGY_DEFAULTS["theta1"], 0.05, help=help_text("theta1")
    )
    phi1 = st.slider(
        "phi1", 0.1, 1.0, HYDROLOGY_DEFAULTS["phi1"], 0.05, help=help_text("phi1")
    )
    phi2 = st.slider(
        "phi2", 0.1, 1.0, HYDROLOGY_DEFAULTS["phi2"], 0.05, help=help_text("phi2")
    )

    st.header("Infrastructure")
    gamma8 = st.slider(
        "gamma8",
        0.1,
        1.0,
        INFRASTRUCTURE_DEFAULTS["gamma8"],
        0.05,
        help=help_text("gamma8"),
    )
    varphi8 = st.slider(
        "varphi8",
        0.1,
        1.0,
        INFRASTRUCTURE_DEFAULTS["varphi8"],
        0.05,
        help=help_text("varphi8"),
    )
    kappa9 = st.slider(
        "kappa9",
        0.1,
        1.0,
        INFRASTRUCTURE_DEFAULTS["kappa9"],
        0.05,
        help=help_text("kappa9"),
    )
    rho9 = st.slider(
        "rho9",
        0.1,
        1.0,
        INFRASTRUCTURE_DEFAULTS["rho9"],
        0.05,
        help=help_text("rho9"),
    )

    st.header("Displacement")
    p0 = st.slider(
        "p0", 0.0, 0.5, DISPLACEMENT_DEFAULTS["p0"], 0.02, help=help_text("p0")
    )
    p1 = st.slider(
        "p1", 0.1, 1.0, DISPLACEMENT_DEFAULTS["p1"], 0.05, help=help_text("p1")
    )
    r3 = st.slider(
        "r3", 0.05, 1.0, DISPLACEMENT_DEFAULTS["r3"], 0.05, help=help_text("r3")
    )
    r3_u = st.slider(
        "r3_u", 0.05, 1.0, DISPLACEMENT_DEFAULTS["r3_u"], 0.05, help=help_text("r3_u")
    )
    eta12 = st.slider(
        "eta12",
        0.1,
        1.0,
        DISPLACEMENT_DEFAULTS["eta12"],
        0.05,
        help=help_text("eta12"),
    )

    st.header("Contamination")
    omega0 = st.slider(
        "omega0",
        0.1,
        1.5,
        CONTAMINATION_DEFAULTS["omega0"],
        0.05,
        help=help_text("omega0"),
    )
    beta12_c = st.slider(
        "beta12_c",
        0.1,
        1.0,
        CONTAMINATION_DEFAULTS["beta12"],
        0.05,
        help=help_text("beta12_c"),
    )
    delta = st.slider(
        "delta",
        0.05,
        1.0,
        CONTAMINATION_DEFAULTS["delta"],
        0.05,
        help=help_text("delta"),
    )
    sigma_p = st.slider(
        "sigma_p",
        0.1,
        1.0,
        CONTAMINATION_DEFAULTS["sigma_p"],
        0.05,
        help=help_text("sigma_p"),
    )

    st.header("Epidemic")
    beta0 = st.slider(
        "beta0",
        0.0,
        0.1,
        EPIDEMIC_DEFAULTS["beta0"],
        0.005,
        help=help_text("beta0"),
    )
    beta6 = st.slider(
        "beta6",
        0.1,
        1.0,
        EPIDEMIC_DEFAULTS["beta6"],
        0.05,
        help=help_text("beta6"),
    )
    beta12_e = st.slider(
        "beta12_e",
        0.1,
        1.0,
        EPIDEMIC_DEFAULTS["beta12"],
        0.05,
        help=help_text("beta12_e"),
    )
    rho3 = st.slider(
        "rho3", 0.1, 1.0, EPIDEMIC_DEFAULTS["rho3"], 0.05, help=help_text("rho3")
    )
    pi = st.slider(
        "pi", 0.0, 0.2, EPIDEMIC_DEFAULTS["pi"], 0.01, help=help_text("pi")
    )
    nu = st.slider(
        "nu", 0.0, 0.2, EPIDEMIC_DEFAULTS["nu"], 0.01, help=help_text("nu")
    )
    gamma = st.slider(
        "gamma",
        0.1,
        1.0,
        EPIDEMIC_DEFAULTS["gamma"],
        0.05,
        help=help_text("gamma"),
    )
    sigma4 = st.slider(
        "sigma4",
        0.1,
        1.0,
        EPIDEMIC_DEFAULTS["sigma4"],
        0.05,
        help=help_text("sigma4"),
    )
    sigma4_u = st.slider(
        "sigma4_u",
        0.1,
        1.0,
        EPIDEMIC_DEFAULTS["sigma4_u"],
        0.05,
        help=help_text("sigma4_u"),
    )
    mu_i = st.slider(
        "mu_i",
        0.1,
        1.0,
        EPIDEMIC_DEFAULTS["mu_i"],
        0.05,
        help=help_text("mu_i"),
    )

    st.header("Economy & Mortality")
    psi_p = st.slider(
        "psi_p",
        0.1,
        1.0,
        ECONOMY_DEFAULTS["psi_p"],
        0.05,
        help=help_text("psi_p"),
    )
    c_i = st.slider(
        "c_i", 0.1, 1.0, ECONOMY_DEFAULTS["c_i"], 0.05, help=help_text("c_i")
    )
    c_d = st.slider(
        "c_d", 0.1, 1.0, ECONOMY_DEFAULTS["c_d"], 0.05, help=help_text("c_d")
    )
    mu_f = st.slider(
        "mu_f",
        0.1,
        1.0,
        ECONOMY_DEFAULTS["mu_f"],
        0.05,
        help=help_text("mu_f"),
    )
    mu_d = st.slider(
        "mu_d",
        0.1,
        1.0,
        ECONOMY_DEFAULTS["mu_d"],
        0.05,
        help=help_text("mu_d"),
    )
    mu_i2 = st.slider(
        "mu_i2",
        0.1,
        1.0,
        ECONOMY_DEFAULTS["mu_i"],
        0.05,
        help=help_text("mu_i2"),
    )

    st.header("Operational")
    rho7 = st.slider(
        "rho7", 0.1, 1.0, COUPLED_DEFAULTS["rho7"], 0.05, help=help_text("rho7")
    )
    v7 = st.slider(
        "v7", 0.1, 1.0, COUPLED_DEFAULTS["v7"], 0.05, help=help_text("v7")
    )
    sigma4_x = st.slider(
        "sigma4_x",
        0.1,
        1.0,
        COUPLED_DEFAULTS["sigma4_x"],
        0.05,
        help=help_text("sigma4_x"),
    )
    rho14 = st.slider(
        "rho14", 0.1, 1.0, COUPLED_DEFAULTS["rho14"], 0.05, help=help_text("rho14")
    )
    k14 = st.slider(
        "k14", 0.05, 0.5, COUPLED_DEFAULTS["k14"], 0.05, help=help_text("k14")
    )
    sanitation_decay = st.slider(
        "sanitation_decay",
        0.05,
        0.6,
        COUPLED_DEFAULTS["sanitation_decay"],
        0.05,
        help=help_text("sanitation_decay"),
    )


np.random.seed(int(seed_global))
random.seed(int(seed_global))

params = CoupledParams()
params.hydrology.alpha1 = alpha1
params.hydrology.k1 = k1
params.hydrology.psi8_gain = psi8_gain
params.hydrology.theta1 = theta1
params.hydrology.phi1 = phi1
params.hydrology.phi2 = phi2

params.infrastructure.gamma8 = gamma8
params.infrastructure.varphi8 = varphi8
params.infrastructure.kappa9 = kappa9
params.infrastructure.rho9 = rho9

params.displacement.p0 = p0
params.displacement.p1 = p1
params.displacement.r3 = r3
params.displacement.r3_u = r3_u
params.displacement.eta12 = eta12

params.contamination.omega0 = omega0
params.contamination.beta12 = beta12_c
params.contamination.delta = delta
params.contamination.sigma_p = sigma_p

params.epidemic.beta0 = beta0
params.epidemic.beta6 = beta6
params.epidemic.beta12 = beta12_e
params.epidemic.rho3 = rho3
params.epidemic.pi = pi
params.epidemic.nu = nu
params.epidemic.gamma = gamma
params.epidemic.sigma4 = sigma4
params.epidemic.sigma4_u = sigma4_u
params.epidemic.mu_i = mu_i

params.economy.psi_p = psi_p
params.economy.c_i = c_i
params.economy.c_d = c_d
params.economy.mu_f = mu_f
params.economy.mu_d = mu_d
params.economy.mu_i = mu_i2

params.rho7 = rho7
params.v7 = v7
params.sigma4_x = sigma4_x
params.rho14 = rho14
params.k14 = k14
params.sanitation_decay = sanitation_decay

system = CoupledSystem(params)

if rainfall_type == "stochastic":
    rainfall = RainfallScenario(
        steps=steps,
        baseline=baseline,
        pulse_prob=pulse_prob,
        pulse_mean=pulse_mean,
        pulse_std=pulse_std,
        rng_seed=seed,
    ).generate()
else:
    rainfall = deterministic_rainfall(steps=steps, intensity=intensity, decay=decay)

rule_policy = RuleBasedPolicy(action_high=action_high)
mpc_policy = OneStepMPC(system)


def _evaluate_constant_action(action: np.ndarray, rainfall_override: np.ndarray | None = None) -> dict:
    horizon = min(reco_horizon, len(rainfall))
    rain_series = rainfall_override if rainfall_override is not None else rainfall
    state = default_state()
    total_service = 0.0
    max_contamination_obs = 0.0
    max_shelter_obs = 0.0
    for t in range(horizon):
        state = system.step(state, action, float(rain_series[t]))
        total_service += state[IDX["damage"]] + state[IDX["failures"]]
        max_contamination_obs = max(max_contamination_obs, state[IDX["contamination"]])
        max_shelter_obs = max(max_shelter_obs, state[IDX["shelter_pressure"]])
    deaths_delta = state[IDX["deaths"]]
    econ_delta = state[IDX["econ_loss"]]
    service_mean = total_service / max(horizon, 1)
    action_cost = float(
        cost_sanitation * abs(action[0])
        + cost_evacuation * abs(action[1])
        + cost_healthcare * abs(action[2])
        + cost_personnel * abs(action[3])
    )
    score = (
        w_mortality * deaths_delta
        + w_econ * econ_delta
        + w_services * service_mean
        + w_action * action_cost
    )
    return {
        "deaths_delta": deaths_delta,
        "econ_delta": econ_delta,
        "service_mean": service_mean,
        "max_contamination": max_contamination_obs,
        "max_shelter": max_shelter_obs,
        "action_cost": action_cost,
        "score": score,
    }


def _generate_candidates(levels: list[float]) -> list[dict]:
    candidates = []
    for u_s in levels:
        for u_e in levels:
            for u_h in levels:
                for u_p in levels:
                    action = np.array([u_s, u_e, u_h, u_p], dtype=float)
                    metrics = _evaluate_constant_action(action)
                    candidates.append(
                        {
                            "u_s": u_s,
                            "u_e": u_e,
                            "u_h": u_h,
                            "u_p": u_p,
                            **metrics,
                        }
                    )
    return candidates

config_snapshot = {
    "seed_global": int(seed_global),
    "steps": int(steps),
    "rainfall_type": rainfall_type,
    "rainfall_params": {
        "baseline": float(baseline) if rainfall_type == "stochastic" else None,
        "pulse_prob": float(pulse_prob) if rainfall_type == "stochastic" else None,
        "pulse_mean": float(pulse_mean) if rainfall_type == "stochastic" else None,
        "pulse_std": float(pulse_std) if rainfall_type == "stochastic" else None,
        "seed": int(seed) if rainfall_type == "stochastic" else None,
        "intensity": float(intensity) if rainfall_type == "deterministic" else None,
        "decay": float(decay) if rainfall_type == "deterministic" else None,
    },
    "policy": {
        "action_high": float(action_high),
        "hybrid_alpha": float(hybrid_alpha),
        "rl_model_path": rl_model_path,
    },
    "params": {
        "hydrology": params.hydrology.__dict__,
        "infrastructure": params.infrastructure.__dict__,
        "displacement": params.displacement.__dict__,
        "contamination": params.contamination.__dict__,
        "epidemic": params.epidemic.__dict__,
        "economy": params.economy.__dict__,
        "rho7": params.rho7,
        "v7": params.v7,
        "sigma4_x": params.sigma4_x,
        "rho14": params.rho14,
        "k14": params.k14,
        "sanitation_decay": params.sanitation_decay,
    },
}
config_json = json.dumps(config_snapshot, sort_keys=True)

def _run_policy(policy_name: str) -> tuple[np.ndarray, np.ndarray]:
    state = default_state()
    traj = np.zeros((steps + 1, len(STATE_NAMES)), dtype=float)
    actions = np.zeros((steps, 4), dtype=float)
    traj[0] = state

    if policy_name == "rule_based":
        policy = rule_policy
    elif policy_name == "mpc":
        policy = mpc_policy
    elif policy_name == "zero":
        policy = lambda t, state: np.zeros(4, dtype=float)
    elif policy_name == "hybrid":
        def _hybrid(t: int, state: np.ndarray) -> np.ndarray:
            action_rule = rule_policy(t, state)
            action_mpc = mpc_policy(t, state, float(rainfall[t]))
            action = hybrid_alpha * action_mpc + (1.0 - hybrid_alpha) * action_rule
            return np.clip(action, 0.0, action_high)
        policy = _hybrid
    elif policy_name == "rl":
        if not rl_model_path:
            policy = mpc_policy
        else:
            try:
                from stable_baselines3 import PPO
            except ImportError as exc:  # pragma: no cover
                raise ImportError("stable-baselines3 is required for RL policy") from exc
            model = PPO.load(rl_model_path)
            def _rl(t: int, state: np.ndarray) -> np.ndarray:
                action, _ = model.predict(state, deterministic=True)
                return np.clip(np.asarray(action, dtype=float), 0.0, action_high)
            policy = _rl
    else:
        policy = rule_policy

    for t in range(steps):
        if isinstance(policy, OneStepMPC):
            action = policy(t, state, float(rainfall[t]))
        else:
            action = policy(t, state)
        actions[t] = action
        state = system.step(state, action, float(rainfall[t]))
        traj[t + 1] = state
    return traj, actions


def _run_policy_with_lever(policy_name: str, lever_idx: int, increase_pct: float, horizon: int) -> dict:
    state = default_state()
    steps_local = min(horizon, len(rainfall))
    for t in range(steps_local):
        if policy_name == "mpc":
            action = mpc_policy(t, state, float(rainfall[t]))
        elif policy_name == "hybrid":
            action_rule = rule_policy(t, state)
            action_mpc = mpc_policy(t, state, float(rainfall[t]))
            action = hybrid_alpha * action_mpc + (1.0 - hybrid_alpha) * action_rule
        else:
            action = rule_policy(t, state)
        action = np.asarray(action, dtype=float)
        action[lever_idx] = min(action_high, action[lever_idx] * (1.0 + increase_pct))
        state = system.step(state, action, float(rainfall[t]))
    return {
        "deaths_delta": state[IDX["deaths"]],
        "econ_delta": state[IDX["econ_loss"]],
        "infected": state[IDX["infected"]],
        "contamination": state[IDX["contamination"]],
        "displaced": state[IDX["displaced"]],
    }


def _run_policy_with_sensitivity(policy_name: str, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    state = default_state()
    steps_local = min(horizon, len(rainfall))
    traj = np.zeros((steps_local + 1, len(STATE_NAMES)), dtype=float)
    traj[0] = state
    for t in range(steps_local):
        if policy_name == "mpc":
            action = mpc_policy(t, state, float(rainfall[t]))
        elif policy_name == "hybrid":
            action_rule = rule_policy(t, state)
            action_mpc = mpc_policy(t, state, float(rainfall[t]))
            action = hybrid_alpha * action_mpc + (1.0 - hybrid_alpha) * action_rule
        else:
            action = rule_policy(t, state)
        state = system.step(state, np.asarray(action, dtype=float), float(rainfall[t]))
        traj[t + 1] = state
    return traj, np.asarray(rainfall[: steps_local + 1], dtype=float)


def _score_action_from_state(state: np.ndarray, action: np.ndarray, horizon: int) -> float:
    horizon = max(1, min(horizon, len(rainfall)))
    x = state.copy()
    total_service = 0.0
    for t in range(horizon):
        x = system.step(x, action, float(rainfall[t]))
        total_service += x[IDX["damage"]] + x[IDX["failures"]]
    deaths_delta = x[IDX["deaths"]]
    econ_delta = x[IDX["econ_loss"]]
    service_mean = total_service / horizon
    action_cost = float(
        cost_sanitation * abs(action[0])
        + cost_evacuation * abs(action[1])
        + cost_healthcare * abs(action[2])
        + cost_personnel * abs(action[3])
    )
    score = (
        w_mortality * deaths_delta
        + w_econ * econ_delta
        + w_services * service_mean
        + w_action * action_cost
    )
    return float(score)


def _pareto_front(rows: list[dict]) -> list[dict]:
    front = []
    for i, row in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            if (
                other["deaths_delta"] <= row["deaths_delta"]
                and other["econ_delta"] <= row["econ_delta"]
                and other["service_mean"] <= row["service_mean"]
                and (
                    other["deaths_delta"] < row["deaths_delta"]
                    or other["econ_delta"] < row["econ_delta"]
                    or other["service_mean"] < row["service_mean"]
                )
            ):
                dominated = True
                break
        if not dominated:
            front.append(row)
    return front


def _kmeans(actions: np.ndarray, k: int, iters: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = actions.shape[0]
    if n == 0:
        return np.empty((0, actions.shape[1])), np.empty((0,), dtype=int)
    k = min(k, n)
    indices = rng.choice(n, size=k, replace=False)
    centroids = actions[indices].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dists = np.linalg.norm(actions[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        for i in range(k):
            members = actions[labels == i]
            if len(members) > 0:
                centroids[i] = members.mean(axis=0)
    return centroids, labels


def _composite_indices(action: np.ndarray) -> dict:
    return {
        "A_hydro": action[0] + action[3],
        "A_social": action[1],
        "A_health": action[2],
        "A_total": float(np.sum(action)),
    }

st.subheader("📚 Theoretical Background")
theory_tabs = st.tabs(
    [
        "Optimal Control of Convergent Urban Crises",
        "State Variables",
        "Hydrology + Infrastructure",
        "Displacement + Contamination",
        "Epidemics + Outcomes",
    ]
)

with theory_tabs[0]:
    st.markdown(
        """**Why optimize convergent urban crises?**

Urban systems face compound risks—floods, contamination, displacement, and epidemics—requiring trade-offs between
mortality reduction, service continuity, and economic stability under limited resources.

**Current practice**
- Threshold rules and fixed playbooks
- Hazard models studied in isolation
- Siloed control policies with limited cross-sector coordination

**Opportunity**
With simulation, scenario generation, and multi-objective evaluation, we can test coordinated policies that
respect simplified physical dynamics and explore trade-offs under uncertainty.

With the rise of:
- Scenario generators for rainfall and fragility
- Epidemiological and contamination modeling
- AI and optimization tools for policy search

The question becomes:
"""
    )
    st.info(
        "\"Can we evaluate and compare candidate policies in simulation, using consistent dynamics and transparent objectives?\""
    )
    st.markdown(
        """**Proposed approach (current scope)**
- Coupled, interpretable dynamical models (hydrology, displacement, contamination, epidemics, infrastructure, economy)
- Explicit operational levers (sanitation, evacuation, healthcare, personnel)
- Simulation-based baselines, stress tests, and analysis tools
- Optional RL policy evaluation via externally trained models
"""
    )
    st.markdown(
        """**Problem hypotheses**

**Flood, contamination & infection**
- **H1**: Increasing flood intensity has no effect on the contamination variable or peak infected variable.
- **H2**: Increasing flood intensity increases the contamination variable or peak infected variable.

**Shelter overcrowding amplifies transmission**
- **H3**: For fixed contamination variable, outbreak of the infected variable is independent of shelter pressure.
- **H4**: For fixed contamination variable, larger shelter pressure increases outbreak of the infected variable.
"""
    )

with theory_tabs[1]:
    st.markdown(
        """**State and controls**

State vector:
$$
X(t)=\\big[X_1,\\dots,X_{14}\\big]^\\top
$$

Key state variables:
- $X_1$ flood depth, $X_2$ flow, $X_3$ displaced, $X_4$ vulnerable, $X_5$ infected
- $X_6$ contamination, $X_7$ healthcare capacity, $X_8$ damage, $X_9$ failures
- $X_{10}$ economic loss, $X_{11}$ deaths, $X_{12}$ shelter pressure
- $X_{13}$ sanitation, $X_{14}$ personnel

Control vector:
$$
u(t)=\\big[u_S, u_E, u_H, u_P\\big]
$$
for sanitation, evacuation, healthcare reinforcement, and personnel mobilization.
"""
    )

with theory_tabs[2]:
    st.markdown(
        """**Hydrology and infrastructure**

Hydrology:
$$
\\dot{X}_1 = \\alpha_1 R(t) - k_1 X_1 - \\psi_8(X_8) X_1
$$
$$
\\dot{X}_2 = -\\theta_1 X_2 + \\phi_1(X_1) - \\phi_2(X_8) + h_1(X_1,X_2)
$$

Infrastructure:
$$
\\dot{X}_8 = \\gamma_8(X_1,X_2) - \\varphi_8(u_P) X_8
$$
$$
\\dot{X}_9 = \\kappa_9(X_8)(1-X_9) - \\rho_9(u_P) X_9
$$

**Variables**
- $X_1$: flood depth, $X_2$: flow, $X_8$: physical damage, $X_9$: critical failures, $R(t)$: rainfall driver
- $u_P$: personnel/logistics effort

**Parameters**
- $\\alpha_1$: rainfall-to-depth gain, $k_1$: depth dissipation
- $\\psi_8(\\cdot)$: drainage impairment from damage
- $\\theta_1$: flow dissipation, $\\phi_1,\\phi_2$: coupling terms, $h_1$: nonlinear interaction
- $\\gamma_8$: damage growth rate, $\\varphi_8$: repair effectiveness
- $\\kappa_9$: failure escalation, $\\rho_9$: recovery rate
"""
    )

with theory_tabs[3]:
    st.markdown(
        """**Displacement and contamination**

Displacement:
$$
\\dot{X}_3 = P_0 + P_1 g_3(X_1,X_8,X_9) - r_3(X_3,u_E)
$$
$$
\\dot{X}_{12} = \\frac{X_3}{CH} - \\eta_{12}(u_E) X_{12}
$$

Contamination:
$$
\\dot{X}_6 = \\omega(X_1,X_2,X_8) + \\beta_{12} X_{12} - \\delta X_6 - \\sigma_p(u_S) X_6
$$

**Variables**
- $X_3$: displaced population, $X_{12}$: shelter pressure, $X_6$: contamination
- $X_1, X_2, X_8$: flood/flow/damage drivers, $u_E$: evacuation, $u_S$: sanitation

**Parameters**
- $P_0$: baseline displacement, $P_1$: flood-driven displacement gain
- $r_3$: resettlement rate, $\\eta_{12}$: evacuation effect on shelter pressure
- $CH$: shelter capacity, $\\omega(\\cdot)$: contamination source
- $\\beta_{12}$: crowding effect, $\\delta$: decay, $\\sigma_p$: sanitation effectiveness
"""
    )

with theory_tabs[4]:
    st.markdown(
        """**Epidemics, healthcare, and outcomes**

Epidemics:
$$
\\dot{X}_4 = -\\lambda(X_6,X_{12}) X_4 + \\Pi(t) - \\nu X_4
$$
$$
\\dot{X}_5 = \\lambda(X_6,X_{12}) X_4 - \\big(\\gamma + \\sigma_4(X_7,u_H)\\big) X_5 - \\mu_I(X_7) X_5
$$
with
$$
\\lambda(X_6,X_{12}) = \\beta_0 + \\beta_6 \\rho_3(X_1) X_6 + \\beta_{12} H_c(X_{12})
$$

Healthcare and outcomes:
$$
\\dot{X}_7 = \\rho_7(u_H) - v_7(X_9) X_7 + \\sigma_4(X_5)(1-X_7)
$$
$$
\\dot{X}_{10} = \\psi_p(X_8,X_9) + c_I X_5 + c_D X_3
$$
$$
\\dot{X}_{11} = \\mu_F(X_1) + \\mu_D(X_3,X_{12}) + \\mu_I(X_5,X_7)
$$

**Variables**
- $X_4$: vulnerable, $X_5$: infected, $X_6$: contamination, $X_{12}$: shelter pressure
- $X_7$: healthcare capacity, $X_{10}$: economic loss, $X_{11}$: deaths
- $X_1, X_3, X_8, X_9$: flood/displacement/damage/failures, $u_H$: healthcare

**Parameters**
- $\\beta_0,\\beta_6,\\beta_{12}$: infection pressure coefficients
- $\\rho_3$: flood-to-exposure scaling, $H_c(\\cdot)$: crowding response
- $\\Pi(t)$: vulnerable inflow, $\\nu$: waning rate
- $\\gamma$: recovery, $\\sigma_4$: healthcare effect, $\\mu_I$: infection mortality
- $\\rho_7$: reinforcement rate, $v_7$: degradation from failures
- $\\psi_p, c_I, c_D$: economic loss terms
- $\\mu_F,\\mu_D,\\mu_I$: mortality contributions
"""
    )

st.subheader("🧪 Analysis Tools")
feature_tabs = st.tabs(
    [
        "Policy recommender",
        "Constraint-aware optimizer",
        "Scenario stress testing",
        "Counterfactual insights",
        "Explainability panel",
        "Alerting rules",
        "Pareto analysis",
        "Policy clustering",
        "2D slices",
    ]
)

with feature_tabs[0]:
    st.subheader("Policy recommender")
    st.markdown(
        "**Goal**: rank candidate constant actions by a multi-objective score (mortality, economic loss, services).\n"
        "Top options are shown with trade-off explanations."
    )
    run_reco = st.button("Run recommender", key="run_recommender")
    if run_reco:
        if len(reco_levels) < 1:
            st.error("Select at least one action level.")
            st.stop()
        levels = [float(level) * action_high for level in sorted(set(reco_levels))]
        candidates = _generate_candidates(levels)
        candidates.sort(key=lambda row: row["score"])
        top = candidates[:3]
        st.subheader("Top 3 recommendations")
        st.dataframe(top, use_container_width=True)
        for idx, row in enumerate(top, start=1):
            total = max(row["score"], 1e-9)
            contribs = {
                "mortality": w_mortality * row["deaths_delta"] / total,
                "economic": w_econ * row["econ_delta"] / total,
                "services": w_services * row["service_mean"] / total,
                "action": w_action * row["action_cost"] / total,
            }
            st.markdown(
                f"**Option {idx} trade-offs**: "
                f"mortality {contribs['mortality']:.2f}, "
                f"economic {contribs['economic']:.2f}, "
                f"services {contribs['services']:.2f}, "
                f"action {contribs['action']:.2f}."
            )

with feature_tabs[1]:
    st.subheader("Constraint-aware optimizer")
    st.markdown(
        "**Goal**: enforce budget/safety limits and compare best feasible vs best unconstrained actions."
    )
    run_opt = st.button("Run optimizer", key="run_optimizer")
    if run_opt:
        if len(reco_levels) < 1:
            st.error("Select at least one action level.")
            st.stop()
        levels = [float(level) * action_high for level in sorted(set(reco_levels))]
        candidates = _generate_candidates(levels)
        candidates.sort(key=lambda row: row["score"])
        best_unconstrained = candidates[0] if candidates else None

        def _feasible(row: dict) -> bool:
            if budget_enabled:
                cost = (
                    b_s * row["u_s"]
                    + b_e * row["u_e"]
                    + b_h * row["u_h"]
                    + b_p * row["u_p"]
                )
                if cost > budget_max:
                    return False
            if safety_enabled:
                if row["max_contamination"] > max_contamination:
                    return False
                if row["max_shelter"] > max_shelter:
                    return False
            return True

        feasible = [row for row in candidates if _feasible(row)]
        best_feasible = feasible[0] if feasible else None

        st.subheader("Best unconstrained")
        if best_unconstrained is None:
            st.caption("No candidates generated.")
        else:
            st.json(best_unconstrained)

        st.subheader("Best feasible (with constraints)")
        if best_feasible is None:
            st.warning("No feasible actions found with current constraints.")
        else:
            st.json(best_feasible)

        if best_unconstrained and best_feasible:
            gap_score = best_feasible["score"] - best_unconstrained["score"]
            st.markdown(
                f"**Feasibility gap**: score difference = {gap_score:.3f} "
                f"(lower is better)."
            )

with feature_tabs[2]:
    st.subheader("Scenario stress testing")
    st.markdown(
        "**Goal**: evaluate robustness across rainfall/fragility ensembles and recommend the most stable option."
    )
    run_stress = st.button("Run stress test", key="run_stress")
    if run_stress:
        if len(reco_levels) < 1:
            st.error("Select at least one action level.")
            st.stop()
        rng = np.random.default_rng(int(stress_seed))
        levels = [float(level) * action_high for level in sorted(set(reco_levels))]
        candidates = _generate_candidates(levels)
        if not candidates:
            st.error("No candidates generated.")
            st.stop()

        action_list = [
            np.array([row["u_s"], row["u_e"], row["u_h"], row["u_p"]], dtype=float)
            for row in candidates
        ]
        scores = {idx: [] for idx in range(len(action_list))}
        for _ in range(int(stress_ensembles)):
            rain = RainfallScenario(
                steps=steps,
                baseline=baseline if rainfall_type == "stochastic" else 0.2,
                pulse_prob=stress_pulse_prob,
                pulse_mean=stress_pulse_mean,
                pulse_std=stress_pulse_std,
                rng_seed=int(rng.integers(0, 1_000_000)),
            ).generate()
            for idx, action in enumerate(action_list):
                metrics = _evaluate_constant_action(action, rainfall_override=rain)
                score = metrics["score"] + fragility_gain * (
                    metrics["max_contamination"] + metrics["max_shelter"]
                )
                scores[idx].append(score)

        ranked = []
        for idx, action in enumerate(action_list):
            arr = np.asarray(scores[idx], dtype=float)
            ranked.append(
                {
                    "u_s": float(action[0]),
                    "u_e": float(action[1]),
                    "u_h": float(action[2]),
                    "u_p": float(action[3]),
                    "mean_score": float(np.mean(arr)),
                    "std_score": float(np.std(arr)),
                    "worst_score": float(np.max(arr)),
                }
            )
        ranked.sort(key=lambda row: (row["mean_score"], row["std_score"]))
        best = ranked[0]
        st.subheader("Most robust option")
        st.json(best)
        st.subheader("Top 5 robust candidates")
        st.dataframe(ranked[:5], use_container_width=True)

with feature_tabs[3]:
    st.subheader("Counterfactual insights")
    st.markdown(
        "**Goal**: estimate marginal impact of increasing a single lever relative to a baseline policy."
    )
    run_cf = st.button("Run counterfactual", key="run_counterfactual")
    if run_cf:
        lever_map = {
            "sanitation": 0,
            "evacuation": 1,
            "healthcare": 2,
            "personnel": 3,
        }
        lever_idx = lever_map[cf_lever]
        baseline = _run_policy_with_lever(
            cf_base_policy, lever_idx, increase_pct=0.0, horizon=cf_horizon
        )
        boosted = _run_policy_with_lever(
            cf_base_policy,
            lever_idx,
            increase_pct=cf_increase_pct / 100.0,
            horizon=cf_horizon,
        )
        deltas = {
            "deaths_delta": boosted["deaths_delta"] - baseline["deaths_delta"],
            "econ_delta": boosted["econ_delta"] - baseline["econ_delta"],
            "infected": boosted["infected"] - baseline["infected"],
            "contamination": boosted["contamination"] - baseline["contamination"],
            "displaced": boosted["displaced"] - baseline["displaced"],
        }
        st.subheader("Baseline vs boosted")
        st.dataframe(
            {
                "metric": list(baseline.keys()),
                "baseline": list(baseline.values()),
                "boosted": list(boosted.values()),
                "delta": [deltas[k] for k in baseline.keys()],
            },
            use_container_width=True,
        )

with feature_tabs[4]:
    st.subheader("Explainability panel")
    st.markdown(
        "**Goal**: attribute outcome changes to key drivers via simple sensitivity correlations."
    )
    run_expl = st.button("Run explainability", key="run_explainability")
    if run_expl:
        traj, _ = _run_policy_with_sensitivity("rule_based", horizon=expl_horizon)
        infected = traj[1:, IDX["infected"]]
        drivers = {
            "flood_depth": traj[1:, IDX["flood_depth"]],
            "contamination": traj[1:, IDX["contamination"]],
            "displaced": traj[1:, IDX["displaced"]],
        }
        scores = []
        for name, series in drivers.items():
            if np.std(series) < 1e-9:
                corr = 0.0
            else:
                corr = float(np.corrcoef(series, infected)[0, 1])
            scores.append({"driver": name, "sensitivity": abs(corr), "signed_corr": corr})
        scores.sort(key=lambda row: row["sensitivity"], reverse=True)
        st.subheader("Driver sensitivities (|corr|)")
        st.dataframe(scores, use_container_width=True)

with feature_tabs[5]:
    st.subheader("Alerting rules")
    st.markdown(
        "**Goal**: trigger recommended actions when thresholds are crossed, with justification and effect size."
    )
    if not alert_enabled:
        st.caption("Alerting disabled in sidebar.")
    else:
        run_alerts = st.button("Run alerts", key="run_alerts")
        if run_alerts:
            traj, _ = _run_policy_with_sensitivity("rule_based", horizon=alert_horizon)
            infected = traj[:, IDX["infected"]]
            contamination = traj[:, IDX["contamination"]]
            displaced = traj[:, IDX["displaced"]]

            alerts = []
            for t in range(1, len(traj)):
                if infected[t] >= alert_infected:
                    alerts.append(
                        {
                            "step": t,
                            "trigger": "infected",
                            "value": float(infected[t]),
                            "recommended": "increase healthcare (u_H)",
                            "effect": "expected to reduce recovery time",
                        }
                    )
                if contamination[t] >= alert_contamination:
                    alerts.append(
                        {
                            "step": t,
                            "trigger": "contamination",
                            "value": float(contamination[t]),
                            "recommended": "increase sanitation (u_S)",
                            "effect": "expected to reduce transmission pressure",
                        }
                    )
                if displaced[t] >= alert_displaced:
                    alerts.append(
                        {
                            "step": t,
                            "trigger": "displaced",
                            "value": float(displaced[t]),
                            "recommended": "increase evacuation (u_E)",
                            "effect": "expected to reduce shelter pressure",
                        }
                    )

            if not alerts:
                st.success("No alerts triggered under current thresholds.")
            else:
                st.subheader("Triggered alerts")
                st.dataframe(alerts, use_container_width=True)

with feature_tabs[6]:
    st.subheader("Pareto analysis")
    st.markdown(
        "**Goal**: identify non-dominated actions across mortality, economic loss, and service disruption."
    )
    run_pareto = st.button("Run Pareto analysis", key="run_pareto") or st.session_state.get(
        "trigger_pareto", False
    )
    if run_pareto:
        st.session_state["trigger_pareto"] = False
    if run_pareto:
        if len(reco_levels) < 1:
            st.error("Select at least one action level.")
            st.stop()
        levels = [float(level) * action_high for level in sorted(set(reco_levels))]
        candidates = _generate_candidates(levels)
        if not candidates:
            st.error("No candidates generated.")
            st.stop()
        front = _pareto_front(candidates)
        if len(front) > pareto_max_points:
            front = front[:pareto_max_points]
        st.session_state["pareto_front"] = front

    front = st.session_state.get("pareto_front")
    if front:
        st.subheader("Pareto front (sample)")
        st.dataframe(front, use_container_width=True)
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            [c["deaths_delta"] for c in front],
            [c["econ_delta"] for c in front],
            [c["service_mean"] for c in front],
            s=30,
        )
        ax.set_xlabel("Deaths (a.u.)")
        ax.set_ylabel("Economic loss (a.u.)")
        ax.set_zlabel("Service disruption (a.u.)")
        ax.set_title("Pareto front")
        st.pyplot(fig)
    else:
        st.caption("Run the Pareto analysis to view results.")

with feature_tabs[7]:
    st.subheader("Policy clustering")
    st.markdown("**Goal**: cluster policies by action patterns and compare average outcomes.")
    run_cluster = st.button("Run clustering", key="run_cluster") or st.session_state.get(
        "trigger_cluster", False
    )
    if run_cluster:
        st.session_state["trigger_cluster"] = False
    if run_cluster:
        if len(reco_levels) < 1:
            st.error("Select at least one action level.")
            st.stop()
        levels = [float(level) * action_high for level in sorted(set(reco_levels))]
        candidates = _generate_candidates(levels)
        actions = np.array(
            [[c["u_s"], c["u_e"], c["u_h"], c["u_p"]] for c in candidates], dtype=float
        )
        rng = np.random.default_rng(int(seed_global))
        centroids, labels = _kmeans(actions, cluster_k, cluster_iters, rng)
        summary = []
        for i in range(len(centroids)):
            idxs = np.where(labels == i)[0]
            if len(idxs) == 0:
                continue
            subset = [candidates[j] for j in idxs]
            summary.append(
                {
                    "cluster": i,
                    "count": len(idxs),
                    "center_u_s": float(centroids[i][0]),
                    "center_u_e": float(centroids[i][1]),
                    "center_u_h": float(centroids[i][2]),
                    "center_u_p": float(centroids[i][3]),
                    "mean_deaths": float(np.mean([c["deaths_delta"] for c in subset])),
                    "mean_econ": float(np.mean([c["econ_delta"] for c in subset])),
                    "mean_service": float(np.mean([c["service_mean"] for c in subset])),
                }
            )
        st.subheader("Cluster summary")
        st.dataframe(summary, use_container_width=True)

with feature_tabs[8]:
    st.subheader("2D slices (composite action indices)")
    st.markdown(
        "**Goal**: explore action space by composite indices (A_hydro, A_social, A_health, A_total)."
    )
    comp_options = ["A_hydro", "A_social", "A_health", "A_total"]
    x_comp = st.selectbox("X composite", comp_options, index=0, key="comp_x")
    y_comp = st.selectbox("Y composite", comp_options, index=1, key="comp_y")
    if x_comp == y_comp:
        st.warning("Choose two different composites.")
        st.stop()
    run_slice = st.button("Run composite slice", key="run_slice") or st.session_state.get(
        "trigger_slice", False
    )
    if run_slice:
        st.session_state["trigger_slice"] = False
    if run_slice:
        if len(reco_levels) < 1:
            st.error("Select at least one action level.")
            st.stop()
        levels = [float(level) * action_high for level in sorted(set(reco_levels))]
        candidates = _generate_candidates(levels)
        grid = np.full((slice_bins, slice_bins), np.nan)
        x_vals = np.linspace(0.0, action_high * 2.0, slice_bins)
        y_vals = np.linspace(0.0, action_high * 2.0, slice_bins)
        for i, y in enumerate(y_vals):
            for j, x in enumerate(x_vals):
                best = None
                best_score = float("inf")
                for c in candidates:
                    comp = _composite_indices(
                        np.array([c["u_s"], c["u_e"], c["u_h"], c["u_p"]], dtype=float)
                    )
                    if abs(comp[x_comp] - x) < (x_vals[1] - x_vals[0]) / 2 and abs(
                        comp[y_comp] - y
                    ) < (y_vals[1] - y_vals[0]) / 2:
                        if c["score"] < best_score:
                            best_score = c["score"]
                            best = c
                if best is not None:
                    grid[i, j] = best_score
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            grid,
            origin="lower",
            extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
            aspect="auto",
        )
        ax.set_xlabel(f"{x_comp} (a.u.)")
        ax.set_ylabel(f"{y_comp} (a.u.)")
        ax.set_title("Best score by composite indices")
        plt.colorbar(im, ax=ax, label="Score")
        st.pyplot(fig)



st.subheader("🧭 Policy Simulation Views")
tabs = st.tabs(["Rule-based", "MPC", "RL-based", "Hybrid", "Comparison", "Policy heatmap"])


def _render(policy_name: str, doc: str, require_model: bool = False):
    st.markdown(doc)
    if require_model and not rl_model_path:
        st.info("No PPO model path provided. This tab will run MPC as a fallback.")
    changed = st.session_state.get("last_run_config") != config_json
    if changed:
        st.warning("Parameters changed. Click Run to update trajectories.")
    run_clicked = st.button("Run scenario", key=f"run_{policy_name}")
    if run_clicked:
        st.session_state["last_run_config"] = config_json
        try:
            traj, actions = _run_policy(policy_name)
        except Exception as exc:
            st.error(str(exc))
            return
        st.session_state[f"traj_{policy_name}"] = traj
        st.session_state[f"actions_{policy_name}"] = actions

    traj = st.session_state.get(f"traj_{policy_name}")
    actions = st.session_state.get(f"actions_{policy_name}")
    if traj is None or actions is None:
        st.caption("Run the scenario to see trajectories.")
        return

    st.subheader("Trajectories")
    t_axis = np.arange(traj.shape[0])
    fig1, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
    axes = axes.ravel()
    series = [
        ("flood_depth", "Flood depth"),
        ("contamination", "Contamination"),
        ("infected", "Infected"),
        ("displaced", "Displaced"),
        ("econ_loss", "Economic loss"),
        ("deaths", "Deaths"),
    ]
    for ax, (key, label) in zip(axes, series):
        ax.plot(t_axis, traj[:, IDX[key]])
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    axes[-2].set_xlabel("Step")
    axes[-1].set_xlabel("Step")
    fig1.tight_layout()
    st.pyplot(fig1)

    st.subheader("Actions")
    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    axes2 = axes2.ravel()
    action_labels = ["Sanitation", "Evacuation", "Healthcare", "Personnel"]
    for ax, label, idx in zip(axes2, action_labels, range(4)):
        ax.plot(np.arange(actions.shape[0]), actions[:, idx])
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    axes2[-2].set_xlabel("Step")
    axes2[-1].set_xlabel("Step")
    fig2.tight_layout()
    st.pyplot(fig2)

    st.subheader("Sample crisis trajectory")
    show_all_states = st.checkbox(
        "Show all state trajectories", value=False, key=f"show_all_states_{policy_name}"
    )
    state_options = [
        ("flood_depth", "Flood depth"),
        ("contamination", "Contamination"),
        ("infected", "Infected"),
        ("displaced", "Displaced"),
        ("econ_loss", "Economic loss"),
        ("deaths", "Deaths"),
        ("flow", "Flow"),
        ("vulnerable", "Vulnerable"),
        ("healthcare", "Healthcare"),
        ("damage", "Damage"),
        ("failures", "Failures"),
        ("shelter_pressure", "Shelter pressure"),
        ("sanitation", "Sanitation"),
        ("personnel", "Personnel"),
    ]
    default_labels = ["Flood depth", "Contamination", "Infected"]
    selected_labels = st.multiselect(
        "Select trajectories",
        options=[label for _, label in state_options],
        default=default_labels,
        key=f"traj_select_{policy_name}",
    )
    selected_keys = [key for key, label in state_options if label in selected_labels]
    t_axis = np.arange(traj.shape[0])
    fig3, ax1 = plt.subplots(figsize=(10, 4))
    if show_all_states:
        for key, label in state_options:
            ax1.plot(t_axis, traj[:, IDX[key]], label=key)
    else:
        if not selected_keys:
            st.warning("Select at least one trajectory.")
        else:
            for key in selected_keys:
                label = dict(state_options)[key]
                ax1.plot(t_axis, traj[:, IDX[key]], label=label)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("State level (normalized)")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    action_id = np.argmax(actions, axis=1)
    ax2.step(np.arange(actions.shape[0]), action_id, where="post", color="tab:red", label="Action ID")
    ax2.set_ylabel("Action ID (0-3)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig3.tight_layout()
    st.pyplot(fig3)


with tabs[0]:
    _render(
        "rule_based",
        """**Policy overview**
Rule-based control applies fixed thresholds to trigger actions. It mirrors typical emergency playbooks.

**Mathematical form**
Let $x$ be the state and $T_i$ thresholds. For each action component:
$$
u_i = \\begin{cases}
u_{\\max}, & x_{k(i)} \\ge T_i \\\\
0, & \\text{otherwise}
\\end{cases}
$$

**Purpose**
Provide a transparent, interpretable baseline that reflects common operational playbooks.

**Rationale**
Thresholds mimic standard emergency triggers (e.g., flood depth or contamination exceeds a limit), enabling
fast response without complex optimization. This is a strong reference point for comparing adaptive methods.

**Policy structure**
Let $u = [u_S, u_E, u_H, u_P]$. Threshold logic:

$$
u_i = u_{\\max} \\; \\text{if} \\; x_i \\ge T_i, \\quad \\text{otherwise} \\; u_i = 0.
$$
""",
    )
with tabs[1]:
    _render(
        "mpc",
        """**Policy overview**
Model Predictive Control (MPC) evaluates candidate actions using the system dynamics and selects the best
one-step action from a discrete grid.

**Mathematical form**
Given dynamics $x_{t+1} = f(x_t, u_t)$ and instantaneous reward $r(x_t, u_t)$:
$$
u_t^* = \\arg\\max_{u \\in \\mathcal{U}} r(x_t, u)
$$

**Purpose**
Provide a control-theoretic baseline that selects actions by optimizing the next-step objective.

**Rationale**
MPC approximates optimal control by evaluating candidate actions against the model, yielding a policy that
explicitly trades off health, economic, and service objectives while respecting the dynamics.

**Objective**
$$
r_t = -\\big(w_I X_5 + w_C X_6 + w_M \\, dX_{11} + w_E \\, dX_{10} + w_U \\lVert u \\rVert_1\\big)
$$

The chosen action is:
$$
u_t^* = \\arg\\max_{u \\in \\mathcal{U}} r_t
$$
over the action grid (one-step lookahead).
""",
    )
with tabs[2]:
    _render(
        "rl",
        """**Policy overview**
Reinforcement Learning (RL) learns a policy from simulation data that maps observed state to actions by
optimizing long-term reward.

**Mathematical form**
The policy optimizes expected return:
$$
\\pi_\\theta = \\arg\\max_{\\pi} \\mathbb{E}_\\pi \\left[ \\sum_{t=0}^{T} \\gamma^t r(x_t, u_t) \\right]
$$
with $u_t = \\pi_\\theta(x_t)$.

**Purpose**
Enable adaptive decision-making from simulation data, potentially outperforming fixed rules in complex regimes.

**Rationale**
RL learns a mapping from observed state to actions by optimizing long-term rewards in simulation, capturing
nonlinear feedbacks and delayed effects across subsystems.

**Policy**
$$
u_t = \\pi_\\theta(x_t)
$$
where $\\pi_\\theta$ is a trained neural policy (loaded externally).

**Fallback behavior**
If no PPO model path is provided, this tab automatically falls back to the MPC policy. This preserves
functionality while keeping the RL interface consistent. When a PPO model is available, it overrides the
fallback and runs purely RL inference.
""",
        require_model=True,
    )
with tabs[3]:
    _render(
        "hybrid",
        """**Policy overview**
Hybrid control blends rule-based actions with MPC to balance robustness and optimization.

**Mathematical form**
Let $u^{\\text{MPC}}$ and $u^{\\text{rule}}$ be the two actions:
$$
u = \\alpha u^{\\text{MPC}} + (1-\\alpha) u^{\\text{rule}}
$$

**Purpose**
Combine interpretability and responsiveness by blending reactive rules with model-based optimization.

**Rationale**
Hybrid control hedges against modeling errors while still leveraging optimization when it is reliable.
The blend weight controls the balance between robustness (rules) and efficiency (MPC).

**Blend**
$$
u = \\alpha \\, u_{\\text{MPC}} + (1-\\alpha) \\, u_{\\text{rule}}
$$
with $\\alpha$ set by the sidebar slider.
""",
    )
with tabs[4]:
    st.markdown(
        """**Scenario/approach**

Compare trajectories across all approaches using the same scenario and parameters.
"""
    )
    compare_options = [
        ("rule_based", "Rule-based"),
        ("mpc", "MPC"),
        ("hybrid", "Hybrid"),
        ("rl", "RL-based"),
    ]
    selected_labels = st.multiselect(
        "Select approaches to compare (choose at least 2)",
        options=[label for _, label in compare_options],
        default=["Rule-based", "MPC", "Hybrid"],
    )
    selected_names = [
        name for name, label in compare_options if label in selected_labels
    ]
    if len(selected_names) < 2:
        st.warning("Select at least two approaches to compare.")
        st.stop()
    changed = st.session_state.get("last_run_config_compare") != config_json
    if changed:
        st.warning("Parameters changed. Click Run to update comparison.")
    run_clicked = st.button("Run comparison", key="run_compare")
    if run_clicked:
        st.session_state["last_run_config_compare"] = config_json
        results = {}
        for name in selected_names:
            try:
                traj, actions = _run_policy(name)
            except Exception as exc:
                st.error(str(exc))
                st.stop()
            results[name] = {"traj": traj, "actions": actions}
        st.session_state["compare_results"] = results

    results = st.session_state.get("compare_results")
    if not results:
        st.caption("Run the comparison to see trajectories.")
    else:
        st.subheader("Trajectories (Overlay)")
        fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
        axes = axes.ravel()
        series = [
            ("flood_depth", "Flood depth"),
            ("contamination", "Contamination"),
            ("infected", "Infected"),
            ("displaced", "Displaced"),
            ("econ_loss", "Economic loss"),
            ("deaths", "Deaths"),
        ]
        t_axis = np.arange(next(iter(results.values()))["traj"].shape[0])
        for ax, (key, label) in zip(axes, series):
            for name, payload in results.items():
                ax.plot(t_axis, payload["traj"][:, IDX[key]], label=name)
            ax.set_title(label)
            ax.grid(True, alpha=0.3)
        axes[0].legend(loc="best")
        axes[-2].set_xlabel("Step")
        axes[-1].set_xlabel("Step")
        fig.tight_layout()
        st.pyplot(fig)

with tabs[5]:
    st.markdown(
        "**Policy heatmap**: best action per (flood depth, contamination) state cell using a short-horizon score."
    )
    axis_options = [
        ("flood_depth", "Flood depth (X1)"),
        ("contamination", "Contamination (X6)"),
        ("infected", "Infected (X5)"),
        ("displaced", "Displaced (X3)"),
        ("shelter_pressure", "Shelter pressure (X12)"),
        ("damage", "Damage (X8)"),
    ]
    axis_labels = [label for _, label in axis_options]
    x_axis_label = st.selectbox(
        "X-axis",
        options=axis_labels,
        index=axis_labels.index("Contamination (X6)"),
        key="heat_x_axis",
    )
    y_axis_label = st.selectbox(
        "Y-axis",
        options=axis_labels,
        index=axis_labels.index("Flood depth (X1)"),
        key="heat_y_axis",
    )
    x_axis_key = next(key for key, label in axis_options if label == x_axis_label)
    y_axis_key = next(key for key, label in axis_options if label == y_axis_label)
    if x_axis_key == y_axis_key:
        st.warning("Choose two different axes.")
        st.stop()
    heat_bins = st.slider("Bins per axis", 3, 12, 5, 1, key="heat_bins")
    heat_horizon = st.slider("Heatmap horizon", 1, 10, 3, 1, key="heat_horizon")
    run_heat = st.button("Generate heatmap", key="run_heatmap")
    if run_heat:
        x_min, x_max = 0.0, 2.0
        y_min, y_max = 0.0, 2.0
        x_edges = np.linspace(x_min, x_max, heat_bins)
        y_edges = np.linspace(y_min, y_max, heat_bins)

        action_catalog = [
            ("do_nothing", np.array([0.0, 0.0, 0.0, 0.0])),
            ("sanitize", np.array([action_high, 0.0, 0.0, 0.0])),
            ("evacuate", np.array([0.0, action_high, 0.0, 0.0])),
            ("healthcare", np.array([0.0, 0.0, action_high, 0.0])),
            ("personnel", np.array([0.0, 0.0, 0.0, action_high])),
            ("balanced", np.array([0.5, 0.5, 0.5, 0.5]) * action_high),
        ]

        heatmap = np.zeros((heat_bins, heat_bins), dtype=int)
        for i, y_val in enumerate(y_edges):
            for j, x_val in enumerate(x_edges):
                state = default_state()
                state[IDX[y_axis_key]] = y_val
                state[IDX[x_axis_key]] = x_val
                best_idx = 0
                best_score = float("inf")
                for k, (_, action) in enumerate(action_catalog):
                    score = _score_action_from_state(state, action, heat_horizon)
                    if score < best_score:
                        best_score = score
                        best_idx = k
                heatmap[i, j] = best_idx

        fig, ax = plt.subplots(figsize=(7, 5))
        cmap = plt.get_cmap("tab10", len(action_catalog))
        im = ax.imshow(
            heatmap,
            origin="lower",
            cmap=cmap,
            extent=[x_min, x_max, y_min, y_max],
            aspect="auto",
        )
        ax.set_xlabel(f"{x_axis_label} (normalized)")
        ax.set_ylabel(f"{y_axis_label} (normalized)")
        ax.set_title("Best action per state cell")
        cbar = plt.colorbar(im, ax=ax, ticks=range(len(action_catalog)))
        cbar.ax.set_yticklabels([name for name, _ in action_catalog])
        handles = [
            plt.Line2D([0], [0], marker="s", linestyle="", color=cmap(i), label=name)
            for i, (name, _) in enumerate(action_catalog)
        ]
        ax.legend(handles=handles, title="Actions", loc="upper right", frameon=True)
        st.pyplot(fig)
