"""CLI wrapper to run scenarios with saved configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from control.mpc_baseline import OneStepMPC
from control.rule_based import RuleBasedPolicy
from experiments.trajectory_io import write_trajectory_csv
from models.coupled_system import CoupledParams, CoupledSystem, STATE_NAMES, default_state
from models.param_io import load_params, save_params
from scenarios.rainfall_generator import RainfallScenario, deterministic_rainfall


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text())


def _build_rainfall(cfg: dict) -> np.ndarray:
    if cfg.get("type") == "deterministic":
        return deterministic_rainfall(
            steps=int(cfg.get("steps", 60)),
            intensity=float(cfg.get("intensity", 0.8)),
            decay=float(cfg.get("decay", 0.98)),
        )
    scenario = RainfallScenario(
        steps=int(cfg.get("steps", 60)),
        baseline=float(cfg.get("baseline", 0.2)),
        pulse_prob=float(cfg.get("pulse_prob", 0.1)),
        pulse_mean=float(cfg.get("pulse_mean", 2.0)),
        pulse_std=float(cfg.get("pulse_std", 0.5)),
        rng_seed=cfg.get("seed"),
    )
    return scenario.generate()


def _build_policy(policy_cfg: dict, system: CoupledSystem):
    policy_type = policy_cfg.get("type", "rule_based")
    if policy_type == "mpc":
        return OneStepMPC(system)
    if policy_type == "zero":
        return lambda t, state, rainfall=None: np.zeros(4, dtype=float)
    action_high = float(policy_cfg.get("action_high", 1.0))
    return RuleBasedPolicy(action_high=action_high)


def run_scenario(config_path: str | None, output_path: str) -> None:
    cfg = _load_config(config_path)
    params_path = cfg.get("params_path")
    params = load_params(params_path) if params_path else CoupledParams()
    system = CoupledSystem(params)
    rainfall = _build_rainfall(cfg.get("rainfall", {}))
    policy = _build_policy(cfg.get("policy", {}), system)

    steps = len(rainfall)
    traj = np.zeros((steps + 1, len(STATE_NAMES)), dtype=float)
    actions = np.zeros((steps, 4), dtype=float)
    state = default_state()
    traj[0] = state
    for t, r in enumerate(rainfall):
        if isinstance(policy, OneStepMPC):
            action = policy(t, state, float(r))
        else:
            action = policy(t, state)
        actions[t] = action
        state = system.step(state, action, float(r))
        traj[t + 1] = state

    write_trajectory_csv(output_path, rainfall, actions, traj, STATE_NAMES)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dual crisis scenario and export CSV output.")
    parser.add_argument("--config", help="Path to scenario config JSON")
    parser.add_argument("--output", default="experiments/scenario_output.csv", help="Output CSV path")
    parser.add_argument("--export-params", help="Export default params to JSON and exit")
    args = parser.parse_args()

    if args.export_params:
        save_params(CoupledParams(), args.export_params)
        return

    run_scenario(args.config, args.output)


if __name__ == "__main__":
    main()
