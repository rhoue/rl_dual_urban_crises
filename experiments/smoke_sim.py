"""Quick smoke simulation and trajectory plot."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cache_dir = ROOT / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
cache_dir.mkdir(parents=True, exist_ok=True)
(cache_dir / "mpl").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from control.mpc_baseline import OneStepMPC
from control.rule_based import RuleBasedPolicy
from experiments.trajectory_io import write_trajectory_csv
from models.coupled_system import CoupledSystem, IDX, STATE_NAMES, default_state
from scenarios.rainfall_generator import RainfallScenario


def main() -> None:
    steps = 80
    rainfall = RainfallScenario(steps=steps, baseline=0.2, pulse_prob=0.15).generate()
    system = CoupledSystem()
    rule_policy = RuleBasedPolicy()
    mpc_policy = OneStepMPC(system)

    def rollout(policy) -> tuple[np.ndarray, np.ndarray]:
        state = default_state()
        traj = np.zeros((steps + 1, len(IDX)), dtype=float)
        actions = np.zeros((steps, 4), dtype=float)
        traj[0] = state
        for t in range(steps):
            if isinstance(policy, OneStepMPC):
                action = policy(t, state, float(rainfall[t]))
            else:
                action = policy(t, state)
            actions[t] = action
            state = system.step(state, action, rainfall[t])
            traj[t + 1] = state
        return traj, actions

    rule_traj, rule_actions = rollout(rule_policy)
    mpc_traj, mpc_actions = rollout(mpc_policy)

    write_trajectory_csv(
        "experiments/smoke_sim_rule.csv", rainfall, rule_actions, rule_traj, STATE_NAMES
    )
    write_trajectory_csv(
        "experiments/smoke_sim_mpc.csv", rainfall, mpc_actions, mpc_traj, STATE_NAMES
    )

    t_axis = np.arange(steps + 1)
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
    for ax, (key, label) in zip(axes, series):
        ax.plot(t_axis, rule_traj[:, IDX[key]], label="Rule-based")
        ax.plot(t_axis, mpc_traj[:, IDX[key]], label="MPC")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="best")
    axes[-2].set_xlabel("Day")
    axes[-1].set_xlabel("Day")
    fig.suptitle("Dual Crisis Smoke Simulation (Rule vs MPC)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = "experiments/smoke_sim_compare.png"
    fig.savefig(out_path, dpi=150)


if __name__ == "__main__":
    main()
