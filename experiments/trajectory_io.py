"""CSV helpers for trajectory outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def write_trajectory_csv(
    path: str | Path,
    rainfall: np.ndarray,
    actions: np.ndarray,
    traj: np.ndarray,
    state_names: list[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "t",
        "rainfall",
        "action_s",
        "action_e",
        "action_h",
        "action_p",
    ] + state_names
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        steps = traj.shape[0] - 1
        for t in range(steps + 1):
            rain = float(rainfall[t]) if t < len(rainfall) else float(rainfall[-1])
            action = actions[t] if t < len(actions) else actions[-1]
            row = [t, rain, *action.tolist(), *traj[t].tolist()]
            writer.writerow(row)
