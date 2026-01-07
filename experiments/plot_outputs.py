"""Plot CSV experiment outputs into PNG trajectories."""

from __future__ import annotations

import argparse
import csv
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


def _read_csv(path: str | Path) -> tuple[list[str], np.ndarray]:
    with Path(path).open() as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(x) for x in row] for row in reader]
    return header, np.asarray(rows, dtype=float)


def _col_index(header: list[str], name: str) -> int:
    try:
        return header.index(name)
    except ValueError as exc:
        raise ValueError(f"Missing column: {name}") from exc


def plot_csv(input_path: str, output_path: str, metrics: list[str]) -> None:
    header, data = _read_csv(input_path)
    t = data[:, _col_index(header, "t")]

    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
    axes = axes.ravel()
    for ax, metric in zip(axes, metrics):
        ax.plot(t, data[:, _col_index(header, metric)])
        ax.set_title(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
    axes[-2].set_xlabel("Step")
    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CSV experiment output to PNG.")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument(
        "--metrics",
        default="flood_depth,contamination,infected,displaced,econ_loss,deaths",
        help="Comma-separated list of metrics to plot",
    )
    args = parser.parse_args()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if len(metrics) < 1:
        raise SystemExit("No metrics provided")
    plot_csv(args.input, args.output, metrics)


if __name__ == "__main__":
    sys.exit(main())
