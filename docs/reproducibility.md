# Reproducibility

- Use fixed RNG seeds in scenarios and environments when running comparisons.
- Persist parameter snapshots to JSON via `models/param_io.py` and store them with experiment outputs.
- Store rainfall scenarios (or their seeds) alongside results to recreate trajectories.
