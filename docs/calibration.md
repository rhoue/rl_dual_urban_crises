# Calibration and parameter IO

This project supports JSON-based parameter snapshots for reproducibility and calibration.

## Save and load parameters

Use `models/param_io.py` to serialize `CoupledParams` to JSON and restore it later:

```python
from models.coupled_system import CoupledParams
from models.param_io import save_params, load_params

params = CoupledParams()
save_params(params, "calibration/default_params.json")

loaded = load_params("calibration/default_params.json")
```

## Calibration hooks

Calibration workflows can update a JSON file (e.g. via external fitting routines) and reload it into the simulator. The loader merges missing fields with defaults so partial updates are safe.

Recommended workflow:
1. Export a baseline JSON snapshot.
2. Run your fitting routine to adjust selected fields.
3. Reload the JSON in your simulation or environment.

Example JSON structure:
```json
{
  "hydrology": {"alpha1": 0.85, "k1": 0.25},
  "infrastructure": {"gamma8": 0.5},
  "rho7": 0.55
}
```
