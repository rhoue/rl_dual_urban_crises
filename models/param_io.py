"""Parameter IO helpers for reproducibility and calibration."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

from models.coupled_system import CoupledParams


def coupled_params_to_dict(params: CoupledParams) -> dict:
    return asdict(params)


def _merge_dataclass(default, data: dict):
    if not data:
        return default
    return replace(default, **data)


def coupled_params_from_dict(data: dict) -> CoupledParams:
    base = CoupledParams()
    return CoupledParams(
        hydrology=_merge_dataclass(base.hydrology, data.get("hydrology", {})),
        infrastructure=_merge_dataclass(base.infrastructure, data.get("infrastructure", {})),
        displacement=_merge_dataclass(base.displacement, data.get("displacement", {})),
        contamination=_merge_dataclass(base.contamination, data.get("contamination", {})),
        epidemic=_merge_dataclass(base.epidemic, data.get("epidemic", {})),
        economy=_merge_dataclass(base.economy, data.get("economy", {})),
        rho7=data.get("rho7", base.rho7),
        v7=data.get("v7", base.v7),
        sigma4_x=data.get("sigma4_x", base.sigma4_x),
        rho14=data.get("rho14", base.rho14),
        k14=data.get("k14", base.k14),
        sanitation_decay=data.get("sanitation_decay", base.sanitation_decay),
    )


def save_params(params: CoupledParams, path: str | Path) -> None:
    payload = coupled_params_to_dict(params)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_params(path: str | Path) -> CoupledParams:
    data = json.loads(Path(path).read_text())
    return coupled_params_from_dict(data)
