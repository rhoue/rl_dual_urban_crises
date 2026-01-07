"""Observation model with partial observability and noise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.coupled_system import IDX


@dataclass
class ObservationParams:
    noise_std: float = 0.01
    include_latent: bool = False

    @property
    def observed_indices(self) -> list[int]:
        base = [
            IDX["flood_depth"],
            IDX["flow"],
            IDX["displaced"],
            IDX["shelter_pressure"],
            IDX["damage"],
            IDX["failures"],
        ]
        if self.include_latent:
            base += [
                IDX["contamination"],
                IDX["healthcare"],
            ]
        return base


def observe(state: np.ndarray, params: ObservationParams, rng: np.random.Generator) -> np.ndarray:
    idx = params.observed_indices
    obs = state[idx]
    noise = rng.normal(0.0, params.noise_std, size=obs.shape)
    return np.clip(obs + noise, 0.0, None)
