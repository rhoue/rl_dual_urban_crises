"""Infrastructure fragility profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FragilityProfile:
    threshold: float = 1.0
    slope: float = 3.0

    def damage_multiplier(self, intensity: float) -> float:
        return 1.0 / (1.0 + np.exp(-self.slope * (intensity - self.threshold)))


def apply_fragility(intensity: float, profile: FragilityProfile) -> float:
    return profile.damage_multiplier(intensity)
