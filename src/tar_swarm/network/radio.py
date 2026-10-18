"""Toy bidirectional radius model; no calibrated RF or packet behavior."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class LinkModel(Protocol):
    def usable(self, first: np.ndarray, second: np.ndarray) -> bool:
        """Return whether two active positions have a physical connection."""
        ...


@dataclass(frozen=True)
class RadiusLink:
    radius_m: float

    def __post_init__(self):
        if not np.isfinite(self.radius_m) or self.radius_m <= 0:
            raise ValueError("Communication radius must be positive and finite")

    def usable(self, first: np.ndarray, second: np.ndarray) -> bool:
        return bool(np.linalg.norm(first - second) <= self.radius_m)
