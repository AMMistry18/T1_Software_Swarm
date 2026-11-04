"""Toy two-way radio-range model; no calibrated RF or packet behavior."""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from tar_swarm.common.world import line_of_sight


class LinkModel(Protocol):
    def usable(
        self, first_role: str, second_role: str, first: np.ndarray, second: np.ndarray
    ) -> bool:
        """Return whether two active drones (roles, positions) have a link."""
        ...


@dataclass
class RangeLink:
    """Each kind of drone (role) has a radio range, in meters.

    Two drones are linked when their 3D distance is at most the SMALLER of their
    two ranges, and no obstacle box stands between them on the map. A link
    carries messages both ways, so each drone must reach the other: a base with a
    600 m range still can't link with a 300 m network drone that is 400 m away,
    because the network drone can't answer.

    range_m maps role -> range; the live window's sliders change it during a run.
    obstacles is a list of boxes (x_min, y_min, x_max, y_max) that block signals;
    the simulator shares its own list, so boxes drawn during a run count at once.
    """

    range_m: dict[str, float]
    # field(default_factory=list) gives every RangeLink its own new empty list.
    obstacles: list = field(default_factory=list)

    def __post_init__(self):
        for role, range_m in self.range_m.items():
            if not np.isfinite(range_m) or range_m <= 0:
                raise ValueError(f"Radio range for {role} must be positive and finite")

    def usable(self, first_role, second_role, first, second) -> bool:
        limit_m = min(self.range_m[first_role], self.range_m[second_role])
        # The 1e-6 m (a thousandth of a millimeter) of slack stops rounding errors
        # from breaking a hop placed exactly at its limit.
        if np.linalg.norm(first - second) > limit_m + 1e-6:
            return False
        # [:2] keeps x and y: the obstacles are 2D boxes on the map.
        return line_of_sight(first[:2], second[:2], self.obstacles)
