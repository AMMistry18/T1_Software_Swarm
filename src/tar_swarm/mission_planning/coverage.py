"""Geometric survey coverage tracking (not verified LiDAR map quality)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union


@dataclass
class CoverageTracker:
    """Accumulate covered area once; repeated passes do not farm reward."""

    region: Polygon
    swath_m: float
    _covered: object | None = field(default=None, init=False, repr=False)
    covered_area_m2: float = 0.0

    def __post_init__(self):
        self._covered = None
        self.covered_area_m2 = 0.0

    @property
    def fraction(self) -> float:
        area = float(self.region.area)
        if area <= 0:
            return 0.0
        return min(1.0, self.covered_area_m2 / area)

    def observe_survey_motion(
        self, start: np.ndarray, end: np.ndarray, surveying: bool
    ) -> float:
        """Integrate a survey path segment; return newly covered area (m^2)."""
        if not surveying:
            return 0.0
        start_xy = (float(start[0]), float(start[1]))
        end_xy = (float(end[0]), float(end[1]))
        if np.linalg.norm(np.asarray(end_xy) - np.asarray(start_xy)) < 1e-9:
            footprint = Point(start_xy).buffer(self.swath_m * 0.5)
        else:
            footprint = LineString([start_xy, end_xy]).buffer(
                self.swath_m * 0.5, cap_style=2, join_style=2
            )
        piece = footprint.intersection(self.region)
        if piece.is_empty:
            return 0.0
        if self._covered is None:
            new_area = float(piece.area)
            self._covered = piece
        else:
            before = float(self._covered.area)
            self._covered = unary_union([self._covered, piece])
            new_area = max(0.0, float(self._covered.area) - before)
        self.covered_area_m2 = float(self._covered.area)
        return new_area
