"""Local metric coordinates and geographic conversion helpers.

Conventions
-----------
- Internal planning uses a local East-North-Up (ENU) frame in meters.
- x = east, y = north, z = up, matching the network simulator and Step 0 MuJoCo world.
- Survey altitude is an absolute z coordinate above the local datum (same units
  as positions). For flat-field planning this equals AGL when the datum is local
  ground. Terrain-relative AGL is not modeled.
- Geographic inputs use WGS84 lat/lon degrees. Conversion to local meters uses a
  local tangent-plane (equirectangular) approximation about a declared origin.
  This is intended for survey regions of a few kilometers; larger extents should
  use a projected CRS outside this package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6_378_137.0


@dataclass(frozen=True)
class LocalFrame:
    """Origin of a local ENU frame on the WGS84 ellipsoid."""

    origin_lat_deg: float
    origin_lon_deg: float
    origin_alt_m: float = 0.0

    def geo_to_local(self, lat_deg: float, lon_deg: float, alt_m: float = 0.0):
        lat0 = math.radians(self.origin_lat_deg)
        dlat = math.radians(lat_deg - self.origin_lat_deg)
        dlon = math.radians(lon_deg - self.origin_lon_deg)
        east = EARTH_RADIUS_M * dlon * math.cos(lat0)
        north = EARTH_RADIUS_M * dlat
        up = alt_m - self.origin_alt_m
        return np.asarray([east, north, up], dtype=float)

    def local_to_geo(self, position_m: np.ndarray):
        east, north, up = (float(v) for v in np.asarray(position_m, dtype=float))
        lat0 = math.radians(self.origin_lat_deg)
        lat = self.origin_lat_deg + math.degrees(north / EARTH_RADIUS_M)
        lon = self.origin_lon_deg + math.degrees(
            east / (EARTH_RADIUS_M * math.cos(lat0))
        )
        alt = self.origin_alt_m + up
        return lat, lon, alt


def assert_enu_vector(value, name: str = "position") -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite 3-vector [east, north, up] meters")
    return arr
