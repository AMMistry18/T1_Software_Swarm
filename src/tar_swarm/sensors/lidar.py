"""Simulated Unitree L2 LiDAR: scan pattern, noise, and conversion to points."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

PATTERNS = ("random_uniform",)


@dataclass(frozen=True)
class LidarParams:
    site: str
    rate_hz: float
    points_per_scan: int
    max_polar_deg: float
    min_range_m: float
    max_range_m: float
    range_noise_m: float
    dropout_probability: float
    pattern: str


def validate_lidar_params(params: dict[str, Any]) -> None:
    for key in (
        "site",
        "rate_hz",
        "points_per_scan",
        "max_polar_deg",
        "min_range_m",
        "max_range_m",
        "range_noise_m",
        "dropout_probability",
        "pattern",
    ):
        if key not in params:
            raise ValueError(f"Missing LiDAR parameter: {key}")
    if not isinstance(params["site"], str) or not params["site"]:
        raise ValueError("LiDAR site must be a nonempty string")
    for key in ("rate_hz", "min_range_m", "max_range_m"):
        if not np.isfinite(params[key]) or params[key] <= 0:
            raise ValueError(f"Invalid LiDAR parameter: {key}")
    count = params["points_per_scan"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("LiDAR points_per_scan must be a positive integer")
    if not np.isfinite(params["max_polar_deg"]) or not (
        0 < params["max_polar_deg"] <= 180
    ):
        raise ValueError("LiDAR max_polar_deg must be in (0, 180]")
    if params["min_range_m"] >= params["max_range_m"]:
        raise ValueError("LiDAR min_range_m must be below max_range_m")
    if not np.isfinite(params["range_noise_m"]) or params["range_noise_m"] < 0:
        raise ValueError("LiDAR range_noise_m must be non-negative")
    if not 0 <= params["dropout_probability"] < 1:
        raise ValueError("LiDAR dropout_probability must be in [0, 1)")
    if params["pattern"] not in PATTERNS:
        raise ValueError(f"Unknown LiDAR pattern: {params['pattern']}")


def load_lidar_params(
    sensors: dict[str, Any] | None, vehicle_type: str
) -> LidarParams | None:
    """Parameters for a vehicle type, or None when it carries no LiDAR."""
    raw = (sensors or {}).get("lidar", {}).get(vehicle_type)
    if raw is None:
        return None
    validate_lidar_params(raw)
    return LidarParams(**{key: raw[key] for key in LidarParams.__annotations__})


def scan_directions(params: LidarParams, rng: np.random.Generator) -> NDArray:
    """Unit directions (N, 3) in the site frame; +z is the sensor axis.

    ASSUMPTION: the real L2 non-repetitive pattern is unpublished. random_uniform
    draws a fresh set of directions per scan, uniform over the spherical cap of
    half-angle max_polar_deg.
    """
    n = params.points_per_scan
    cos_polar = rng.uniform(np.cos(np.radians(params.max_polar_deg)), 1.0, n)
    azimuth = rng.uniform(0.0, 2 * np.pi, n)
    sin_polar = np.sqrt(1.0 - cos_polar**2)
    return np.column_stack(
        [sin_polar * np.cos(azimuth), sin_polar * np.sin(azimuth), cos_polar]
    )


def apply_noise(
    distances: NDArray, params: LidarParams, rng: np.random.Generator
) -> NDArray:
    """Measured distances; NaN marks no-return (miss, dropout, or out of range).

    Input distances use -1 for no hit, as returned by the ray caster.
    """
    distances = np.asarray(distances, dtype=float)
    noise = rng.normal(0.0, params.range_noise_m, distances.shape)
    dropped = rng.uniform(size=distances.shape) < params.dropout_probability
    measured = distances + noise
    invalid = (
        (distances < 0)
        | dropped
        | (measured < params.min_range_m)
        | (measured > params.max_range_m)
    )
    return np.where(invalid, np.nan, measured)


def to_points(directions: NDArray, distances: NDArray) -> NDArray:
    """Site-frame points (M, 3) for valid returns only."""
    valid = np.isfinite(distances)
    return directions[valid] * distances[valid, None]
