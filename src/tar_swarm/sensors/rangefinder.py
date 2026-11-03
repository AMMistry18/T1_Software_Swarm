"""Simulated Benewake TF-Luna single-point rangefinder: beam, noise, limits."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

GOLDEN_ANGLE = np.pi * (3 - np.sqrt(5))


@dataclass(frozen=True)
class RangefinderParams:
    site: str
    rate_hz: float
    min_range_m: float
    max_range_m: float
    effective_max_range_m: float
    noise_sigma_near_m: float
    noise_sigma_far_fraction: float
    noise_switch_m: float
    dropout_probability: float
    num_rays: int
    beam_half_angle_deg: float


def validate_rangefinder_params(params: dict[str, Any]) -> None:
    for key in ("enabled", *RangefinderParams.__annotations__):
        if key not in params:
            raise ValueError(f"Missing rangefinder parameter: {key}")
    if not isinstance(params["enabled"], bool):
        raise ValueError("Rangefinder enabled must be true or false")
    if not isinstance(params["site"], str) or not params["site"]:
        raise ValueError("Rangefinder site must be a nonempty string")
    for key in (
        "rate_hz",
        "min_range_m",
        "max_range_m",
        "effective_max_range_m",
        "noise_switch_m",
    ):
        if not np.isfinite(params[key]) or params[key] <= 0:
            raise ValueError(f"Invalid rangefinder parameter: {key}")
    if not (
        params["min_range_m"] < params["effective_max_range_m"] <= params["max_range_m"]
    ):
        raise ValueError(
            "Rangefinder needs min_range_m < effective_max_range_m <= max_range_m"
        )
    for key in ("noise_sigma_near_m", "noise_sigma_far_fraction"):
        if not np.isfinite(params[key]) or params[key] < 0:
            raise ValueError(f"Rangefinder {key} must be non-negative")
    if not 0 <= params["dropout_probability"] < 1:
        raise ValueError("Rangefinder dropout_probability must be in [0, 1)")
    count = params["num_rays"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("Rangefinder num_rays must be a positive integer")
    if not np.isfinite(params["beam_half_angle_deg"]) or not (
        0 <= params["beam_half_angle_deg"] < 90
    ):
        raise ValueError("Rangefinder beam_half_angle_deg must be in [0, 90)")


def load_rangefinder_params(
    sensors: dict[str, Any] | None, vehicle_type: str
) -> RangefinderParams | None:
    """Parameters for a vehicle type, or None when absent or disabled."""
    raw = (sensors or {}).get("rangefinder", {}).get(vehicle_type)
    if raw is None:
        return None
    validate_rangefinder_params(raw)
    if not raw["enabled"]:
        return None
    return RangefinderParams(
        **{key: raw[key] for key in RangefinderParams.__annotations__}
    )


def beam_directions(params: RangefinderParams) -> NDArray:
    """Unit directions (N, 3) in the site frame; +z is the beam axis.

    One ray is the beam axis. A bundle adds rays spread evenly (sunflower
    pattern) over the cone of half-angle beam_half_angle_deg, to approximate
    the real beam footprint.
    """
    n = params.num_rays
    if n == 1:
        return np.array([[0.0, 0.0, 1.0]])
    i = np.arange(n)
    polar = np.radians(params.beam_half_angle_deg) * np.sqrt(i / (n - 1))
    azimuth = i * GOLDEN_ANGLE
    return np.column_stack(
        [
            np.sin(polar) * np.cos(azimuth),
            np.sin(polar) * np.sin(azimuth),
            np.cos(polar),
        ]
    )


def nearest_hit(distances: NDArray) -> int:
    """Index of the nearest hit in a bundle, or -1 when no ray hit."""
    distances = np.asarray(distances, dtype=float)
    hits = np.flatnonzero(distances >= 0)
    return int(hits[np.argmin(distances[hits])]) if len(hits) else -1


def apply_noise(
    distances: NDArray, params: RangefinderParams, rng: np.random.Generator
) -> NDArray:
    """Measured distances; NaN marks no-return (miss, dropout, or out of range).

    Input distances use -1 for no hit, as returned by the ray caster. Readings
    below min_range_m are also NaN, so a very close object looks like no return.
    """
    distances = np.asarray(distances, dtype=float)
    sigma = np.where(
        distances <= params.noise_switch_m,
        params.noise_sigma_near_m,
        params.noise_sigma_far_fraction * distances,
    )
    noise = rng.normal(0.0, 1.0, distances.shape) * sigma
    dropped = rng.uniform(size=distances.shape) < params.dropout_probability
    measured = distances + noise
    invalid = (
        (distances < 0)
        | dropped
        | (measured < params.min_range_m)
        | (measured > params.effective_max_range_m)
    )
    return np.where(invalid, np.nan, measured)
