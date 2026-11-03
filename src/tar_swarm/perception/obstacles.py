"""Obstacle distances from L2 scans: ground removal, height band, sectors.

Frames: the *level frame* has its origin at the body origin, z up along
gravity, and x along the body heading projected onto the horizontal plane.
Sector 0 is centred on the heading; indices increase counter-clockwise (left).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ObstacleParams:
    num_sectors: int
    band_below_m: float
    band_above_m: float
    ransac_iterations: int
    ransac_threshold_m: float
    ground_max_tilt_deg: float
    ground_min_points: int
    min_range_m: float


@dataclass
class GroundPlane:
    normal: NDArray[np.float64]  # unit, level frame, z component > 0
    offset: float  # plane: normal @ p + offset = 0


@dataclass
class ObstacleScan:
    points_level: NDArray[np.float64]
    ground_mask: NDArray[np.bool_]
    ground: GroundPlane | None
    sector_distance_m: NDArray[np.float64]  # NaN = no return in that sector
    sector_count: NDArray[np.int64]


def validate_obstacle_params(params: dict[str, Any]) -> None:
    for key in ("enabled", *ObstacleParams.__annotations__):
        if key not in params:
            raise ValueError(f"Missing obstacle parameter: {key}")
    if not isinstance(params["enabled"], bool):
        raise ValueError("Obstacle enabled must be true or false")
    for key in ("num_sectors", "ransac_iterations", "ground_min_points"):
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 3:
            raise ValueError(f"Obstacle {key} must be an integer >= 3")
    for key in ("band_below_m", "band_above_m", "min_range_m"):
        if not np.isfinite(params[key]) or params[key] < 0:
            raise ValueError(f"Obstacle {key} must be non-negative")
    if not np.isfinite(params["ransac_threshold_m"]) or (
        params["ransac_threshold_m"] <= 0
    ):
        raise ValueError("Obstacle ransac_threshold_m must be positive")
    if not 0 < params["ground_max_tilt_deg"] < 90:
        raise ValueError("Obstacle ground_max_tilt_deg must be in (0, 90)")


def load_obstacle_params(
    perception: dict[str, Any] | None, vehicle_type: str
) -> ObstacleParams | None:
    """Parameters for a vehicle type, or None when absent."""
    raw = (perception or {}).get("l2_obstacles", {}).get(vehicle_type)
    if raw is None:
        return None
    validate_obstacle_params(raw)
    return ObstacleParams(**{key: raw[key] for key in ObstacleParams.__annotations__})


def obstacles_enabled(perception: dict[str, Any] | None, vehicle_type: str) -> bool:
    raw = (perception or {}).get("l2_obstacles", {}).get(vehicle_type)
    return bool(raw and raw.get("enabled"))


def heading_rotation(body_rotation: NDArray) -> NDArray:
    """Yaw-only rotation (level frame to world) of a body-to-world rotation."""
    yaw = np.arctan2(body_rotation[1, 0], body_rotation[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def to_level_frame(
    points_sensor: NDArray,
    mount_rotation: NDArray,
    mount_position: NDArray,
    body_rotation: NDArray,
) -> NDArray:
    """Sensor-frame points to the level frame.

    ASSUMPTION: roll and pitch come from the simulated attitude (the IMU on
    hardware). Only the tilt part of body_rotation is used; yaw is removed.
    """
    tilt = heading_rotation(body_rotation).T @ body_rotation
    return (points_sensor @ mount_rotation.T + mount_position) @ tilt.T


def fit_ground_plane(
    points: NDArray, params: ObstacleParams, rng: np.random.Generator
) -> GroundPlane | None:
    """RANSAC plane with near-vertical normal and the most inliers, refined by SVD.

    Uses only the measured points; the true ground height is never consulted.
    Returns None when no candidate reaches ground_min_points inliers.
    """
    if len(points) < max(3, params.ground_min_points):
        return None
    sample = points[rng.integers(0, len(points), (params.ransac_iterations, 3))]
    normals = np.cross(sample[:, 1] - sample[:, 0], sample[:, 2] - sample[:, 0])
    length = np.linalg.norm(normals, axis=1)
    usable = length > 1e-9
    normals = normals[usable] / length[usable, None]
    normals *= np.where(normals[:, 2] < 0, -1.0, 1.0)[:, None]
    level = normals[:, 2] >= np.cos(np.radians(params.ground_max_tilt_deg))
    normals = normals[level]
    if not len(normals):
        return None
    offsets = -np.einsum("ij,ij->i", normals, sample[usable][level][:, 0])
    inliers = np.abs(points @ normals.T + offsets) < params.ransac_threshold_m
    best = int(np.argmax(inliers.sum(axis=0)))
    chosen = inliers[:, best]
    if chosen.sum() < params.ground_min_points:
        return None
    centroid = points[chosen].mean(axis=0)
    # Thin SVD: the full one builds an N x N matrix (~300 MB per L2 scan).
    normal = np.linalg.svd(points[chosen] - centroid, full_matrices=False)[2][-1]
    normal = normal if normal[2] > 0 else -normal
    if normal[2] < np.cos(np.radians(params.ground_max_tilt_deg)):
        normal = normals[best]
    return GroundPlane(normal, float(-normal @ centroid))


def ground_mask(points: NDArray, ground: GroundPlane | None, threshold: float):
    """Points on or below the ground plane (below = noise)."""
    if ground is None:
        return np.zeros(len(points), dtype=bool)
    return points @ ground.normal + ground.offset < threshold


def sector_distances(
    points_level: NDArray, params: ObstacleParams
) -> tuple[NDArray, NDArray]:
    """Nearest horizontal distance and point count per sector (NaN = no return)."""
    z = points_level[:, 2]
    horizontal = np.hypot(points_level[:, 0], points_level[:, 1])
    keep = (
        (z >= -params.band_below_m)
        & (z <= params.band_above_m)
        & (horizontal >= params.min_range_m)
    )
    width = 2 * np.pi / params.num_sectors
    azimuth = np.arctan2(points_level[keep, 1], points_level[keep, 0])
    index = np.floor((azimuth + width / 2) / width).astype(int) % params.num_sectors
    nearest = np.full(params.num_sectors, np.inf)
    np.minimum.at(nearest, index, horizontal[keep])
    count = np.bincount(index, minlength=params.num_sectors)
    return np.where(np.isinf(nearest), np.nan, nearest), count


def sector_centers(num_sectors: int) -> NDArray:
    """Sector centre azimuths in the level frame, radians."""
    return np.arange(num_sectors) * 2 * np.pi / num_sectors


def extract_obstacles(
    points_sensor: NDArray,
    mount_rotation: NDArray,
    mount_position: NDArray,
    body_rotation: NDArray,
    params: ObstacleParams,
    rng: np.random.Generator,
) -> ObstacleScan:
    points = to_level_frame(
        points_sensor, mount_rotation, mount_position, body_rotation
    )
    ground = fit_ground_plane(points, params, rng)
    on_ground = ground_mask(points, ground, params.ransac_threshold_m)
    distance, count = sector_distances(points[~on_ground], params)
    return ObstacleScan(points, on_ground, ground, distance, count)
