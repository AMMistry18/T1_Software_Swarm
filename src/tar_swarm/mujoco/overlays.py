"""Sensor overlays drawn into a viewer scene from measurement-side data only.

Geoms are appended after any existing ones (renderer.draw_overlays runs first).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from tar_swarm.common.config import read_yaml
from tar_swarm.perception.obstacles import sector_centers

CLEAR = (0.1, 0.9, 0.2, 0.9)
DANGER = (1.0, 0.1, 0.1, 0.9)
NO_RETURN = (0.6, 0.6, 0.6, 0.35)
GROUND_POINT = (0.55, 0.4, 0.2, 0.6)
OBSTACLE_POINT = (1.0, 0.85, 0.05, 1.0)
COMMANDED = (0.15, 0.35, 1.0, 0.9)
LIMITED = (1.0, 0.55, 0.0, 0.95)
OVERLAYS = ("rangefinder_beam", "lidar_points", "sector_ring", "velocity_arrows")


@dataclass
class Beam:
    origin: NDArray[np.float64]
    axis: NDArray[np.float64]
    distance_m: float
    max_range_m: float


@dataclass
class L2View:
    """One L2 scan after obstacle extraction, placed in the world for drawing."""

    body_position: NDArray[np.float64]
    heading: NDArray[np.float64]  # level frame to world (yaw only)
    points_level: NDArray[np.float64]
    ground_mask: NDArray[np.bool_]
    sector_distance_m: NDArray[np.float64]


@dataclass
class VelocityArrow:
    """Commanded vs avoidance-limited velocity of one vehicle (world frame)."""

    origin: NDArray[np.float64]
    commanded: NDArray[np.float64]
    limited: NDArray[np.float64]
    status: str


def load_overlay_settings(path: str | Path) -> dict[str, Any]:
    settings = read_yaml(path) or {}
    for name in settings:
        if name not in OVERLAYS:
            raise ValueError(f"Unknown overlay: {name}")
        if not isinstance(settings[name].get("enabled"), bool):
            raise ValueError(f"Overlay {name} needs enabled: true or false")
    beam = settings.get("rangefinder_beam", {})
    if beam.get("enabled") and not beam.get("d_stop_m", -1) >= 0:
        raise ValueError("rangefinder_beam d_stop_m must be non-negative")
    points = settings.get("lidar_points", {})
    if points.get("enabled"):
        count = points.get("max_points")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("lidar_points max_points must be a positive integer")
        if not isinstance(points.get("show_ground_points", False), bool):
            raise ValueError("lidar_points show_ground_points must be true or false")
    ring = settings.get("sector_ring", {})
    if ring.get("enabled") and not (
        ring.get("d_stop_m", -1) >= 0 and ring.get("no_return_length_m", -1) >= 0
    ):
        raise ValueError("sector_ring d_stop_m and no_return_length_m must be >= 0")
    arrows = settings.get("velocity_arrows", {})
    if arrows.get("enabled") and not arrows.get("scale_s", -1) > 0:
        raise ValueError("velocity_arrows scale_s must be positive")
    return settings


def add_line(
    scene: mujoco.MjvScene,
    start: NDArray,
    end: NDArray,
    width: float,
    rgba: Any,
) -> bool:
    if scene.ngeom >= scene.maxgeom or np.linalg.norm(end - start) < 1e-7:
        return False
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).ravel(),
        np.array(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, start, end)
    scene.ngeom += 1
    return True


def add_point(
    scene: mujoco.MjvScene, position: NDArray, radius: float, rgba: Any
) -> bool:
    if scene.ngeom >= scene.maxgeom:
        return False
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        np.asarray(position, dtype=np.float64),
        np.eye(3).ravel(),
        np.array(rgba, dtype=np.float32),
    )
    scene.ngeom += 1
    return True


def draw_lidar_points(
    scene: mujoco.MjvScene, view: L2View, max_points: int, show_ground: bool = True
) -> None:
    """Obstacle points first, then (optionally) ground points in what is left."""
    world = view.body_position + view.points_level @ view.heading.T
    obstacles = world[~view.ground_mask][:max_points]
    ground = world[view.ground_mask]
    room = max_points - len(obstacles)
    if show_ground and room > 0 and len(ground):
        ground = ground[:: max(1, int(np.ceil(len(ground) / room)))][:room]
    else:
        ground = ground[:0]
    for point in obstacles:
        add_point(scene, point, 0.03, OBSTACLE_POINT)
    for point in ground:
        add_point(scene, point, 0.02, GROUND_POINT)


def draw_sector_ring(
    scene: mujoco.MjvScene, view: L2View, d_stop_m: float, no_return_length_m: float
) -> None:
    """One line per sector from the body, length = nearest obstacle distance.

    Sectors with no return get a short grey stub instead.
    """
    angles = sector_centers(len(view.sector_distance_m))
    directions = np.column_stack(
        [np.cos(angles), np.sin(angles), np.zeros_like(angles)]
    )
    for direction, distance in zip(
        directions @ view.heading.T, view.sector_distance_m, strict=True
    ):
        if np.isfinite(distance):
            color = DANGER if distance < d_stop_m else CLEAR
        else:
            color, distance = NO_RETURN, no_return_length_m
        add_line(
            scene,
            view.body_position,
            view.body_position + distance * direction,
            0.004,
            color,
        )


def draw_velocity_arrow(
    scene: mujoco.MjvScene, arrow: VelocityArrow, scale_s: float
) -> None:
    """Commanded velocity in blue; limited velocity green when equal to it,
    orange when limited, red when stopped. Each has a sphere at its tip."""
    if arrow.status == "stop" or arrow.status == "stale_hold":
        color = DANGER
    elif np.allclose(arrow.commanded, arrow.limited, atol=1e-6):
        color = CLEAR
    else:
        color = LIMITED
    for velocity, rgba, width in (
        (arrow.commanded, COMMANDED, 0.012),
        (arrow.limited, color, 0.018),
    ):
        tip = arrow.origin + scale_s * velocity
        if add_line(scene, arrow.origin, tip, width, rgba):
            add_point(scene, tip, 1.8 * width, rgba)
        elif rgba is color:
            add_point(scene, arrow.origin, 0.05, rgba)  # zero velocity: a dot


def draw_rangefinder_beam(scene: mujoco.MjvScene, beam: Beam, d_stop_m: float) -> None:
    """Sensor to hit point; green when clear, red inside d_stop_m.

    A no-return reading is drawn faint grey out to the effective maximum range.
    """
    if np.isfinite(beam.distance_m):
        color = DANGER if beam.distance_m < d_stop_m else CLEAR
        length = beam.distance_m
    else:
        color, length = NO_RETURN, beam.max_range_m
    add_line(scene, beam.origin, beam.origin + length * beam.axis, 0.006, color)


def needs_l2(settings: dict[str, Any] | None) -> bool:
    return bool(settings) and any(
        settings.get(name, {}).get("enabled")
        for name in ("lidar_points", "sector_ring")
    )


def draw_sensor_overlays(
    scene: mujoco.MjvScene,
    settings: dict[str, Any],
    beams: dict[str, Beam],
    l2: dict[str, L2View] | None = None,
    arrows: dict[str, VelocityArrow] | None = None,
) -> None:
    beam_settings = settings.get("rangefinder_beam", {})
    if beam_settings.get("enabled"):
        for beam in beams.values():
            draw_rangefinder_beam(scene, beam, beam_settings["d_stop_m"])
    ring = settings.get("sector_ring", {})
    points = settings.get("lidar_points", {})
    for view in (l2 or {}).values():
        if ring.get("enabled"):
            draw_sector_ring(scene, view, ring["d_stop_m"], ring["no_return_length_m"])
        if points.get("enabled"):
            draw_lidar_points(
                scene,
                view,
                points["max_points"],
                points.get("show_ground_points", False),
            )
    velocity = settings.get("velocity_arrows", {})
    if velocity.get("enabled"):
        for arrow in (arrows or {}).values():
            draw_velocity_arrow(scene, arrow, velocity["scale_s"])
