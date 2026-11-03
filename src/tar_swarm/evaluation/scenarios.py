"""AV4 avoidance test scenarios: one drone flying a straight lane into a test world.

A scenario is the base configuration with an avoidance world, a single agent
(the mapper with the L2, or a relay with the TF-Luna), avoidance enabled for it,
and a straight +x mission whose peak speed is set by the caller.
"""

from pathlib import Path
from typing import Any

import numpy as np

from tar_swarm.common.config import load_config, read_yaml, validate_config

ROOT = Path(__file__).resolve().parents[3]
WORLDS = ("avoid_wall", "avoid_pole", "avoid_thin", "avoid_offaxis", "clear_path")
# Sensor name -> (agent id, vehicle type, spawn height in m).
SENSORS = {
    "l2": ("mapper_0", "mapping_drone", 0.23),
    "rangefinder": ("relay_0", "relay_drone", 0.05),
}
ALTITUDE_M = 2.0
LANE_END_X_M = 30.0
TAKEOFF_S, HOVER_S, SETTLE_S = 4.0, 2.0, 4.0
# Ground beyond the farthest ray footprint; covers sensor offsets and tracking error.
EDGE_MARGIN_M = 1.0
# Peak / mean speed of a single quintic segment.
QUINTIC_PEAK_RATIO = 1.875


def lane_waypoints(spawn_z: float, peak_speed_m_s: float) -> list[dict[str, Any]]:
    """Take off, hover, fly the lane in one quintic segment, settle at the end."""
    # Rounded up to 0.1 s so the duration is a whole number of physics steps;
    # the realised peak speed is then at most the requested one.
    cruise = (
        float(np.ceil(QUINTIC_PEAK_RATIO * LANE_END_X_M / peak_speed_m_s * 10)) / 10
    )
    start = TAKEOFF_S + HOVER_S
    return [
        {"time_s": 0, "position_m": [0.0, 0.0, spawn_z]},
        {"time_s": TAKEOFF_S, "position_m": [0.0, 0.0, ALTITUDE_M]},
        {"time_s": start, "position_m": [0.0, 0.0, ALTITUDE_M]},
        {"time_s": start + cruise, "position_m": [LANE_END_X_M, 0.0, ALTITUDE_M]},
        {
            "time_s": start + cruise + SETTLE_S,
            "position_m": [LANE_END_X_M, 0.0, ALTITUDE_M],
        },
    ]


def coverage_range_m(config: dict[str, Any]) -> float:
    """Longest ray any sensor can cast; the ground must extend this far beyond flight.

    Uses the L2 range even in TF-Luna scenarios so every world is checked alike.
    """
    sensors = config.get("sensors", {})
    ranges = [p["max_range_m"] for p in sensors.get("lidar", {}).values()]
    ranges += [p["max_range_m"] for p in sensors.get("rangefinder", {}).values()]
    return max(ranges)


def ground_edge_margin(config: dict[str, Any], positions: np.ndarray) -> float:
    """Ground half-size minus (farthest horizontal position + ray range + margin).

    Negative means some ray could land beyond the ground edge and be lost.
    """
    reach = np.max(np.abs(np.asarray(positions)[:, :2]))
    return float(
        config["world"]["ground_half_size_m"]
        - (reach + coverage_range_m(config) + EDGE_MARGIN_M)
    )


def check_ground_edge(config: dict[str, Any], positions: np.ndarray) -> None:
    """Hard check: raise if any ray could be lost off the ground edge."""
    margin = ground_edge_margin(config, positions)
    if margin < 0:
        raise ValueError(
            f"Ground too small: rays could leave the ground edge by {-margin:.2f} m"
        )


def avoidance_config(
    world: str,
    sensor: str,
    peak_speed_m_s: float = 2.0,
    *,
    seed: int = 42,
    noise: bool = True,
    base: Path = ROOT / "configs/simulation.yaml",
) -> dict[str, Any]:
    if world not in WORLDS:
        raise ValueError(f"Unknown avoidance world: {world}")
    if sensor not in SENSORS:
        raise ValueError(f"Unknown avoidance sensor: {sensor}")
    if not np.isfinite(peak_speed_m_s) or peak_speed_m_s <= 0:
        raise ValueError("Peak speed must be positive")
    config = load_config(base)
    config["world"] = read_yaml(base.parent / "worlds" / f"{world}.yaml")
    agent_id, vehicle, spawn_z = SENSORS[sensor]
    waypoints = lane_waypoints(spawn_z, peak_speed_m_s)
    config["mission"] = {
        "hover_windows_s": [[TAKEOFF_S, TAKEOFF_S + HOVER_S]],
        "agents": [
            {
                "id": agent_id,
                "vehicle": vehicle,
                "spawn_m": waypoints[0]["position_m"],
                "yaw_rad": 0.0,
                "waypoints": waypoints,
            }
        ],
    }
    config["simulation"]["seed"] = seed
    config["simulation"]["duration_s"] = waypoints[-1]["time_s"]
    if sensor == "l2":
        config["perception"]["l2_obstacles"][vehicle]["enabled"] = True
    config["avoidance"][vehicle]["enabled"] = True
    config["avoidance"][vehicle]["sources"] = [sensor]
    if not noise:
        lidar = config["sensors"]["lidar"]["mapping_drone"]
        lidar["range_noise_m"] = lidar["dropout_probability"] = 0.0
        for luna in config["sensors"]["rangefinder"].values():
            luna["noise_sigma_near_m"] = luna["noise_sigma_far_fraction"] = 0.0
            luna["dropout_probability"] = 0.0
    validate_config(config)
    planned = np.array([w["position_m"] for w in waypoints])
    check_ground_edge(config, planned)
    return config
