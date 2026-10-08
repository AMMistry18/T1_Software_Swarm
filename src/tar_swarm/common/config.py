"""Configuration loading, with paths relative to the simulation YAML."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_config(path: str | Path, world: str | Path | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    settings = read_yaml(path)
    result = {
        "simulation": settings,
        "vehicles": read_yaml(path.parent / settings["vehicles_file"]),
        "world": read_yaml(
            Path(world).resolve() if world else path.parent / settings["world_file"]
        ),
        "mission": read_yaml(path.parent / settings["mission_file"]),
    }
    validate_config(result)
    return deepcopy(result)


def interval_steps(interval: float, timestep: float) -> int:
    ratio = interval / timestep
    if not np.isfinite(ratio) or ratio < 1 or not np.isclose(ratio, round(ratio)):
        raise ValueError("Intervals must be positive integer multiples of physics dt")
    return round(ratio)


def validate_config(config: dict[str, Any]) -> None:
    sim = config["simulation"]
    dt = sim["physics_timestep_s"]
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Physics timestep must be positive and finite")
    for key in (
        "controller_interval_s",
        "mission_interval_s",
        "visualization_interval_s",
        "log_interval_s",
        "duration_s",
    ):
        interval_steps(sim[key], dt)
    if not np.isfinite(sim["gravity_m_s2"]) or sim["gravity_m_s2"] <= 0:
        raise ValueError("Gravity magnitude must be positive")
    for vehicle in config["vehicles"].values():
        if vehicle["motor_count"] != 4:
            raise ValueError("Step 0 models quadrotors only")
        for key in (
            "arm_length_m",
            "propeller_diameter_in",
            "motor_mass_kg",
            "propeller_mass_kg",
            "arm_mass_kg",
            "motor_radius_m",
            "motor_height_m",
            "arm_width_m",
            "max_rotor_thrust_n",
            "reaction_torque_ratio_m",
        ):
            if not np.isfinite(vehicle[key]) or vehicle[key] <= 0:
                raise ValueError(f"Invalid vehicle parameter: {key}")
        if np.sqrt(2) * vehicle["arm_length_m"] <= (
            vehicle["propeller_diameter_in"] * 0.0254
        ):
            raise ValueError("Adjacent propeller disks overlap")
        for component in vehicle["components"]:
            values = [component["mass_kg"], *component["size_m"]]
            if not np.isfinite(values).all() or min(values) <= 0:
                raise ValueError("Component mass and dimensions must be positive")
            if not np.isfinite(component["pos_m"]).all():
                raise ValueError("Invalid component position")
        gains = vehicle["controller"]
        if gains["type"] != "cascaded_pd":
            raise ValueError("Only cascaded_pd is implemented in Step 0")
        for key in ("position_kp", "velocity_kd", "attitude_kp", "rate_kd"):
            if (
                len(gains[key]) != 3
                or not np.isfinite(gains[key]).all()
                or min(gains[key]) <= 0
            ):
                raise ValueError(f"Invalid controller gain: {key}")
        if not 0 < gains["max_tilt_deg"] < 80 or gains["max_acceleration_m_s2"] <= 0:
            raise ValueError("Invalid controller limits")
        mass = sum(c["mass_kg"] for c in vehicle["components"]) + 4 * sum(
            vehicle[k] for k in ("motor_mass_kg", "propeller_mass_kg", "arm_mass_kg")
        )
        if 4 * vehicle["max_rotor_thrust_n"] <= mass * sim["gravity_m_s2"]:
            raise ValueError("Configured rotor limits cannot support hover and takeoff")
        names = [c["name"] for c in vehicle["components"]]
        if len(set(names)) != len(names):
            raise ValueError("Component names must be unique within a vehicle")
    agents = config["mission"]["agents"]
    ids = [agent["id"] for agent in agents]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Agent IDs must be nonempty and unique")
    for agent in agents:
        if not np.isfinite(agent.get("yaw_rad", 0)):
            raise ValueError("Yaw must be finite")
        if agent["vehicle"] not in config["vehicles"]:
            raise ValueError("Unknown vehicle class")
        points = agent["waypoints"]
        times = [point["time_s"] for point in points]
        if (
            not times
            or times[0] != 0
            or not np.isfinite(times).all()
            or any(b <= a for a, b in zip(times, times[1:]))
        ):
            raise ValueError("Waypoint times must increase strictly from zero")
        positions = [point["position_m"] for point in points]
        if (
            np.asarray(positions).shape != (len(points), 3)
            or not np.isfinite(positions).all()
        ):
            raise ValueError("Waypoint positions must be finite 3-vectors")
        if len(agent["spawn_m"]) != 3 or not np.isfinite(agent["spawn_m"]).all():
            raise ValueError("Invalid spawn")
        if not np.allclose(agent["spawn_m"], positions[0]):
            raise ValueError("First reference must equal spawn for smooth takeoff")
    world = config["world"]
    if not np.isfinite(world["ground_half_size_m"]) or world["ground_half_size_m"] <= 0:
        raise ValueError("Ground dimensions must be positive")
    for obstacle in [world["ground_station"], *world["obstacles"]]:
        if (
            len(obstacle["size_m"]) != 3
            or not np.isfinite(obstacle["size_m"]).all()
            or min(obstacle["size_m"]) <= 0
        ):
            raise ValueError("Obstacle sizes must be positive finite 3-vectors")
        if (
            len(obstacle["position_m"]) != 3
            or not np.isfinite(obstacle["position_m"]).all()
        ):
            raise ValueError("Obstacle positions must be finite 3-vectors")
