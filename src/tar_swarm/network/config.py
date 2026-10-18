"""Small standalone-network config, reusing the shared YAML reader."""

from copy import deepcopy
from pathlib import Path

import numpy as np

from tar_swarm.common.config import read_yaml


def step_count(interval: float, dt: float) -> int:
    ratio = interval / dt
    if (
        not np.isfinite(ratio)
        or ratio < 1
        or not np.isclose(ratio, round(ratio), rtol=0, atol=1e-9)
    ):
        raise ValueError("Duration/graph interval must be positive multiples of dt_s")
    return round(ratio)


def validate_config(config: dict) -> dict:
    config = deepcopy(config)
    for key in ("dt_s", "duration_s", "graph_interval_s", "communication_radius_m"):
        value = config[key]
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    step_count(config["duration_s"], config["dt_s"])
    step_count(config["graph_interval_s"], config["dt_s"])
    if not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    bounds = np.asarray(config["plot_bounds_m"], dtype=float)
    if (
        bounds.shape != (2, 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] >= bounds[:, 1])
    ):
        raise ValueError("plot_bounds_m must be [[xmin, xmax], [ymin, ymax]]")
    agents = config["agents"]
    ids = [agent["id"] for agent in agents]
    if (
        not ids
        or any(not isinstance(name, str) or not name.strip() for name in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Agent IDs must be nonempty unique strings")
    if sum(agent["role"] == "ground_station" for agent in agents) != 1:
        raise ValueError("Exactly one ground_station is required")
    if not any(agent["role"] == "mapper" for agent in agents):
        raise ValueError("At least one mapper is required")
    for agent in agents:
        if agent["role"] not in {"ground_station", "mapper", "relay"}:
            raise ValueError("Unknown agent role")
        agent.setdefault("active", True)
        agent.setdefault("speed_m_s", 0.0)
        agent.setdefault("waypoints_m", [])
        if not isinstance(agent["active"], bool):
            raise ValueError("active must be a boolean")
        position = np.asarray(agent["position_m"], dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("position_m must be a finite 3-vector")
        speed = agent["speed_m_s"]
        if not np.isfinite(speed) or speed < 0:
            raise ValueError("speed_m_s must be nonnegative and finite")
        for waypoint in agent["waypoints_m"]:
            point = np.asarray(waypoint, dtype=float)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError("Waypoints must be finite 3-vectors")
            if point[2] != position[2]:
                raise ValueError("This kinematic demo requires fixed-altitude paths")
        if agent["role"] == "ground_station" and (
            not agent["active"] or speed != 0 or agent["waypoints_m"]
        ):
            raise ValueError(
                "Ground station must be active, stationary, without a path"
            )
    return config


def load_config(path: str | Path) -> dict:
    return validate_config(read_yaml(path))
