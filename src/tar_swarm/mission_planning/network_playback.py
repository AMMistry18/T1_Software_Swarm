"""Replay a mission-planning MissionPlan inside the packet/network simulator.

Phase-1 bridge: planning still trains on the fast kinematic + distance graph;
this module converts an exported plan into a network scene so you can watch
real radio-range links, routing, and packet traffic instead of (or in addition
to) the planning GIF replay.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from tar_swarm.mission_planning.schemas import (
    ActionKind,
    MissionPlan,
    MissionRequest,
    Role,
    WaypointAction,
)
from tar_swarm.network.config import validate_config
from tar_swarm.network.results import save_results
from tar_swarm.network.simulator import Simulator


def _as_xyz(point) -> np.ndarray:
    arr = np.asarray(point, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"expected 3-vector, got {arr.shape}")
    return arr


def _flatten_altitude(points: list[tuple[float, float, float]], z: float):
    """Network kinematics currently require fixed-altitude paths per agent."""
    return [(float(p[0]), float(p[1]), float(z)) for p in points]


def mission_actions_for_drone(
    plan: MissionPlan, drone_id: str
) -> list[WaypointAction]:
    return [a for a in plan.actions if a.drone_id == drone_id]


class MissionPlanPolicy:
    """Advance each drone through MissionPlan actions, honoring depends_on."""

    def __init__(
        self,
        plan: MissionPlan,
        *,
        arrival_tol_m: float = 1.0,
        agent_altitudes: Mapping[str, float] | None = None,
    ):
        self.plan = plan
        self.arrival_tol_m = float(arrival_tol_m)
        self.agent_altitudes = dict(agent_altitudes or {})
        self._by_drone: dict[str, list[WaypointAction]] = {}
        for action in plan.actions:
            self._by_drone.setdefault(action.drone_id, []).append(action)
        self._index = {drone: 0 for drone in self._by_drone}
        self._completed: set[str] = set()
        self._hold_until: dict[str, float] = {}

    def _action_ready(self, action: WaypointAction) -> bool:
        return all(dep in self._completed for dep in action.depends_on)

    def _mark_arrived(self, action: WaypointAction, time_s: float) -> None:
        if action.kind == ActionKind.HOLD and action.hold_s > 0:
            until = self._hold_until.get(action.action_id)
            if until is None:
                self._hold_until[action.action_id] = time_s + float(action.hold_s)
                return
            if time_s + 1e-9 < until:
                return
        self._completed.add(action.action_id)

    def target(
        self, agent_id: str, time_s: float, positions: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        pos = np.asarray(positions[agent_id], dtype=float)
        z = float(self.agent_altitudes.get(agent_id, pos[2]))
        actions = self._by_drone.get(agent_id, [])
        idx = self._index.get(agent_id, 0)

        # Skip already-finished actions (e.g. after reload).
        while idx < len(actions) and actions[idx].action_id in self._completed:
            idx += 1
        self._index[agent_id] = idx

        if idx >= len(actions):
            return pos.copy()

        action = actions[idx]
        if not self._action_ready(action):
            # Wait in place until dependencies complete.
            return pos.copy()

        waypoints = _flatten_altitude(list(action.waypoints_m), z)
        if not waypoints:
            self._completed.add(action.action_id)
            self._index[agent_id] = idx + 1
            return pos.copy()

        # Walk sequential waypoints inside the action.
        goal = np.asarray(waypoints[0], dtype=float)
        for wp in waypoints:
            candidate = np.asarray(wp, dtype=float)
            if float(np.linalg.norm(pos - candidate)) > self.arrival_tol_m:
                goal = candidate
                break
            goal = candidate
        else:
            # At final waypoint of this action.
            self._mark_arrived(action, time_s)
            if action.action_id in self._completed:
                self._index[agent_id] = idx + 1
            return goal.copy()

        if float(np.linalg.norm(pos - goal)) <= self.arrival_tol_m:
            # Reached current goal; if it was the last, complete above.
            if np.allclose(goal, waypoints[-1]):
                self._mark_arrived(action, time_s)
                if action.action_id in self._completed:
                    self._index[agent_id] = idx + 1
        return goal.copy()


def build_network_config_from_mission(
    request: MissionRequest,
    plan: MissionPlan,
    *,
    dt_s: float = 0.1,
    graph_interval_s: float = 0.5,
    seed: int = 0,
    duration_margin_s: float = 30.0,
    radio_range_m: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a validated network.yaml-compatible scene from a planned mission."""
    usable = float(request.communication.usable_range_m)
    ranges = radio_range_m or {
        "ground_station": usable,
        "mapper": usable,
        "relay": usable,
    }
    xs = [p[0] for p in request.region_polygon_m] + [request.base_position_m[0]]
    ys = [p[1] for p in request.region_polygon_m] + [request.base_position_m[1]]
    pad = max(usable * 0.25, 20.0)
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad
    duration = max(float(plan.expected_duration_s) + duration_margin_s, 60.0)
    # duration_s / graph_interval_s must be integer multiples of dt_s.
    duration = float(np.ceil(duration / dt_s) * dt_s)

    agents: list[dict[str, Any]] = [
        {
            "id": "ground",
            "role": "ground_station",
            "position_m": [
                float(request.base_position_m[0]),
                float(request.base_position_m[1]),
                0.0,
            ],
        }
    ]
    for drone in request.drones:
        z = float(drone.launch_position_m[2])
        role = (
            "mapper"
            if drone.role == Role.MAPPER
            else "relay"
            if drone.role == Role.RELAY
            else "ground_station"
        )
        # Flatten plan waypoints to this drone's launch altitude for network kinematics.
        path: list[list[float]] = []
        for action in mission_actions_for_drone(plan, drone.id):
            for wp in _flatten_altitude(list(action.waypoints_m), z):
                if not path or path[-1] != list(wp):
                    path.append(list(wp))
        agents.append(
            {
                "id": drone.id,
                "role": role,
                "position_m": list(map(float, drone.launch_position_m)),
                "speed_m_s": float(drone.speed_m_s),
                # Waypoints listed for config validation / headless WaypointPolicy
                # fallback; MissionPlanPolicy overrides motion with dependencies.
                "waypoints_m": path,
            }
        )

    config = {
        "seed": int(seed),
        "dt_s": float(dt_s),
        "duration_s": duration,
        "graph_interval_s": float(graph_interval_s),
        "radio_range_m": {k: float(v) for k, v in ranges.items()},
        "plot_bounds_m": [[float(xmin), float(xmax)], [float(ymin), float(ymax)]],
        "grid_step_m": float(max(10.0, min(xmax - xmin, ymax - ymin) / 8.0)),
        "new_relay_altitude_m": 15.0,
        "new_relay_speed_m_s": 4.0,
        "new_mapper_altitude_m": float(request.sensor.survey_altitude_m),
        "new_mapper_speed_m_s": 5.0,
        "packets": {
            "link_speed_zones": [
                {"up_to_range_fraction": 0.3, "mbps": 30.0},
                {"up_to_range_fraction": 0.6, "mbps": 10.0},
                {"up_to_range_fraction": 1.0, "mbps": 1.0},
            ],
            "per_packet_overhead_s": 0.0003,
        },
        "obstacles_m": [],
        "agents": agents,
        # Non-network metadata for humans / tools.
        "mission_playback": {
            "mission_id": plan.mission_id,
            "region_polygon_m": [list(p) for p in request.region_polygon_m],
            "expected_coverage_fraction": plan.expected_coverage_fraction,
        },
    }
    return validate_config(config)


def play_mission_on_network(
    request: MissionRequest,
    plan: MissionPlan,
    output_dir: str | Path,
    *,
    animate: bool = False,
    dt_s: float = 0.1,
    seed: int = 0,
    radio_range_m: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run the network simulator on a planned mission; save packet/link outputs."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    config = build_network_config_from_mission(
        request,
        plan,
        dt_s=dt_s,
        seed=seed,
        radio_range_m=radio_range_m,
    )

    altitudes = {
        d.id: float(d.launch_position_m[2]) for d in request.drones
    }
    altitudes["ground"] = 0.0
    policy = MissionPlanPolicy(
        plan,
        arrival_tol_m=float(request.planning.arrival_tolerance_m),
        agent_altitudes=altitudes,
    )
    simulator = Simulator(config, policy=policy)

    if animate:
        from tar_swarm.network.visualization import animate as network_animate

        if not network_animate(simulator):
            return {
                "saved": False,
                "output_dir": str(output_dir),
                "note": "Results not saved (unchecked in the window).",
            }
    else:
        simulator.run()

    # save_results creates the output directory (must not already exist).
    summary = save_results(simulator, output_dir)
    (output_dir / "network_scene.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output_dir / "mission_plan.json").write_text(
        plan.model_dump_json(indent=2), encoding="utf-8"
    )
    from tar_swarm.network.visualization import save_plot

    plot_path = output_dir / "trajectory_connectivity.png"
    save_plot(simulator, plot_path)
    summary_path = output_dir / "playback_summary.json"
    payload = {
        "saved": True,
        "output_dir": str(output_dir),
        "plot": str(plot_path),
        "network_summary": summary,
    }
    summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload
