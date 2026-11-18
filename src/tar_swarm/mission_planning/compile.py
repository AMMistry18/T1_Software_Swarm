"""Compile simulator rollouts / baseline traces into exportable MissionPlan."""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np

from tar_swarm.mission_planning.schemas import (
    ActionKind,
    MissionPlan,
    PreparedScenario,
    WaypointAction,
)
from tar_swarm.mission_planning.simulator import PlanningSimulator

EXECUTION_CONTRACT = {
    "version": "1.0.0",
    "coordinate_frame": "local_enu_m",
    "altitude": "absolute_z_meters_local_datum",
    "dependency_model": (
        "Prefer readiness conditions (depends_on action_ids completed within "
        "completion_tolerance_m) over synchronized wall-clock timestamps."
    ),
    "hold_semantics": (
        "HOLD actions keep the vehicle at the last waypoint until hold_s elapses "
        "(after arrival) and dependencies of subsequent actions are satisfied."
    ),
    "missed_dependency": (
        "Pause safely at last achieved waypoint, report the unmet dependency, "
        "and require replanning or the configured fallback. Do not auto-upload "
        "to hardware."
    ),
    "degraded_link": (
        "On degraded/lost link during execution, pause survey release of new "
        "segments where feasible and request replan. Connectivity in planning "
        "is a distance assumption, not measured RF."
    ),
    "not_established_by_validator": [
        "Real RF throughput for LiDAR/point-cloud traffic",
        "Actual propulsion/battery curves",
        "Wind, GNSS integrity, and flight-controller tracking error",
        "Obstacle/terrain clearance beyond planar separation checks",
    ],
    "compile_notes": (
        "Commands are compiled in decision order with unique action_ids so "
        "revisits are not collapsed. Relays and mappers run concurrent chains; "
        "survey release does not wait for later relay redeploys. Idle gaps "
        "between a drone's own commands are encoded as HOLD hold_s."
    ),
}


def _travel_time_s(
    start_xyz: tuple[float, float, float] | np.ndarray,
    end_xyz: tuple[float, float, float] | np.ndarray,
    speed_m_s: float,
) -> float:
    a = np.asarray(start_xyz, dtype=float)[:2]
    b = np.asarray(end_xyz, dtype=float)[:2]
    return float(np.linalg.norm(b - a) / max(float(speed_m_s), 1e-6))


def compile_from_sim(
    scenario: PreparedScenario,
    sim: PlanningSimulator,
    source_commands: list[dict[str, Any]] | None = None,
) -> MissionPlan:
    """Build a waypoint mission that preserves timed relay/release interleaving.

    Earlier compiles dumped every relay move up front (with reusable action_ids),
    so network playback skipped revisits and forced all redeploys before survey.
    This version:
    - gives every command occurrence a unique action_id
    - chains each drone only to its own previous action
    - starts survey segments on release (not after every future relay hold)
    - encodes idle gaps as HOLD ``hold_s`` from decision timing
    """
    request = scenario.request
    tol = request.planning.arrival_tolerance_m
    decision_dt = float(request.planning.decision_dt_s)
    cand_by_id = {c.id: c.position_m for c in scenario.candidates}
    speed = {d.id: float(d.speed_m_s) for d in request.drones}
    launch = {d.id: d.launch_position_m for d in request.drones}
    segs_by_mapper = {
        m.id: sorted(
            [s for s in scenario.segments if s.mapper_id == m.id],
            key=lambda s: s.id,
        )
        for m in request.mappers
    }

    actions: list[WaypointAction] = []
    last_action: dict[str, str] = {}
    pose: dict[str, tuple[float, float, float]] = {
        d.id: tuple(map(float, d.launch_position_m)) for d in request.drones
    }
    # Earliest decision-time (s) when this drone is free for a new command.
    free_at: dict[str, float] = {d.id: 0.0 for d in request.drones}
    next_seg_idx = {m.id: 0 for m in request.mappers}
    survey_ids: list[str] = []
    seq = 0

    def _deps_for(drone_id: str) -> tuple[str, ...]:
        prev = last_action.get(drone_id)
        return (prev,) if prev is not None else ()

    def _append(action: WaypointAction) -> None:
        nonlocal actions
        actions.append(action)
        last_action[action.drone_id] = action.action_id

    commands = list(source_commands or [])
    if not commands:
        # No trace: fall back to releasing every segment with no relay moves.
        for mapper in request.mappers:
            prev: tuple[str, ...] = ()
            for seg in segs_by_mapper[mapper.id]:
                aid = f"survey_{seg.id}"
                duration = seg.length_m / max(mapper.speed_m_s, 1e-6)
                actions.append(
                    WaypointAction(
                        action_id=aid,
                        drone_id=mapper.id,
                        kind=ActionKind.SURVEY,
                        waypoints_m=list(seg.waypoints_m),
                        depends_on=prev,
                        completion_tolerance_m=tol,
                        expected_duration_s=duration,
                        metadata={"segment_id": seg.id},
                    )
                )
                survey_ids.append(aid)
                prev = (aid,)
                last_action[mapper.id] = aid
    else:
        for cmd_i, cmd in enumerate(commands):
            t_cmd = float(cmd_i) * decision_dt
            kind = cmd.get("cmd")
            if kind == "relay":
                rid = str(cmd["relay"])
                cid = str(cmd["candidate"])
                target = tuple(map(float, cand_by_id[cid]))
                seq += 1
                # Idle at current pose until this decision (after prior motion).
                idle = max(0.0, t_cmd - free_at[rid])
                if idle > 1e-6 or not _deps_for(rid):
                    # Always park with an explicit hold when waiting, so the next
                    # transit cannot start early in network playback.
                    hold_id = f"relay_wait_{rid}_{seq:04d}"
                    _append(
                        WaypointAction(
                            action_id=hold_id,
                            drone_id=rid,
                            kind=ActionKind.HOLD,
                            waypoints_m=[pose[rid]],
                            hold_s=float(idle),
                            depends_on=_deps_for(rid),
                            completion_tolerance_m=tol,
                            metadata={"decision_index": cmd_i},
                        )
                    )
                deploy_id = f"relay_deploy_{rid}_{cid}_{seq:04d}"
                travel = _travel_time_s(pose[rid], target, speed[rid])
                _append(
                    WaypointAction(
                        action_id=deploy_id,
                        drone_id=rid,
                        kind=ActionKind.TRANSIT,
                        waypoints_m=[target],
                        depends_on=_deps_for(rid),
                        completion_tolerance_m=tol,
                        expected_duration_s=travel,
                        metadata={
                            "candidate_id": cid,
                            "decision_index": cmd_i,
                            "seq": seq,
                        },
                    )
                )
                pose[rid] = target
                free_at[rid] = max(free_at[rid], t_cmd) + travel
            elif kind == "release":
                mid = str(cmd["mapper"])
                segs = segs_by_mapper[mid]
                si = next_seg_idx[mid]
                if si >= len(segs):
                    continue
                seg = segs[si]
                seq += 1
                idle = max(0.0, t_cmd - free_at[mid])
                if idle > 1e-6 or si == 0:
                    wait_id = f"mapper_wait_{mid}_{seq:04d}"
                    _append(
                        WaypointAction(
                            action_id=wait_id,
                            drone_id=mid,
                            kind=ActionKind.HOLD,
                            waypoints_m=[pose[mid]],
                            hold_s=float(idle),
                            depends_on=_deps_for(mid),
                            completion_tolerance_m=tol,
                            metadata={"decision_index": cmd_i},
                        )
                    )
                survey_id = f"survey_{seg.id}"
                duration = float(seg.length_m / max(speed[mid], 1e-6))
                # Also include transit from current pose to first survey waypoint.
                first = tuple(map(float, seg.waypoints_m[0]))
                duration += _travel_time_s(pose[mid], first, speed[mid])
                _append(
                    WaypointAction(
                        action_id=survey_id,
                        drone_id=mid,
                        kind=ActionKind.SURVEY,
                        waypoints_m=list(seg.waypoints_m),
                        depends_on=_deps_for(mid),
                        completion_tolerance_m=tol,
                        expected_duration_s=duration,
                        metadata={
                            "segment_id": seg.id,
                            "coverage_area_m2": seg.coverage_area_m2,
                            "decision_index": cmd_i,
                        },
                    )
                )
                survey_ids.append(survey_id)
                pose[mid] = tuple(map(float, seg.waypoints_m[-1]))
                free_at[mid] = max(free_at[mid], t_cmd) + duration
                next_seg_idx[mid] = si + 1
            elif kind == "return":
                # Explicit return command from baselines; handled below as well.
                continue

    if request.planning.require_return_to_base:
        for drone in request.drones:
            home = launch[drone.id]
            if np.allclose(pose[drone.id][:2], np.asarray(home)[:2], atol=tol):
                continue
            deps = _deps_for(drone.id)
            if drone.role.value == "relay" and survey_ids:
                # Relays may return once surveys are done (matches prior contract).
                deps = tuple(dict.fromkeys([*deps, *survey_ids]))
            elif drone.role.value == "mapper" and survey_ids:
                # Mapper return after its own last survey (already in deps).
                pass
            aid = f"return_{drone.id}"
            _append(
                WaypointAction(
                    action_id=aid,
                    drone_id=drone.id,
                    kind=ActionKind.TRANSIT,
                    waypoints_m=[home],
                    depends_on=deps,
                    completion_tolerance_m=tol,
                )
            )

    return MissionPlan(
        mission_id=f"mission_{uuid.uuid4().hex[:10]}",
        actions=actions,
        expected_duration_s=float(sim.metrics.elapsed_s),
        expected_coverage_fraction=float(sim.metrics.coverage_fraction),
        validation={},
        execution_contract=dict(EXECUTION_CONTRACT),
    )
