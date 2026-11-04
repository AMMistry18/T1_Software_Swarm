"""Compile simulator rollouts / baseline traces into exportable MissionPlan."""

from __future__ import annotations

import uuid
from typing import Any

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
        "HOLD actions keep the vehicle at the last waypoint until dependencies "
        "of subsequent actions are satisfied or timeout_s expires."
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
}


def compile_from_sim(
    scenario: PreparedScenario,
    sim: PlanningSimulator,
    source_commands: list[dict[str, Any]] | None = None,
) -> MissionPlan:
    """Build a compact waypoint mission with hold/release dependencies."""
    actions: list[WaypointAction] = []
    relay_hold_ids: list[str] = []
    cand_by_id = {c.id: c.position_m for c in scenario.candidates}
    tol = scenario.request.planning.arrival_tolerance_m

    if source_commands:
        for cmd in source_commands:
            if cmd.get("cmd") != "relay":
                continue
            rid = cmd["relay"]
            cid = cmd["candidate"]
            target = cand_by_id[cid]
            aid = f"relay_deploy_{rid}_{cid}"
            actions.append(
                WaypointAction(
                    action_id=aid,
                    drone_id=rid,
                    kind=ActionKind.TRANSIT,
                    waypoints_m=[target],
                    depends_on=(),
                    completion_tolerance_m=tol,
                    metadata={"candidate_id": cid},
                )
            )
            hold_id = f"relay_hold_{rid}_{cid}"
            actions.append(
                WaypointAction(
                    action_id=hold_id,
                    drone_id=rid,
                    kind=ActionKind.HOLD,
                    waypoints_m=[target],
                    hold_s=0.0,
                    depends_on=(aid,),
                    completion_tolerance_m=tol,
                )
            )
            relay_hold_ids.append(hold_id)

    survey_ids: list[str] = []
    for mapper in scenario.request.mappers:
        segs = [s for s in scenario.segments if s.mapper_id == mapper.id]
        segs.sort(key=lambda s: s.id)
        prev: tuple[str, ...] = tuple(relay_hold_ids)
        for seg in segs:
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
                    metadata={
                        "segment_id": seg.id,
                        "coverage_area_m2": seg.coverage_area_m2,
                    },
                )
            )
            survey_ids.append(aid)
            prev = (aid,)
        if scenario.request.planning.require_return_to_base:
            actions.append(
                WaypointAction(
                    action_id=f"return_{mapper.id}",
                    drone_id=mapper.id,
                    kind=ActionKind.TRANSIT,
                    waypoints_m=[mapper.launch_position_m],
                    depends_on=prev,
                    completion_tolerance_m=tol,
                )
            )

    if scenario.request.planning.require_return_to_base:
        deps = tuple(survey_ids)
        for relay in scenario.request.relays:
            actions.append(
                WaypointAction(
                    action_id=f"return_{relay.id}",
                    drone_id=relay.id,
                    kind=ActionKind.TRANSIT,
                    waypoints_m=[relay.launch_position_m],
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
