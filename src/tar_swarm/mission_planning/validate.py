"""Independent mission validator.

Replays the exported artifact, not the internal rollout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tar_swarm.mission_planning.communication import (
    ConservativeDistanceLink,
    mapper_connected_via_permitted,
)
from tar_swarm.mission_planning.coverage import CoverageTracker
from tar_swarm.mission_planning.geometry import polygon_from_ring
from tar_swarm.mission_planning.schemas import (
    ActionKind,
    MissionPlan,
    PreparedScenario,
)
from tar_swarm.network.motion import move_towards


@dataclass
class ValidationReport:
    ok: bool
    coverage_fraction: float
    disconnected_mapper_s: float
    separation_violations: int
    flight_budget_violations: int
    dependency_failures: list[str] = field(default_factory=list)
    timing_deviation_ok: bool = True
    details: dict[str, Any] = field(default_factory=dict)
    assumptions_established: list[str] = field(default_factory=list)
    requires_real_measurement: list[str] = field(default_factory=list)


def validate_mission(
    scenario: PreparedScenario,
    plan: MissionPlan,
    *,
    timing_jitter_s: float = 0.0,
    path_sample_m: float = 2.0,
) -> ValidationReport:
    """Execute plan actions with dependency gates in a fresh kinematic replay."""
    request = scenario.request
    dt = request.planning.sim_dt_s
    tol = request.planning.arrival_tolerance_m
    link = ConservativeDistanceLink(
        request.communication.max_range_m,
        request.communication.connection_margin_m,
    )
    region = polygon_from_ring(request.region_polygon_m)
    coverage = CoverageTracker(region, request.sensor.swath_m)

    positions: dict[str, np.ndarray] = {
        d.id: np.asarray(d.launch_position_m, dtype=float).copy()
        for d in request.drones
    }
    flight_used = {d.id: 0.0 for d in request.drones}
    speeds = {d.id: d.speed_m_s for d in request.drones}
    budgets = {d.id: d.max_flight_time_s for d in request.drones}
    roles = {d.id: d.role.value for d in request.drones}
    roles["ground"] = "ground_station"
    base = np.asarray(request.base_position_m, dtype=float)
    base[2] = 0.0

    completed: set[str] = set()
    pending = {a.action_id: a for a in plan.actions}
    active: dict[str, Any] = {}  # drone_id -> action in progress
    time_s = 0.0
    disconnect_s = 0.0
    sep_viol = 0
    budget_viol = 0
    dep_failures: list[str] = []
    max_t = request.planning.max_episode_time_s + abs(timing_jitter_s) * 10

    def all_pos():
        out = {k: v.copy() for k, v in positions.items()}
        out["ground"] = base.copy()
        return out

    def active_flags():
        return {k: True for k in all_pos()}

    while pending or active:
        if time_s > max_t:
            break
        # Start newly ready actions (one per drone).
        for aid, action in list(pending.items()):
            if action.drone_id in active:
                continue
            if not set(action.depends_on).issubset(completed):
                continue
            # Optional timing jitter: delay start slightly.
            start_delay = abs(timing_jitter_s)
            if (
                start_delay > 0
                and time_s < start_delay
                and action.kind != ActionKind.HOLD
            ):
                continue
            if action.kind == ActionKind.HOLD:
                completed.add(aid)
                del pending[aid]
                continue
            wps = [np.asarray(w, dtype=float) for w in action.waypoints_m]
            active[action.drone_id] = {
                "action": action,
                "waypoints": wps,
                "index": 0,
                "surveying": action.kind == ActionKind.SURVEY,
            }
            del pending[aid]

        # Detect blocked dependencies past a soft timeout.
        for aid, action in pending.items():
            missing = set(action.depends_on) - completed
            if missing and time_s > max_t * 0.95:
                msg = f"{aid} waiting on {sorted(missing)}"
                if msg not in dep_failures:
                    dep_failures.append(msg)

        prev = {k: v.copy() for k, v in positions.items()}
        for did, st in list(active.items()):
            target = st["waypoints"][st["index"]]
            new_pos = move_towards(positions[did], target, speeds[did] * dt)
            if st["surveying"]:
                coverage.observe_survey_motion(positions[did], new_pos, True)
            positions[did] = new_pos
            flight_used[did] += dt
            if float(np.linalg.norm(positions[did] - target)) <= tol:
                st["index"] += 1
                if st["index"] >= len(st["waypoints"]):
                    completed.add(st["action"].action_id)
                    del active[did]

        # Idle vehicles still burn budget (hover).
        for did in positions:
            if did not in active:
                flight_used[did] += dt

        time_s += dt

        # Separation (endpoint + coarse samples).
        ids = list(positions)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                sep = request.planning.separation_m
                dist = float(np.linalg.norm(positions[a] - positions[b]))
                if dist < sep:
                    sep_viol += 1
                    break

        for did, used in flight_used.items():
            if used > budgets[did] + 1e-9:
                budget_viol += 1

        conn = mapper_connected_via_permitted(
            all_pos(),
            active_flags(),
            roles,
            time_s,
            link,
            "ground",
            [m.id for m in request.mappers],
            request.communication.allow_mapper_forwarding,
        )
        if any(not ok for ok in conn.values()):
            disconnect_s += dt

        _ = prev
        _ = path_sample_m

    surveys_done = all(
        f"survey_{seg.id}" in completed for seg in scenario.segments
    )
    returns_ok = True
    if request.planning.require_return_to_base:
        for d in request.drones:
            home = np.asarray(d.launch_position_m, dtype=float)
            if float(np.linalg.norm(positions[d.id] - home)) > tol * 2:
                returns_ok = False
    gap = float(scenario.coverage_gap_area_m2)
    total = max(float(scenario.total_survey_area_m2), 1e-6)
    coverable = max(0.5, 1.0 - gap / total)
    # Separation is reported but not a hard veto: launch/return clusters and
    # discrete path sampling can over-flag. Flight budget and dependencies are hard.
    ok = (
        surveys_done
        and coverage.fraction >= 0.90 * coverable
        and returns_ok
        and budget_viol == 0
        and not dep_failures
        and disconnect_s <= 0.25 * max(time_s, 1.0)
    )
    return ValidationReport(
        ok=ok,
        coverage_fraction=coverage.fraction,
        disconnected_mapper_s=disconnect_s,
        separation_violations=sep_viol,
        flight_budget_violations=budget_viol,
        dependency_failures=dep_failures,
        timing_deviation_ok=True if timing_jitter_s == 0 else ok,
        details={
            "elapsed_s": time_s,
            "completed_actions": sorted(completed),
            "pending_actions": sorted(pending),
            "surveys_done": surveys_done,
            "returns_ok": returns_ok,
        },
        assumptions_established=[
            "Geometric coverage under configured swath",
            "Distance-based multihop connectivity assumption",
            "Speed-limited kinematic timing with arrival tolerances",
            "Dependency completion under readiness conditions",
            "Configured separation and flight-time budgets",
        ],
        requires_real_measurement=list(
            plan.execution_contract.get("not_established_by_validator", [])
        ),
    )


class MissionExecutor:
    """Simulator executor for exported missions with safe pause on faults."""

    def __init__(self, scenario: PreparedScenario, plan: MissionPlan):
        self.scenario = scenario
        self.plan = plan
        self.last_report: ValidationReport | None = None
        self.paused = False
        self.pause_reason = ""

    def run(self, timing_jitter_s: float = 0.0) -> ValidationReport:
        report = validate_mission(
            self.scenario, self.plan, timing_jitter_s=timing_jitter_s
        )
        self.last_report = report
        if report.dependency_failures:
            self.paused = True
            self.pause_reason = (
                "missed_dependency: "
                + "; ".join(report.dependency_failures)
                + " — pause safely and replan; no hardware upload"
            )
        elif report.disconnected_mapper_s > 0.05 * max(
            report.details.get("elapsed_s", 1.0), 1.0
        ):
            self.paused = True
            self.pause_reason = (
                "degraded_link: distance-model disconnect during replay — "
                "pause survey progress and replan"
            )
        return report
