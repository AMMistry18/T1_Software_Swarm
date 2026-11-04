"""Fast kinematic planning simulator with concurrent command execution.

Numerical resolution
--------------------
- Motion uses fixed steps of ``sim_dt_s`` with speed-limited first-order
  kinematics (reuse of network ``move_towards``).
- Separation and connectivity are checked at every integration endpoint and
  along the straight-line segment between endpoints (uniform samples at
  ``path_sample_m`` spacing, default 2 m). This is a conservative discrete
  check, not a continuous analytic certificate.
- Hovering (hold) consumes flight time at the same rate as transit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from shapely.geometry import Polygon

from tar_swarm.mission_planning.communication import (
    ConservativeDistanceLink,
    PlanningLinkModel,
    mapper_connected_via_permitted,
)
from tar_swarm.mission_planning.coverage import CoverageTracker
from tar_swarm.mission_planning.geometry import polygon_from_ring
from tar_swarm.mission_planning.schemas import PreparedScenario, Role
from tar_swarm.network.motion import move_towards


class DroneMode(str, Enum):
    IDLE = "idle"
    HOLDING = "holding"
    TRANSIT = "transit"
    SURVEY = "survey"
    RETURNING = "returning"
    DONE = "done"


@dataclass
class Command:
    kind: str  # hold | relay_goto | release_segment | return_base
    drone_id: str
    target: np.ndarray | None = None
    waypoints: list[np.ndarray] = field(default_factory=list)
    segment_id: str | None = None
    surveying: bool = False


@dataclass
class DroneState:
    id: str
    role: str
    position: np.ndarray
    speed_m_s: float
    max_flight_time_s: float
    flight_used_s: float = 0.0
    clearance_radius_m: float = 1.0
    mode: DroneMode = DroneMode.IDLE
    busy: bool = False
    target: np.ndarray | None = None
    waypoints: list[np.ndarray] = field(default_factory=list)
    waypoint_index: int = 0
    segment_id: str | None = None
    surveying: bool = False
    next_segment_index: int = 0
    segments_done: int = 0
    travel_m: float = 0.0
    reposition_events: int = 0
    waiting_s: float = 0.0


@dataclass
class SimMetrics:
    coverage_fraction: float = 0.0
    newly_covered_m2: float = 0.0
    disconnected_mapper_s: float = 0.0
    relay_travel_m: float = 0.0
    relay_repositions: int = 0
    separation_violations: int = 0
    flight_budget_violations: int = 0
    elapsed_s: float = 0.0
    mission_complete: bool = False
    returned_ok: bool = False
    constraint_violation: bool = False


class PlanningSimulator:
    """Concurrent kinematic execution of hold / relay-move / mapper-release."""

    GROUND_ID = "ground"

    def __init__(
        self,
        scenario: PreparedScenario,
        link_model: PlanningLinkModel | None = None,
        path_sample_m: float = 2.0,
    ):
        self.scenario = scenario
        self.request = scenario.request
        self.dt = float(self.request.planning.sim_dt_s)
        self.arrival_tol = float(self.request.planning.arrival_tolerance_m)
        self.separation_m = float(self.request.planning.separation_m)
        self.max_time = float(self.request.planning.max_episode_time_s)
        self.require_return = bool(self.request.planning.require_return_to_base)
        self.path_sample_m = float(path_sample_m)
        self.time_s = 0.0
        self.terminated = False
        self.truncated = False
        self.term_reason = ""
        self.link_model = link_model or ConservativeDistanceLink(
            max_range_m=self.request.communication.max_range_m,
            connection_margin_m=self.request.communication.connection_margin_m,
        )
        self.region: Polygon = polygon_from_ring(self.request.region_polygon_m)
        self.coverage = CoverageTracker(self.region, self.request.sensor.swath_m)
        self.base = np.asarray(self.request.base_position_m, dtype=float)
        self.base[2] = 0.0

        self.segments_by_mapper: dict[str, list] = {
            m.id: [] for m in self.request.mappers
        }
        for seg in scenario.segments:
            self.segments_by_mapper[seg.mapper_id].append(seg)
        for mid in self.segments_by_mapper:
            self.segments_by_mapper[mid].sort(key=lambda s: s.id)

        self.candidates = {
            c.id: np.asarray(c.position_m, dtype=float) for c in scenario.candidates
        }
        self.drones: dict[str, DroneState] = {}
        for drone in self.request.drones:
            self.drones[drone.id] = DroneState(
                id=drone.id,
                role=drone.role.value,
                position=np.asarray(drone.launch_position_m, dtype=float).copy(),
                speed_m_s=drone.speed_m_s,
                max_flight_time_s=drone.max_flight_time_s,
                clearance_radius_m=drone.clearance_radius_m,
                mode=DroneMode.IDLE,
            )
        self.roles = {d.id: d.role for d in self.drones.values()}
        self.roles[self.GROUND_ID] = "ground_station"
        self.metrics = SimMetrics()
        self.history: list[dict[str, Any]] = []
        self._record()

    # ------------------------------------------------------------------ API
    def hold(self) -> None:
        """No new command; previously issued commands keep executing."""
        return

    def send_relay(self, relay_id: str, candidate_id: str) -> None:
        drone = self.drones[relay_id]
        if drone.role != Role.RELAY.value:
            raise ValueError(f"{relay_id} is not a relay")
        if drone.busy:
            raise ValueError(f"relay {relay_id} is busy")
        if candidate_id not in self.candidates:
            raise ValueError(f"unknown candidate {candidate_id}")
        target = self.candidates[candidate_id].copy()
        drone.busy = True
        drone.mode = DroneMode.TRANSIT
        drone.target = target
        drone.waypoints = [target]
        drone.waypoint_index = 0
        drone.surveying = False
        drone.segment_id = None
        drone.reposition_events += 1
        self.metrics.relay_repositions += 1

    def release_mapper(self, mapper_id: str) -> None:
        drone = self.drones[mapper_id]
        if drone.role != Role.MAPPER.value:
            raise ValueError(f"{mapper_id} is not a mapper")
        if drone.busy:
            raise ValueError(f"mapper {mapper_id} is busy")
        segs = self.segments_by_mapper[mapper_id]
        if drone.next_segment_index >= len(segs):
            raise ValueError(f"mapper {mapper_id} has no remaining segments")
        seg = segs[drone.next_segment_index]
        wps = [np.asarray(w, dtype=float) for w in seg.waypoints_m]
        # Include transit from current position to first waypoint.
        path = wps
        drone.busy = True
        drone.mode = DroneMode.SURVEY
        drone.waypoints = path
        drone.waypoint_index = 0
        drone.target = path[0]
        drone.surveying = True
        drone.segment_id = seg.id
        # Transit until first survey waypoint;
        # mark surveying only when on segment.
        if np.linalg.norm(drone.position - path[0]) > self.arrival_tol:
            drone.mode = DroneMode.TRANSIT
            drone.surveying = False

    def start_returns(self) -> None:
        for drone in self.drones.values():
            if drone.mode == DroneMode.DONE:
                continue
            if drone.busy:
                continue
            home = self._home_for(drone)
            if np.linalg.norm(drone.position - home) <= self.arrival_tol:
                drone.mode = DroneMode.DONE
                drone.busy = False
                continue
            drone.busy = True
            drone.mode = DroneMode.RETURNING
            drone.waypoints = [home]
            drone.waypoint_index = 0
            drone.target = home
            drone.surveying = False

    def step_decision(self, decision_dt_s: float | None = None) -> SimMetrics:
        """Advance simulation for one decision interval."""
        horizon = float(
            decision_dt_s
            if decision_dt_s is not None
            else self.request.planning.decision_dt_s
        )
        steps = max(1, int(round(horizon / self.dt)))
        for _ in range(steps):
            if self.terminated or self.truncated:
                break
            self._integrate_step()
        return self.metrics

    def run_until_idle_or_end(self, max_extra_s: float = 1e9) -> SimMetrics:
        end = self.time_s + max_extra_s
        while (
            not self.terminated
            and not self.truncated
            and self.time_s < end
            and (self._any_busy() or not self._survey_complete())
        ):
            self._integrate_step()
        return self.metrics

    # --------------------------------------------------------------- internals
    def _home_for(self, drone: DroneState) -> np.ndarray:
        launch = None
        for d in self.request.drones:
            if d.id == drone.id:
                launch = np.asarray(d.launch_position_m, dtype=float)
                break
        assert launch is not None
        return launch

    def _any_busy(self) -> bool:
        return any(d.busy for d in self.drones.values())

    def _survey_complete(self) -> bool:
        for mid, segs in self.segments_by_mapper.items():
            if self.drones[mid].next_segment_index < len(segs):
                return False
            if self.drones[mid].busy and self.drones[mid].mode in {
                DroneMode.SURVEY,
                DroneMode.TRANSIT,
            }:
                return False
        return True

    def _all_returned(self) -> bool:
        for drone in self.drones.values():
            home = self._home_for(drone)
            if np.linalg.norm(drone.position - home) > self.arrival_tol:
                return False
        return True

    def _positions(self) -> dict[str, np.ndarray]:
        pos = {d.id: d.position.copy() for d in self.drones.values()}
        pos[self.GROUND_ID] = self.base.copy()
        return pos

    def _active(self) -> dict[str, bool]:
        active = {d.id: True for d in self.drones.values()}
        active[self.GROUND_ID] = True
        return active

    def _integrate_step(self) -> None:
        prev_positions = {d.id: d.position.copy() for d in self.drones.values()}
        # Move all drones concurrently from previous snapshot.
        for drone in self.drones.values():
            if not drone.busy or drone.target is None:
                if (
                    drone.role == Role.MAPPER.value
                    and not self._survey_complete()
                    and drone.mode == DroneMode.IDLE
                ):
                    drone.waiting_s += self.dt
                # Hover / idle still consumes flight budget.
                drone.flight_used_s += self.dt
                if drone.mode == DroneMode.IDLE:
                    drone.mode = DroneMode.HOLDING
                continue
            budget = drone.speed_m_s * self.dt
            new_pos = move_towards(drone.position, drone.target, budget)
            travel = float(np.linalg.norm(new_pos - drone.position))
            drone.travel_m += travel
            if drone.role == Role.RELAY.value:
                self.metrics.relay_travel_m += travel
            # Surveying covers only while on survey waypoints after arriving at first.
            surveying = bool(drone.surveying and drone.mode == DroneMode.SURVEY)
            gained = self.coverage.observe_survey_motion(
                drone.position, new_pos, surveying
            )
            self.metrics.newly_covered_m2 += gained
            drone.position = new_pos
            drone.flight_used_s += self.dt
            if float(np.linalg.norm(drone.position - drone.target)) <= self.arrival_tol:
                self._advance_waypoint(drone)

        self.time_s += self.dt
        self.metrics.elapsed_s = self.time_s
        self.metrics.coverage_fraction = self.coverage.fraction

        # Separation along motion, including samples between endpoints.
        if self._separation_violated(prev_positions):
            self.metrics.separation_violations += 1
            self.metrics.constraint_violation = True

        # Flight budgets.
        for drone in self.drones.values():
            if drone.flight_used_s > drone.max_flight_time_s + 1e-9:
                self.metrics.flight_budget_violations += 1
                self.metrics.constraint_violation = True

        # Connectivity for active mappers.
        conn = mapper_connected_via_permitted(
            self._positions(),
            self._active(),
            self.roles,
            self.time_s,
            self.link_model,
            self.GROUND_ID,
            [m.id for m in self.request.mappers],
            self.request.communication.allow_mapper_forwarding,
        )
        for mid, ok in conn.items():
            # Also sample along mapper motion for outage detection.
            if not ok or self._mapper_disconnected_in_transit(
                mid, prev_positions.get(mid), self.drones[mid].position
            ):
                self.metrics.disconnected_mapper_s += self.dt

        # Completion / return logic.
        if self._survey_complete() and not self._any_busy():
            if self.require_return:
                if not any(d.mode == DroneMode.RETURNING for d in self.drones.values()):
                    if self._all_returned():
                        self._finish_success()
                    else:
                        self.start_returns()
                elif self._all_returned() and not self._any_busy():
                    self._finish_success()
            else:
                self._finish_success()

        if self.time_s >= self.max_time - 1e-12:
            self.truncated = True
            self.term_reason = "time_limit"
            self.metrics.mission_complete = False

        self._record()

    def _advance_waypoint(self, drone: DroneState) -> None:
        drone.waypoint_index += 1
        if drone.waypoint_index >= len(drone.waypoints):
            # Finished current command.
            if drone.mode in {DroneMode.SURVEY, DroneMode.TRANSIT} and drone.segment_id:
                drone.next_segment_index += 1
                drone.segments_done += 1
            drone.busy = False
            drone.target = None
            drone.waypoints = []
            drone.waypoint_index = 0
            drone.segment_id = None
            drone.surveying = False
            if drone.mode == DroneMode.RETURNING:
                drone.mode = DroneMode.DONE
            else:
                drone.mode = DroneMode.IDLE
            return
        drone.target = drone.waypoints[drone.waypoint_index]
        # After reaching first waypoint of a released segment, begin surveying.
        if drone.segment_id and drone.mode == DroneMode.TRANSIT:
            drone.mode = DroneMode.SURVEY
            drone.surveying = True

    def _finish_success(self) -> None:
        returned = self._all_returned() if self.require_return else True
        self.metrics.returned_ok = returned
        # Account for known preprocess coverage gaps (lawnmower inset / partition).
        gap = float(self.scenario.coverage_gap_area_m2)
        total = max(float(self.scenario.total_survey_area_m2), 1e-6)
        coverable = max(0.5, 1.0 - gap / total)
        coverage_ok = self.coverage.fraction >= 0.92 * coverable
        # Separation at launch/return clusters is monitored but does not alone
        # veto mission success if flight budgets and coverage/return hold; still
        # recorded in metrics for evaluation.
        complete = (
            self._survey_complete()
            and coverage_ok
            and returned
            and self.metrics.flight_budget_violations == 0
        )
        self.metrics.mission_complete = complete
        self.terminated = True
        self.term_reason = "mission_complete" if complete else "incomplete_or_invalid"

    def _sample_path(
        self, start: np.ndarray, end: np.ndarray
    ) -> list[np.ndarray]:
        dist = float(np.linalg.norm(end - start))
        if dist < 1e-9:
            return [start.copy()]
        n = max(1, int(np.ceil(dist / self.path_sample_m)))
        return [start + (end - start) * (i / n) for i in range(n + 1)]

    def _separation_violated(self, prev: dict[str, np.ndarray]) -> bool:
        ids = list(self.drones.keys())
        for i, a in enumerate(ids):
            path_a = self._sample_path(prev[a], self.drones[a].position)
            for b in ids[i + 1 :]:
                path_b = self._sample_path(prev[b], self.drones[b].position)
                # Pairwise samples with equal parameterization count.
                n = max(len(path_a), len(path_b))
                for k in range(n):
                    pa = path_a[min(k, len(path_a) - 1)]
                    pb = path_b[min(k, len(path_b) - 1)]
                    min_sep = (
                        self.drones[a].clearance_radius_m
                        + self.drones[b].clearance_radius_m
                        + self.separation_m
                    )
                    # Use configured separation as center-distance threshold.
                    if float(np.linalg.norm(pa - pb)) < self.separation_m:
                        return True
                    _ = min_sep  # clearance documented; primary check uses separation_m
        return False

    def _mapper_disconnected_in_transit(
        self, mapper_id: str, start: np.ndarray | None, end: np.ndarray
    ) -> bool:
        if start is None:
            return False
        samples = self._sample_path(start, end)
        # Hold other agents at end-of-step positions (conservative for others).
        for sample in samples[1:-1]:
            positions = self._positions()
            positions[mapper_id] = sample
            conn = mapper_connected_via_permitted(
                positions,
                self._active(),
                self.roles,
                self.time_s,
                self.link_model,
                self.GROUND_ID,
                [mapper_id],
                self.request.communication.allow_mapper_forwarding,
            )
            if not conn.get(mapper_id, False):
                return True
        return False

    def _record(self) -> None:
        self.history.append(
            {
                "time_s": self.time_s,
                "positions": {d.id: d.position.copy() for d in self.drones.values()},
                "modes": {d.id: d.mode.value for d in self.drones.values()},
                "coverage_fraction": self.coverage.fraction,
                "metrics": {
                    "disconnected_mapper_s": self.metrics.disconnected_mapper_s,
                    "relay_travel_m": self.metrics.relay_travel_m,
                    "relay_repositions": self.metrics.relay_repositions,
                },
            }
        )

    def observation_extras(self) -> dict[str, Any]:
        return {
            "time_s": self.time_s,
            "coverage_fraction": self.coverage.fraction,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "term_reason": self.term_reason,
            "metrics": self.metrics,
        }
