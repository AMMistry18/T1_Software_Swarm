"""Gymnasium MaskablePPO environment for offline mission planning."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import Env, spaces

from tar_swarm.mission_planning.preprocess import prepare_scenario
from tar_swarm.mission_planning.rewards import RewardWeights, compute_reward
from tar_swarm.mission_planning.schemas import MissionRequest, PreparedScenario
from tar_swarm.mission_planning.simulator import PlanningSimulator
from tar_swarm.mission_planning.spaces import SpaceLayout


class MissionPlanningEnv(Env):
    """Discrete masked actions over concurrent kinematic planning.

    Limitation (documented): mapper segment order is fixed by preprocessing;
    the policy learns relay placement and mapper release timing only.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        request: MissionRequest | None = None,
        scenario: PreparedScenario | None = None,
        reward_weights: RewardWeights | None = None,
        norm_scale_m: float = 200.0,
        scenario_factory=None,
    ):
        super().__init__()
        self.reward_weights = reward_weights or RewardWeights()
        self.norm_scale_m = float(norm_scale_m)
        self.scenario_factory = scenario_factory
        if scenario is not None:
            self._base_scenario = scenario
            self._base_request = scenario.request
        elif request is not None:
            self._base_request = request
            self._base_scenario = prepare_scenario(request)
        else:
            raise ValueError("request or scenario required")
        caps = self._base_request.planning.capacities
        self.layout = SpaceLayout.from_capacities(caps, norm_scale_m)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.layout.obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.layout.n_actions)
        self.sim: PlanningSimulator | None = None
        self._reward_log: list[dict[str, float]] = []
        self._mapper_ids: list[str] = []
        self._relay_ids: list[str] = []
        self._candidate_ids: list[str] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.scenario_factory is not None:
            scenario = self.scenario_factory(self.np_random)
        else:
            scenario = self._base_scenario
        self.sim = PlanningSimulator(scenario)
        self._mapper_ids = [m.id for m in scenario.request.mappers]
        self._relay_ids = [r.id for r in scenario.request.relays]
        self._candidate_ids = [c.id for c in scenario.candidates]
        self._reward_log = []
        obs = self._obs()
        return obs, {"scenario_warnings": list(scenario.warnings)}

    def action_masks(self) -> np.ndarray:
        assert self.sim is not None
        mask = np.zeros(self.layout.n_actions, dtype=bool)
        mask[0] = True  # hold always
        # Relay moves
        for r, rid in enumerate(self._relay_ids):
            drone = self.sim.drones[rid]
            if drone.busy:
                continue
            for c, cid in enumerate(self._candidate_ids):
                target = self.sim.candidates[cid]
                # Redundant: already at this candidate.
                dist = float(np.linalg.norm(drone.position - target))
                if dist <= self.sim.arrival_tol:
                    continue
                mask[self.layout.encode_relay(r, c)] = True
        # Mapper releases
        for m, mid in enumerate(self._mapper_ids):
            drone = self.sim.drones[mid]
            segs = self.sim.segments_by_mapper[mid]
            if drone.busy:
                continue
            if drone.next_segment_index >= len(segs):
                continue
            mask[self.layout.encode_release(m)] = True
        return mask

    def step(self, action: int):
        assert self.sim is not None
        masks = self.action_masks()
        if not masks[int(action)]:
            # Invalid actions should be masked by MaskablePPO;
            # penalize if forced.
            breakdown = compute_reward(
                self.reward_weights,
                new_coverage_frac=0.0,
                mission_complete=False,
                disconnect_s=0.0,
                decision_dt=self.sim.request.planning.decision_dt_s,
                relay_travel_m=0.0,
                repositions=0,
                violations=1,
                max_episode_time_s=self.sim.max_time,
                norm_scale_m=self.norm_scale_m,
                stagnant=False,
                ended_incomplete=False,
                coverage_fraction=0.0,
            )
            self._reward_log.append(breakdown.as_dict())
            return (
                self._obs(),
                breakdown.total,
                False,
                False,
                {"invalid_action": True, "reward": breakdown.as_dict()},
            )

        kind, i0, i1 = self.layout.decode_action(int(action))
        before = self._snapshot_metrics()
        if kind == "hold":
            self.sim.hold()
        elif kind == "relay":
            assert i0 is not None and i1 is not None
            self.sim.send_relay(self._relay_ids[i0], self._candidate_ids[i1])
        elif kind == "release":
            assert i0 is not None
            self.sim.release_mapper(self._mapper_ids[i0])

        self.sim.step_decision()
        after = self._snapshot_metrics()

        # If survey done and idle, allow returns to proceed within same step budget.
        if self.sim._survey_complete() and not self.sim._any_busy():
            if self.sim.require_return and not self.sim._all_returned():
                self.sim.start_returns()
                self.sim.step_decision()
                after = self._snapshot_metrics()

        new_cov = max(0.0, after["coverage"] - before["coverage"])
        disconnect = max(0.0, after["disconnect"] - before["disconnect"])
        travel = max(0.0, after["relay_travel"] - before["relay_travel"])
        repos = max(0, after["repositions"] - before["repositions"])
        viol = max(0, after["violations"] - before["violations"])
        complete = bool(self.sim.metrics.mission_complete and self.sim.terminated)
        terminated = bool(self.sim.terminated)
        truncated = bool(self.sim.truncated) and not terminated
        # Punish do-nothing: survey still open and nobody is moving/working.
        stagnant = (not self.sim._survey_complete()) and (not self.sim._any_busy())
        ended_incomplete = (terminated or truncated) and not complete

        breakdown = compute_reward(
            self.reward_weights,
            new_coverage_frac=new_cov,
            mission_complete=complete,
            disconnect_s=disconnect,
            decision_dt=self.sim.request.planning.decision_dt_s,
            relay_travel_m=travel,
            repositions=repos,
            violations=viol,
            max_episode_time_s=self.sim.max_time,
            norm_scale_m=self.norm_scale_m,
            stagnant=stagnant,
            ended_incomplete=ended_incomplete,
            coverage_fraction=float(self.sim.metrics.coverage_fraction),
        )
        self._reward_log.append(breakdown.as_dict())
        info: dict[str, Any] = {
            "reward": breakdown.as_dict(),
            "metrics": {
                "coverage_fraction": self.sim.metrics.coverage_fraction,
                "disconnected_mapper_s": self.sim.metrics.disconnected_mapper_s,
                "relay_travel_m": self.sim.metrics.relay_travel_m,
                "relay_repositions": self.sim.metrics.relay_repositions,
                "mission_complete": self.sim.metrics.mission_complete,
                "elapsed_s": self.sim.metrics.elapsed_s,
                "term_reason": self.sim.term_reason,
            },
        }
        return self._obs(), breakdown.total, terminated, truncated, info

    def _snapshot_metrics(self) -> dict[str, float]:
        assert self.sim is not None
        m = self.sim.metrics
        return {
            "coverage": m.coverage_fraction,
            "disconnect": m.disconnected_mapper_s,
            "relay_travel": m.relay_travel_m,
            "repositions": m.relay_repositions,
            "violations": m.separation_violations + m.flight_budget_violations,
        }

    def _norm_xy(self, pos: np.ndarray) -> tuple[float, float]:
        assert self.sim is not None
        base = self.sim.base
        return (
            float((pos[0] - base[0]) / self.norm_scale_m),
            float((pos[1] - base[1]) / self.norm_scale_m),
        )

    def _obs(self) -> np.ndarray:
        assert self.sim is not None
        lay = self.layout
        obs = np.zeros(lay.obs_size, dtype=np.float32)
        # base
        obs[0:2] = 0.0
        obs[2] = 1.0
        offset = 3
        # mappers
        for i in range(lay.max_mappers):
            base_i = offset + 7 * i
            if i >= len(self._mapper_ids):
                continue
            d = self.sim.drones[self._mapper_ids[i]]
            segs = self.sim.segments_by_mapper[d.id]
            x, y = self._norm_xy(d.position)
            obs[base_i : base_i + 7] = [
                1.0,
                1.0 if d.busy else 0.0,
                x,
                y,
                d.flight_used_s / max(d.max_flight_time_s, 1e-6),
                d.next_segment_index / max(len(segs), 1),
                max(0, len(segs) - d.next_segment_index) / max(lay.max_segments, 1),
            ]
        offset += 7 * lay.max_mappers
        # relays
        for i in range(lay.max_relays):
            base_i = offset + 7 * i
            if i >= len(self._relay_ids):
                continue
            d = self.sim.drones[self._relay_ids[i]]
            x, y = self._norm_xy(d.position)
            tx, ty = (0.0, 0.0)
            if d.target is not None:
                tx, ty = self._norm_xy(d.target)
            obs[base_i : base_i + 7] = [
                1.0,
                1.0 if d.busy else 0.0,
                x,
                y,
                d.flight_used_s / max(d.max_flight_time_s, 1e-6),
                tx,
                ty,
            ]
        offset += 7 * lay.max_relays
        # candidates
        for i in range(lay.max_candidates):
            base_i = offset + 2 * i
            if i >= len(self._candidate_ids):
                continue
            cpos = self.sim.candidates[self._candidate_ids[i]]
            x, y = self._norm_xy(cpos)
            obs[base_i : base_i + 2] = [x, y]
        offset += 2 * lay.max_candidates
        # segments
        for mi in range(lay.max_mappers):
            for si in range(lay.max_segments):
                base_i = offset + 6 * (mi * lay.max_segments + si)
                if mi >= len(self._mapper_ids):
                    continue
                mid = self._mapper_ids[mi]
                segs = self.sim.segments_by_mapper[mid]
                if si >= len(segs):
                    continue
                seg = segs[si]
                d = self.sim.drones[mid]
                start = np.asarray(seg.waypoints_m[0], dtype=float)
                end = np.asarray(seg.waypoints_m[-1], dtype=float)
                sx, sy = self._norm_xy(start)
                ex, _ey = self._norm_xy(end)
                obs[base_i : base_i + 6] = [
                    1.0,
                    1.0 if si == d.next_segment_index else 0.0,
                    seg.length_m / self.norm_scale_m,
                    sx,
                    sy,
                    ex,
                ]
        offset += 6 * lay.max_mappers * lay.max_segments
        obs[offset : offset + 5] = [
            self.sim.time_s / max(self.sim.max_time, 1e-6),
            self.sim.coverage.fraction,
            self.sim.metrics.disconnected_mapper_s / max(self.sim.max_time, 1e-6),
            len(self._mapper_ids) / max(lay.max_mappers, 1),
            len(self._relay_ids) / max(lay.max_relays, 1),
        ]
        return obs
