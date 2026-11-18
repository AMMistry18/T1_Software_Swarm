"""Deterministic planning baselines (not learned)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from tar_swarm.mission_planning.schemas import PreparedScenario
from tar_swarm.mission_planning.simulator import PlanningSimulator


@dataclass
class BaselineResult:
    name: str
    success: bool
    metrics: dict[str, Any]
    command_trace: list[dict[str, Any]]
    sim: PlanningSimulator


def _spaced_candidates(scenario: PreparedScenario, n: int) -> list[str]:
    """Pick up to n candidates as a chain from base into the survey region.

    Prefer increasing easting with modest |y| so relays cover mapper lawnmower
    depth instead of clustering near the pad (which caused long disconnects).
    """
    if n <= 0 or not scenario.candidates:
        return []
    base_x = float(scenario.request.base_position_m[0])
    usable = float(scenario.request.communication.usable_range_m)
    # Aim hop spacing ~60% of usable range so the chain stays linked.
    hop = max(usable * 0.55, 1.0)
    xs = [c.position_m[0] for c in scenario.candidates]
    target_depth = max(xs) if xs else base_x
    ranked = sorted(
        scenario.candidates,
        key=lambda c: (c.position_m[0], abs(c.position_m[1]), c.id),
    )
    chosen: list[str] = []
    last_x = base_x
    # Place relays at roughly hop, 2*hop, ... toward the far side.
    for i in range(n):
        goal_x = min(base_x + hop * (i + 1), target_depth)
        best = None
        best_key = None
        # Stagger later relays off the x-axis so multi-mapper corridors stay linked.
        y_bias = 0.0
        if n >= 2 and i == n - 1:
            y_bias = 1.0
        elif n >= 3 and i == n - 2:
            y_bias = -1.0
        for cand in ranked:
            if cand.id in chosen:
                continue
            # Prefer candidates near the goal x, modest |y|, and ahead of last hop.
            dx = abs(cand.position_m[0] - goal_x)
            ahead = cand.position_m[0] >= last_x - 1e-6
            y_term = abs(cand.position_m[1] - y_bias * 15.0)
            key = (0 if ahead else 1, dx, y_term, cand.id)
            if best_key is None or key < best_key:
                best_key = key
                best = cand
        if best is None:
            break
        chosen.append(best.id)
        last_x = best.position_m[0]
    return chosen[:n]


def static_relay_baseline(scenario: PreparedScenario) -> BaselineResult:
    """Place relays on spaced chain-like candidates, then release all segments."""
    sim = PlanningSimulator(scenario)
    trace: list[dict[str, Any]] = []
    relays = [d.id for d in scenario.request.relays]
    cands = _spaced_candidates(scenario, len(relays))
    for i, rid in enumerate(relays):
        if i >= len(cands):
            break
        sim.send_relay(rid, cands[i])
        trace.append({"cmd": "relay", "relay": rid, "candidate": cands[i]})
        sim.step_decision()
    # Wait for relays to arrive.
    for _ in range(50):
        if not any(sim.drones[r].busy for r in relays):
            break
        sim.hold()
        sim.step_decision()
        trace.append({"cmd": "hold"})
    # Release mapper segments one decision at a time.
    mappers = [m.id for m in scenario.request.mappers]
    safety = 0
    while safety < 500 and not sim.terminated and not sim.truncated:
        safety += 1
        progress = False
        for mid in mappers:
            d = sim.drones[mid]
            segs = sim.segments_by_mapper[mid]
            if not d.busy and d.next_segment_index < len(segs):
                sim.release_mapper(mid)
                trace.append({"cmd": "release", "mapper": mid})
                progress = True
                break
        if not progress:
            sim.hold()
            trace.append({"cmd": "hold"})
        sim.step_decision()
        if sim._survey_complete() and not sim._any_busy():
            if sim.require_return and not sim._all_returned():
                sim.start_returns()
                trace.append({"cmd": "return"})
            sim.step_decision()
    return BaselineResult(
        name="static_relay",
        success=bool(sim.metrics.mission_complete),
        metrics=_metrics(sim),
        command_trace=trace,
        sim=sim,
    )


def search_relay_baseline(
    scenario: PreparedScenario, max_combos: int = 96
) -> BaselineResult:
    """Search small combinations of candidate assignments while releasing segments.

    Attempts to preserve connectivity by preferring assignments where each
    mapper start is connected through the static relay set at t=0 of survey.
    This is a heuristic search, not a proof of physical feasibility.
    """
    relays = [d.id for d in scenario.request.relays]
    cands = [c.id for c in scenario.candidates]
    if not relays:
        return static_relay_baseline(scenario)

    # Always try the spaced static assignment first.
    seed_combo = tuple(_spaced_candidates(scenario, len(relays)))
    # Limit search: first K candidates for each relay.
    k = max(1, int(np.ceil(max_combos ** (1 / max(len(relays), 1)))))
    short = cands[: min(len(cands), k)]
    combos = [seed_combo] + [
        combo
        for combo in product(short, repeat=len(relays))
        if combo != seed_combo
    ]
    combos = combos[:max_combos]

    best: BaselineResult | None = None
    for combo in combos:
        if len(combo) != len(relays):
            continue
        sim = PlanningSimulator(scenario)
        trace: list[dict[str, Any]] = []
        for rid, cid in zip(relays, combo, strict=True):
            sim.send_relay(rid, cid)
            trace.append({"cmd": "relay", "relay": rid, "candidate": cid})
            sim.step_decision()
        for _ in range(40):
            if not any(sim.drones[r].busy for r in relays):
                break
            sim.hold()
            sim.step_decision()
            trace.append({"cmd": "hold"})
        mappers = [m.id for m in scenario.request.mappers]
        safety = 0
        while safety < 400 and not sim.terminated and not sim.truncated:
            safety += 1
            progress = False
            for mid in mappers:
                d = sim.drones[mid]
                segs = sim.segments_by_mapper[mid]
                if not d.busy and d.next_segment_index < len(segs):
                    sim.release_mapper(mid)
                    trace.append({"cmd": "release", "mapper": mid})
                    progress = True
                    break
            if not progress:
                sim.hold()
                trace.append({"cmd": "hold"})
            sim.step_decision()
            if sim._survey_complete() and not sim._any_busy():
                if sim.require_return and not sim._all_returned():
                    sim.start_returns()
                    trace.append({"cmd": "return"})
                sim.step_decision()
        result = BaselineResult(
            name="search_relay",
            success=bool(sim.metrics.mission_complete),
            metrics=_metrics(sim),
            command_trace=trace,
            sim=sim,
        )
        if best is None or _score(result) > _score(best):
            best = result
    assert best is not None
    return best


def _metrics(sim: PlanningSimulator) -> dict[str, Any]:
    m = sim.metrics
    waiting = sum(d.waiting_s for d in sim.drones.values() if d.role == "mapper")
    return {
        "coverage_fraction": m.coverage_fraction,
        "disconnected_mapper_s": m.disconnected_mapper_s,
        "relay_travel_m": m.relay_travel_m,
        "relay_repositions": m.relay_repositions,
        "mission_duration_s": m.elapsed_s,
        "mapper_waiting_s": waiting,
        "flight_budget_violations": m.flight_budget_violations,
        "separation_violations": m.separation_violations,
        "mission_complete": m.mission_complete,
        "returned_ok": m.returned_ok,
        "term_reason": sim.term_reason,
    }


def _score(result: BaselineResult) -> float:
    m = result.metrics
    return (
        (100.0 if result.success else 0.0)
        + 20.0 * float(m["coverage_fraction"])
        - 2.0 * float(m["disconnected_mapper_s"])
        - 0.01 * float(m["relay_travel_m"])
        - 0.01 * float(m["mission_duration_s"])
    )
