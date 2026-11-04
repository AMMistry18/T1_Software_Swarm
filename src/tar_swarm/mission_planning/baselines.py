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
    """Pick up to n candidates well-spaced along x from base toward the region."""
    base_x = scenario.request.base_position_m[0]
    ranked = sorted(
        scenario.candidates,
        key=lambda c: (abs(c.position_m[1]), c.position_m[0], c.id),
    )
    chosen: list[str] = []
    last_x = base_x
    for cand in ranked:
        if not chosen:
            chosen.append(cand.id)
            last_x = cand.position_m[0]
            continue
        span = scenario.request.communication.usable_range_m * 0.35
        if cand.position_m[0] - last_x >= span:
            chosen.append(cand.id)
            last_x = cand.position_m[0]
        if len(chosen) >= n:
            break
    # Fall back to first N preprocess-ordered candidates.
    if len(chosen) < n:
        for cand in scenario.candidates:
            if cand.id not in chosen:
                chosen.append(cand.id)
            if len(chosen) >= n:
                break
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
    scenario: PreparedScenario, max_combos: int = 64
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
        if result.success:
            return result
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
        + 10.0 * float(m["coverage_fraction"])
        - 0.1 * float(m["disconnected_mapper_s"])
        - 0.01 * float(m["relay_travel_m"])
        - 0.01 * float(m["mission_duration_s"])
    )
