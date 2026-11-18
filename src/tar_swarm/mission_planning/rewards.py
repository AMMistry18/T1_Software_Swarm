"""Configurable reward components with per-step logging."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RewardWeights:
    """Defaults prioritize full coverage + link uptime over casual travel.

    Priority order for planning quality:
    1. Cover the region (dense coverage + incomplete terminal).
    2. Keep mappers connected (disconnect cost dominates travel).
    3. Prefer short relay moves (travel) over wandering then returning.
    4. Still punish pure hold via ``idle`` so PPO cannot collapse to no-ops.
    """

    coverage: float = 8.0
    completion: float = 50.0
    # Per decision-dt of mapper disconnect — must beat cheap travel.
    disconnect: float = 4.0
    # Bonus per decision when every active surveying/transit mapper is linked.
    connected: float = 0.5
    relay_travel: float = 0.4
    reposition: float = 0.15
    time: float = 0.05
    violation: float = 2.0
    # Per decision while survey incomplete and no drone is busy (do-nothing).
    idle: float = 1.0
    # Terminal: -incomplete * (1 - coverage) when episode ends incomplete.
    incomplete: float = 40.0


@dataclass
class RewardBreakdown:
    coverage: float = 0.0
    completion: float = 0.0
    disconnect: float = 0.0
    connected: float = 0.0
    relay_travel: float = 0.0
    reposition: float = 0.0
    time: float = 0.0
    violation: float = 0.0
    idle: float = 0.0
    incomplete: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.coverage
            + self.completion
            + self.disconnect
            + self.connected
            + self.relay_travel
            + self.reposition
            + self.time
            + self.violation
            + self.idle
            + self.incomplete
        )

    def as_dict(self) -> dict[str, float]:
        data = asdict(self)
        data["total"] = self.total
        return data


def compute_reward(
    weights: RewardWeights,
    *,
    new_coverage_frac: float,
    mission_complete: bool,
    disconnect_s: float,
    decision_dt: float,
    relay_travel_m: float,
    repositions: int,
    violations: int,
    max_episode_time_s: float,
    norm_scale_m: float,
    stagnant: bool = False,
    ended_incomplete: bool = False,
    coverage_fraction: float = 0.0,
    all_mappers_connected: bool = False,
    mappers_need_link: bool = False,
) -> RewardBreakdown:
    """Assemble per-decision reward. Stationary coverage gain is zero.

    Anti-exploit properties:
    - Coverage uses newly covered area only (no double counting).
    - Hovering advances time and burns flight budget in the simulator.
    - Travel distance is measured from actual motion, not teleportation.
    - Completion requires validated return when configured.
    - Full-episode hold is punished via ``idle`` / ``incomplete``.
    - Disconnect is priced above typical relay travel so ignoring RF fails.
    """
    br = RewardBreakdown()
    br.coverage = weights.coverage * float(new_coverage_frac)
    if mission_complete:
        br.completion = weights.completion
    br.disconnect = -weights.disconnect * (disconnect_s / max(decision_dt, 1e-6))
    if mappers_need_link and all_mappers_connected:
        br.connected = weights.connected
    br.relay_travel = -weights.relay_travel * (relay_travel_m / max(norm_scale_m, 1e-6))
    br.reposition = -weights.reposition * float(repositions)
    br.time = -weights.time * (decision_dt / max(max_episode_time_s, 1e-6))
    br.violation = -weights.violation * float(violations)
    if stagnant:
        br.idle = -weights.idle
    if ended_incomplete:
        remaining = max(0.0, 1.0 - float(coverage_fraction))
        br.incomplete = -weights.incomplete * remaining
    return br
