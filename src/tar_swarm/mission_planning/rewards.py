"""Configurable reward components with per-step logging."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RewardWeights:
    coverage: float = 5.0
    completion: float = 20.0
    disconnect: float = 2.0
    relay_travel: float = 0.05
    reposition: float = 0.5
    time: float = 0.1
    violation: float = 5.0


@dataclass
class RewardBreakdown:
    coverage: float = 0.0
    completion: float = 0.0
    disconnect: float = 0.0
    relay_travel: float = 0.0
    reposition: float = 0.0
    time: float = 0.0
    violation: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.coverage
            + self.completion
            + self.disconnect
            + self.relay_travel
            + self.reposition
            + self.time
            + self.violation
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
) -> RewardBreakdown:
    """Assemble per-decision reward. Stationary coverage gain is zero.

    Anti-exploit properties:
    - Coverage uses newly covered area only (no double counting).
    - Hovering advances time and burns flight budget in the simulator.
    - Travel distance is measured from actual motion, not teleportation.
    - Completion requires validated return when configured.
    """
    br = RewardBreakdown()
    br.coverage = weights.coverage * float(new_coverage_frac)
    if mission_complete:
        br.completion = weights.completion
    br.disconnect = -weights.disconnect * (disconnect_s / max(decision_dt, 1e-6))
    br.relay_travel = -weights.relay_travel * (relay_travel_m / max(norm_scale_m, 1e-6))
    br.reposition = -weights.reposition * float(repositions)
    br.time = -weights.time * (decision_dt / max(max_episode_time_s, 1e-6))
    br.violation = -weights.violation * float(violations)
    return br
