"""Configurable reward components with per-step logging."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RewardWeights:
    """Defaults are tuned so pure hold-forever is worse than attempting the mission.

    Hold-only previously scored ~-0.1 (time only) while failed exploration scored
    much worse, so MaskablePPO collapsed to always-hold. ``idle`` + ``incomplete``
    close that exploit; ``disconnect`` / ``violation`` are softer so early tries
    are not catastrophic.
    """

    coverage: float = 5.0
    completion: float = 50.0
    disconnect: float = 0.5
    relay_travel: float = 0.05
    reposition: float = 0.2
    time: float = 0.1
    violation: float = 2.0
    # Per decision while survey incomplete and no drone is busy (do-nothing).
    idle: float = 1.0
    # Applied once when the episode ends without mission_complete, scaled by
    # remaining uncovered fraction: -incomplete * (1 - coverage).
    incomplete: float = 20.0


@dataclass
class RewardBreakdown:
    coverage: float = 0.0
    completion: float = 0.0
    disconnect: float = 0.0
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
) -> RewardBreakdown:
    """Assemble per-decision reward. Stationary coverage gain is zero.

    Anti-exploit properties:
    - Coverage uses newly covered area only (no double counting).
    - Hovering advances time and burns flight budget in the simulator.
    - Travel distance is measured from actual motion, not teleportation.
    - Completion requires validated return when configured.
    - Full-episode hold is punished via ``idle`` while nothing is busy and
      via ``incomplete`` if the episode ends without finishing.
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
    if stagnant:
        br.idle = -weights.idle
    if ended_incomplete:
        remaining = max(0.0, 1.0 - float(coverage_fraction))
        br.incomplete = -weights.incomplete * remaining
    return br
