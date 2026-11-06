"""Observation shapes, action indexing, transitions, rewards, endings.

Observation vector (float32), fixed capacities M, R, C, S:

- ``[0:3]`` base pad
- ``[3:3+7M]`` per mapper: active, busy, x, y, flight_frac,
  next_seg_frac, segs_left_frac
- next ``7R``: per relay active/busy/xy/flight/target_xy
- next ``2C``: candidate xy (padded)
- next ``6*M*S``: per mapper-segment validity/geometry
- final 5: elapsed, coverage, disconnect, n_mappers, n_relays norms

Positions are normalized by ``norm_scale_m`` about the base. Flight budgets use
each vehicle's ``max_flight_time_s``. Elapsed uses ``max_episode_time_s``.

Action indexing (Discrete):

- ``0`` Hold
- ``1 .. R*C`` send relay r to candidate c (``1 + r*C + c``)
- ``1+R*C ..`` release mapper m (``1 + R*C + m``)

Action masks: inactive slots, busy drones, exhausted segments, unknown
candidates, and redundant relay targets are masked. Hold stays available.
Predictive connectivity feasibility is NOT folded into the mask.

Transitions: one command per env step, then ``decision_dt_s`` of concurrent
kinematics. Mapper segment order is deterministic; RL chooses relay placement
and release timing only.

Rewards (configurable; each component logged): coverage gain, completion bonus,
disconnect / relay-travel / reposition / time / violation penalties, plus
``idle`` while survey incomplete and no drone is busy, and ``incomplete``
when the episode ends without mission_complete (scaled by uncovered fraction).

Endings: ``terminated`` after survey+return resolution; ``truncated`` on
time-limit (not claimed as task failure).
"""

from __future__ import annotations

from dataclasses import dataclass

from tar_swarm.mission_planning.schemas import CapacityLimits


@dataclass(frozen=True)
class SpaceLayout:
    max_mappers: int
    max_relays: int
    max_candidates: int
    max_segments: int
    norm_scale_m: float = 200.0

    @classmethod
    def from_capacities(cls, caps: CapacityLimits, norm_scale_m: float = 200.0):
        return cls(
            max_mappers=caps.max_mappers,
            max_relays=caps.max_relays,
            max_candidates=caps.max_candidates,
            max_segments=caps.max_segments_per_mapper,
            norm_scale_m=norm_scale_m,
        )

    @property
    def obs_size(self) -> int:
        m, r, c, s = (
            self.max_mappers,
            self.max_relays,
            self.max_candidates,
            self.max_segments,
        )
        return 3 + 7 * m + 7 * r + 2 * c + 6 * m * s + 5

    @property
    def n_actions(self) -> int:
        return 1 + self.max_relays * self.max_candidates + self.max_mappers

    def decode_action(self, action: int) -> tuple[str, int | None, int | None]:
        if action < 0 or action >= self.n_actions:
            raise ValueError("action out of range")
        if action == 0:
            return ("hold", None, None)
        relay_block = self.max_relays * self.max_candidates
        if action <= relay_block:
            idx = action - 1
            r = idx // self.max_candidates
            c = idx % self.max_candidates
            return ("relay", r, c)
        m = action - 1 - relay_block
        return ("release", m, None)

    def encode_relay(self, relay_index: int, cand_index: int) -> int:
        return 1 + relay_index * self.max_candidates + cand_index

    def encode_release(self, mapper_index: int) -> int:
        return 1 + self.max_relays * self.max_candidates + mapper_index
