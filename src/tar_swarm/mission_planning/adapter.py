"""Hardware / flight-stack adapter boundary.

Existing drone interfaces in this repository:

- ``tar_swarm.control.interfaces.Reference`` / ``State`` / ``FlightController``
  (MuJoCo Step 0 cascaded PD). Suitable for converting exported waypoints into
  ``Reference`` streams offline.
- Network kinematic ``TargetPolicy`` for the standalone network simulator.
- No PX4/MAVLink/ROS upload path is implemented. This module does **not**
  invent hardware protocol details and never auto-uploads missions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol

import numpy as np

from tar_swarm.control.interfaces import Reference
from tar_swarm.mission_planning.schemas import ActionKind, MissionPlan, WaypointAction


class MissionConsumer(Protocol):
    """Example consumer boundary for onboard or SITL executors."""

    def on_action(self, action: WaypointAction) -> None: ...

    def on_pause(self, reason: str) -> None: ...

    def on_replan_required(self, reason: str) -> None: ...


@dataclass
class SafePauseConsumer:
    """Example consumer: records actions and pauses on faults (no upload)."""

    log: list[str]

    def on_action(self, action: WaypointAction) -> None:
        self.log.append(f"accept {action.action_id} for {action.drone_id}")

    def on_pause(self, reason: str) -> None:
        self.log.append(f"PAUSE: {reason}")

    def on_replan_required(self, reason: str) -> None:
        self.log.append(f"REPLAN_REQUIRED: {reason}")


def iter_references_from_action(
    action: WaypointAction,
    speed_m_s: float,
    dt_s: float = 0.05,
) -> Iterator[Reference]:
    """Convert a waypoint action into a simple kinematic Reference stream.

    Uses the existing ``Reference`` contract (position/velocity/acceleration/yaw).
    This is an adapter example for the MuJoCo/control boundary — not a flight
    upload client.
    """
    if action.kind == ActionKind.HOLD:
        if not action.waypoints_m:
            return
        pos = np.asarray(action.waypoints_m[0], dtype=float)
        hold_steps = max(1, int(round(action.hold_s / dt_s))) if action.hold_s else 1
        for _ in range(hold_steps):
            yield Reference(
                position=pos.copy(),
                velocity=np.zeros(3),
                acceleration=np.zeros(3),
                yaw=0.0,
            )
        return

    wps = [np.asarray(w, dtype=float) for w in action.waypoints_m]
    if not wps:
        return
    pos = wps[0].copy()
    for target in wps:
        while True:
            delta = target - pos
            dist = float(np.linalg.norm(delta))
            if dist <= max(speed_m_s * dt_s, action.completion_tolerance_m):
                pos = target.copy()
                yield Reference(
                    position=pos.copy(),
                    velocity=np.zeros(3),
                    acceleration=np.zeros(3),
                    yaw=0.0,
                )
                break
            direction = delta / dist
            vel = direction * speed_m_s
            pos = pos + vel * dt_s
            yield Reference(
                position=pos.copy(),
                velocity=vel.copy(),
                acceleration=np.zeros(3),
                yaw=float(np.arctan2(direction[1], direction[0])),
            )


def dispatch_plan(plan: MissionPlan, consumer: MissionConsumer) -> None:
    """Feed actions to a consumer without uploading to hardware."""
    for action in plan.actions:
        consumer.on_action(action)
