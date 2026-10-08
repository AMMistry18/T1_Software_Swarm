"""Timed quintic waypoint interpolation; no online planning or coordination."""

from typing import Any

import numpy as np

from tar_swarm.control.interfaces import Reference


class WaypointMission:
    def __init__(self, agent: dict[str, Any]):
        self.times = np.array([w["time_s"] for w in agent["waypoints"]])
        self.positions = np.array([w["position_m"] for w in agent["waypoints"]])
        self.yaw = agent.get("yaw_rad", 0.0)

    def sample(self, time_s: float) -> Reference:
        if time_s >= self.times[-1] or len(self.times) == 1:
            return Reference(
                self.positions[-1].copy(), np.zeros(3), np.zeros(3), self.yaw
            )
        i = max(0, int(np.searchsorted(self.times, time_s, side="right") - 1))
        duration = self.times[i + 1] - self.times[i]
        u = np.clip((time_s - self.times[i]) / duration, 0, 1)
        delta = self.positions[i + 1] - self.positions[i]
        s = 10 * u**3 - 15 * u**4 + 6 * u**5
        ds = (30 * u**2 - 60 * u**3 + 30 * u**4) / duration
        dds = (60 * u - 180 * u**2 + 120 * u**3) / duration**2
        return Reference(
            self.positions[i] + s * delta, ds * delta, dds * delta, self.yaw
        )
