"""Speed-limited kinematics and a replaceable movement-target policy."""

from typing import Mapping, Protocol

import numpy as np


class TargetPolicy(Protocol):
    def target(
        self, agent_id: str, time_s: float, positions: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        """Return a 3D target from a snapshot at the beginning of this step."""
        ...


def move_towards(position: np.ndarray, target: np.ndarray, budget_m: float):
    delta = target - position
    distance = float(np.linalg.norm(delta))
    if distance <= budget_m:
        return target.copy()
    return position + delta * (budget_m / distance)


class WaypointPolicy:
    def __init__(self, agents: list[dict]):
        self.paths = {
            agent["id"]: [
                np.asarray(point, dtype=float)
                for point in agent["waypoints_m"] or [agent["position_m"]]
            ]
            for agent in agents
        }
        self.indices = dict.fromkeys(self.paths, 0)

    def target(self, agent_id, time_s, positions):
        path = self.paths[agent_id]
        index = self.indices[agent_id]
        while index < len(path) - 1 and np.array_equal(
            positions[agent_id], path[index]
        ):
            index += 1
        self.indices[agent_id] = index
        return path[index].copy()
