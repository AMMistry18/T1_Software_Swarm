"""Deterministic fixed-step engine. No plotting or MuJoCo imports."""

from dataclasses import dataclass

import numpy as np

from tar_swarm.network.config import step_count, validate_config
from tar_swarm.network.connectivity import reachable_from
from tar_swarm.network.graph import GraphSnapshot, build_graph
from tar_swarm.network.motion import TargetPolicy, WaypointPolicy, move_towards
from tar_swarm.network.radio import LinkModel, RadiusLink


@dataclass(frozen=True)
class Frame:
    time_s: float
    positions: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]
    active: dict[str, bool]
    graph: GraphSnapshot
    reachable: frozenset[str]
    graph_updated: bool


class Simulator:
    def __init__(
        self,
        config: dict,
        policy: TargetPolicy | None = None,
        link_model: LinkModel | None = None,
    ):
        self.config = validate_config(config)
        self.agents = self.config["agents"]
        self.dt_s = self.config["dt_s"]
        self.total_steps = step_count(self.config["duration_s"], self.dt_s)
        self.graph_steps = step_count(self.config["graph_interval_s"], self.dt_s)
        self.tick = 0
        self.rng = np.random.default_rng(self.config["seed"])
        self.positions = {
            a["id"]: np.asarray(a["position_m"], dtype=float) for a in self.agents
        }
        self.active = {a["id"]: a["active"] for a in self.agents}
        self.ground_id = next(
            a["id"] for a in self.agents if a["role"] == "ground_station"
        )
        self.policy = policy if policy is not None else WaypointPolicy(self.agents)
        self.link_model = (
            link_model
            if link_model is not None
            else RadiusLink(self.config["communication_radius_m"])
        )
        self.graph = build_graph(self.positions, self.active, 0.0, self.link_model)
        self.reachable = reachable_from(self.graph, self.ground_id)
        self.history = [self._frame(True)]

    @property
    def time_s(self):
        return self.tick * self.dt_s

    @property
    def done(self):
        return self.tick >= self.total_steps

    def _frame(self, updated):
        positions = {key: value.copy() for key, value in self.positions.items()}
        targets = {}
        for agent in self.agents:
            name = agent["id"]
            target = (
                positions[name].copy()
                if agent["role"] == "ground_station"
                else np.asarray(
                    self.policy.target(name, self.time_s, positions), dtype=float
                ).copy()
            )
            if target.shape != (3,) or not np.isfinite(target).all():
                raise ValueError("Policy targets must be finite 3-vectors")
            if target[2] != self.positions[name][2]:
                raise ValueError("Policy targets must preserve each agent's altitude")
            targets[name] = target
        return Frame(
            self.time_s,
            positions,
            targets,
            self.active.copy(),
            self.graph,
            self.reachable,
            updated,
        )

    def step(self) -> Frame:
        if self.done:
            raise StopIteration("Simulation duration reached")
        previous = self.history[-1]
        for agent in self.agents:
            name = agent["id"]
            if agent["role"] != "ground_station" and self.active[name]:
                self.positions[name] = move_towards(
                    previous.positions[name],
                    previous.targets[name],
                    agent["speed_m_s"] * self.dt_s,
                )
        self.tick += 1
        updated = self.tick % self.graph_steps == 0
        if updated:
            self.graph = build_graph(
                self.positions, self.active, self.time_s, self.link_model
            )
            self.reachable = reachable_from(self.graph, self.ground_id)
        frame = self._frame(updated)
        self.history.append(frame)
        return frame

    def run(self):
        while not self.done:
            self.step()
        return self.history
