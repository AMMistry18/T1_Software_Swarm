"""Deterministic fixed-step engine. No plotting or MuJoCo imports.

Besides stepping the clock, the engine accepts hand edits from the interactive
window (or from code): add or remove a network drone, move any drone except the
base station, give a drone a goal to fly to, or change a radio range. An edit
re-checks the links right away and updates the latest snapshot, so the window
shows the change at once.
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from tar_swarm.network.config import step_count, validate_config
from tar_swarm.network.connectivity import reachable_from
from tar_swarm.network.graph import GraphSnapshot, build_graph
from tar_swarm.network.motion import TargetPolicy, WaypointPolicy, move_towards
from tar_swarm.network.radio import LinkModel, RangeLink


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
        # A separate copy of the drone list: hand edits change this copy, while
        # self.config keeps describing how the run started.
        self.agents = deepcopy(self.config["agents"])
        self.dt_s = self.config["dt_s"]
        self.total_steps = step_count(self.config["duration_s"], self.dt_s)
        self.graph_steps = step_count(self.config["graph_interval_s"], self.dt_s)
        self.tick = 0
        self.rng = np.random.default_rng(self.config["seed"])
        self.positions = {
            a["id"]: np.asarray(a["position_m"], dtype=float) for a in self.agents
        }
        self.active = {a["id"]: a["active"] for a in self.agents}
        # Every drone that has been in this run, including ones removed by hand, so
        # older snapshots can still be labeled with each drone's role.
        self.roles = {a["id"]: a["role"] for a in self.agents}
        # Targets set by hand: a dragged or added drone holds its spot. When a drone
        # has a manual target, it wins over the movement policy.
        self.manual_targets = {}
        self.edited = False  # becomes True at the first hand edit
        # True: the clock never stops at duration_s (the live window runs until it's
        # closed). Headless runs leave this False and stop at duration_s.
        self.endless = False
        # Automatic network-drone placement; see set_relay_planner. While it's on,
        # plan_stretch holds the latest plan's hop stretch (0.83 = 83%).
        self.relay_planner = None
        self.plan_stretch = None
        self.ground_id = next(
            a["id"] for a in self.agents if a["role"] == "ground_station"
        )
        self.policy = policy if policy is not None else WaypointPolicy(self.agents)
        # Obstacle boxes (x_min, y_min, x_max, y_max) that block signals. Boxes are
        # tuples, so each one can be found and removed by its four numbers.
        self.obstacles = [tuple(box) for box in self.config["obstacles_m"]]
        # dict(...) copies the ranges, so moving a slider never changes self.config.
        # The link model shares the obstacle list, so new boxes block links at once.
        self.link_model = (
            link_model
            if link_model is not None
            else RangeLink(dict(self.config["radio_range_m"]), self.obstacles)
        )
        self.graph = build_graph(
            self.positions, self.active, 0.0, self.link_model, self.roles
        )
        self.reachable = reachable_from(self.graph, self.ground_id)
        self.history = [self._frame(True)]

    @property
    def time_s(self):
        return self.tick * self.dt_s

    @property
    def done(self):
        return not self.endless and self.tick >= self.total_steps

    def _frame(self, updated):
        positions = {key: value.copy() for key, value in self.positions.items()}
        # Spots from the automatic placement, if it's on: relay ID -> (x, y).
        planned = {}
        if self.relay_planner is not None:
            planned, self.plan_stretch = self.relay_planner(self, positions)
        targets = {}
        for agent in self.agents:
            name = agent["id"]
            if agent["role"] == "ground_station":
                target = positions[name].copy()
            elif name in planned:
                x, y = planned[name]
                target = np.array([x, y, positions[name][2]])  # keeps its altitude
            elif name in self.manual_targets:
                target = self.manual_targets[name].copy()
            else:
                target = np.asarray(
                    self.policy.target(name, self.time_s, positions), dtype=float
                ).copy()
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
                self.positions, self.active, self.time_s, self.link_model, self.roles
            )
            self.reachable = reachable_from(self.graph, self.ground_id)
        frame = self._frame(updated)
        self.history.append(frame)
        return frame

    def run(self):
        while not self.done:
            self.step()
        return self.history

    # Hand edits. Map points are (x, y) in meters, the same axes as the config.

    def add_drone(self, role, xy_m) -> str:
        """Add a drone at map point xy_m = (x, y); return its new ID.

        role: "relay" (a network drone) or "mapper" (a mapping drone). The drone
        flies at the config's new_<role>_altitude_m and new_<role>_speed_m_s and
        holds its spot until it gets a goal. IDs are never reused, so a name like
        relay_3 means one drone in the saved logs.
        """
        if role not in ("relay", "mapper"):
            raise ValueError("Only network drones and mapping drones can be added")
        number = 1
        while f"{role}_{number}" in self.roles:  # skip names already used this run
            number += 1
        name = f"{role}_{number}"
        position = self._map_point(xy_m, self.config[f"new_{role}_altitude_m"])
        self.agents.append(
            {
                "id": name,
                "role": role,
                "position_m": position.tolist(),
                "speed_m_s": self.config[f"new_{role}_speed_m_s"],
                "waypoints_m": [],
                "active": True,
            }
        )
        self.roles[name] = role
        self.positions[name] = position
        self.active[name] = True
        self.manual_targets[name] = position.copy()
        self._after_edit()
        return name

    def remove_drone(self, name):
        """Take drone `name` out of the run. The base and the last mapper stay."""
        if name not in self.positions or name == self.ground_id:
            raise ValueError(f"{name!r} is not a drone that can be removed")
        mappers = [other for other in self.positions if self.roles[other] == "mapper"]
        if mappers == [name]:
            raise ValueError("The run needs at least one mapping drone")
        self.agents = [agent for agent in self.agents if agent["id"] != name]
        del self.positions[name]
        del self.active[name]
        self.manual_targets.pop(name, None)  # pop(key, None): no error if missing
        self._after_edit()

    def add_obstacle(self, box):
        """Add an obstacle box (x_min, y_min, x_max, y_max), in meters."""
        x_min, y_min, x_max, y_max = (float(value) for value in box)
        if not (x_min < x_max and y_min < y_max):
            raise ValueError("An obstacle needs x_min < x_max and y_min < y_max")
        self.obstacles.append((x_min, y_min, x_max, y_max))
        self._after_edit()

    def remove_obstacle(self, box):
        """Remove an obstacle box, given its four numbers."""
        self.obstacles.remove(tuple(box))  # remove deletes the first equal item
        self._after_edit()

    def move_agent(self, name, xy_m):
        """Put drone `name` at map point xy_m = (x, y), keeping its altitude.

        The drone then holds that spot, so any waypoints it had left are dropped.
        """
        if name not in self.positions or name == self.ground_id:
            raise ValueError(f"{name!r} is not a drone that can be moved")
        position = self._map_point(xy_m, self.positions[name][2])
        self.positions[name] = position
        self.manual_targets[name] = position.copy()
        self._after_edit()

    def set_goal(self, name, xy_m):
        """Send drone `name` flying to map point xy_m = (x, y) at its own speed.

        Unlike move_agent, nothing jumps: the drone flies there over the next
        steps (while the clock runs) and then holds. Its waypoints are dropped.
        """
        if name not in self.positions or name == self.ground_id:
            raise ValueError(f"{name!r} is not a drone that can fly")
        self.manual_targets[name] = self._map_point(xy_m, self.positions[name][2])
        self._after_edit()

    def set_range(self, role, range_m):
        """Change the radio range of every drone of one role (in meters).

        This needs the default RangeLink link model, which keeps one range per role.
        """
        if not np.isfinite(range_m) or range_m <= 0:
            raise ValueError("Radio ranges must be positive and finite")
        self.link_model.range_m[role] = float(range_m)
        self._after_edit()

    def set_relay_planner(self, planner):
        """Switch automatic network-drone placement on (give a planner) or off (None).

        planner(simulator, positions) returns ({relay ID: (x, y)}, stretch). While
        it's on, every step re-plans from where everyone is right now, and each
        network drone flies toward its planned spot, so the chain follows a moving
        mapper. Hand-set goals for network drones are ignored meanwhile. Switched
        off, every network drone holds wherever it is at that moment.
        """
        self.relay_planner = planner
        if planner is None:
            self.plan_stretch = None
            for name in self.positions:
                if self.roles[name] == "relay":
                    self.manual_targets[name] = self.positions[name].copy()
        self._after_edit()

    @staticmethod
    def _map_point(xy_m, altitude_m):
        """Turn a map point (x, y) and an altitude into a checked 3D position."""
        position = np.array([xy_m[0], xy_m[1], altitude_m], dtype=float)
        if not np.isfinite(position).all():
            raise ValueError("Positions must be finite numbers")
        return position

    def _after_edit(self):
        """Re-check the links now and redo the latest snapshot to show the edit.

        Without this, the change would appear only at the next scheduled link check,
        and the next step would start from a snapshot that doesn't know about an
        added or removed drone. The clock does not move.
        """
        self.edited = True
        self.graph = build_graph(
            self.positions, self.active, self.time_s, self.link_model, self.roles
        )
        self.reachable = reachable_from(self.graph, self.ground_id)
        self.history[-1] = self._frame(True)
