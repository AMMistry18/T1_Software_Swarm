"""AV5 avoidance metrics from a saved run directory (evaluation only).

Clearance and collisions compare the true obstacle geometry (scene.xml) with
the logged poses; that is evaluation, so it may use ground truth. Reaction,
stopping and false-stop metrics come from the logged avoidance status.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from tar_swarm.evaluation.scenarios import LANE_END_X_M, TAKEOFF_S
from tar_swarm.mujoco.replay import load_run

REACTING = ("limited", "stop")
# Horizontal speed below which the drone counts as stopped.
STOPPED_M_S = 0.1
# Reached the lane end if within this of it.
LANE_END_TOLERANCE_M = 0.5
# Clearance search limit; farther obstacles report this distance.
CLEARANCE_SEARCH_M = 5.0


@dataclass
class RunMetrics:
    agent: str
    collided: bool  # airframe touched an obstacle
    contacts: int  # most MuJoCo contacts in one logged sample after takeoff
    min_clearance_m: float  # airframe to obstacle; NaN without obstacles
    reaction_time_s: float  # first limited/stop tick; NaN if never
    reaction_x_m: float
    reaction_speed_m_s: float  # horizontal speed at first reaction
    stop_x_m: float  # farthest x along the lane
    stopping_distance_m: float  # stop_x - reaction_x
    time_to_stop_s: float  # reaction to horizontal speed < STOPPED_M_S
    intervention_ticks: int  # non-clear ticks after takeoff starts
    reached_end: bool
    false_stop: bool  # intervened, or never finished, with no obstacle in the world


def first_reaction(
    times: NDArray, statuses: NDArray, after_s: float = TAKEOFF_S
) -> int | None:
    """Index of the first limited/stop tick at or after after_s, or None."""
    found = np.flatnonzero(np.isin(statuses, REACTING) & (times >= after_s))
    return int(found[0]) if len(found) else None


def time_to_stop(
    times: NDArray, speeds: NDArray, start_s: float, threshold: float = STOPPED_M_S
) -> float:
    """Seconds from start_s until speed first drops below threshold; NaN if never."""
    found = np.flatnonzero((times >= start_s) & (speeds < threshold))
    return float(times[found[0]] - start_s) if len(found) else np.nan


def interventions(times: NDArray, statuses: NDArray, after_s: float = TAKEOFF_S) -> int:
    """Non-clear ticks from after_s on (skips the start-up tick before any reading)."""
    return int(np.sum((statuses != "clear") & (times >= after_s)))


def is_false_stop(
    has_obstacles: bool, intervention_ticks: int, reached_end: bool
) -> bool:
    return not has_obstacles and (intervention_ticks > 0 or not reached_end)


def min_clearance(directory: Path, agent: str | None = None) -> float:
    """Smallest airframe-to-obstacle distance over the logged poses (NaN if none)."""
    run = load_run(directory)
    model = run.model
    agent = agent or next(iter(run.tracks))
    obstacles = [model.geom(o["name"]).id for o in run.config["world"]["obstacles"]]
    if not obstacles:
        return np.nan
    root = model.body(agent).id
    airframe = [
        g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] == root
    ]
    data = mujoco.MjData(model)
    address = run.qpos_address[agent]
    track = run.tracks[agent]
    quaternions = np.empty((len(track.time_s), 4))
    for i, rotation in enumerate(track.rotation):
        mujoco.mju_mat2Quat(quaternions[i], rotation.ravel())
    best = np.inf
    data.qpos[:] = model.qpos0
    for position, quaternion in zip(track.position, quaternions):
        data.qpos[address : address + 7] = [*position, *quaternion]
        mujoco.mj_kinematics(model, data)
        for obstacle in obstacles:
            for geom in airframe:
                distance = mujoco.mj_geomDistance(
                    model, data, obstacle, geom, CLEARANCE_SEARCH_M, None
                )
                best = min(best, distance)
    return float(best)


def run_metrics(directory: Path) -> RunMetrics:
    directory = Path(directory)
    with (directory / "states.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    avoidance = np.load(directory / "avoidance.npz")
    agent = str(avoidance["vehicle_id"][0])
    rows = [r for r in rows if r["vehicle_id"] == agent]
    times = np.array([float(r["time_s"]) for r in rows])
    x = np.array([float(r["position_x_m"]) for r in rows])
    speed = np.hypot(
        [float(r["velocity_x_m_s"]) for r in rows],
        [float(r["velocity_y_m_s"]) for r in rows],
    )
    contacts = max(
        (int(r["contacts"]) for r, t in zip(rows, times) if t > TAKEOFF_S / 2),
        default=0,
    )

    mine = avoidance["vehicle_id"] == agent
    ticks, statuses = avoidance["time_s"][mine], avoidance["status"][mine]
    stop_x = float(x.max())
    reaction = first_reaction(ticks, statuses)
    if reaction is None:
        reaction_time = reaction_x = reaction_speed = stopping = stop_time = np.nan
    else:
        reaction_time = float(ticks[reaction])
        at = int(np.searchsorted(times, reaction_time - 1e-9))
        reaction_x, reaction_speed = float(x[at]), float(speed[at])
        stopping = stop_x - reaction_x
        stop_time = time_to_stop(times, speed, reaction_time)

    clearance = min_clearance(directory, agent)
    has_obstacles = bool(np.isfinite(clearance))
    ticks_on = interventions(ticks, statuses)
    reached = stop_x >= LANE_END_X_M - LANE_END_TOLERANCE_M
    return RunMetrics(
        agent=agent,
        collided=bool(has_obstacles and clearance <= 0.0) or contacts > 0,
        contacts=contacts,
        min_clearance_m=clearance,
        reaction_time_s=reaction_time,
        reaction_x_m=reaction_x,
        reaction_speed_m_s=reaction_speed,
        stop_x_m=stop_x,
        stopping_distance_m=stopping,
        time_to_stop_s=stop_time,
        intervention_ticks=ticks_on,
        reached_end=bool(reached),
        false_stop=is_false_stop(has_obstacles, ticks_on, reached),
    )
