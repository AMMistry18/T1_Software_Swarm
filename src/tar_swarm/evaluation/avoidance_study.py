"""AV5 sweep: cases, one-case runner, and pass criteria (evaluation only)."""

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from tar_swarm.evaluation.avoidance_metrics import run_metrics
from tar_swarm.evaluation.scenarios import (
    HOVER_S,
    SENSORS,
    TAKEOFF_S,
    WORLDS,
    avoidance_config,
    ground_edge_margin,
)
from tar_swarm.mujoco.recording import record_replay
from tar_swarm.mujoco.runner import run_scenario

SPEEDS = (1.0, 2.0, 3.0, 5.0)
SEEDS = (42, 43, 44)
NOISE = (True, False)
# Pass criteria from the brief; the other worlds are reported as limits.
REQUIRED_WORLDS = ("avoid_wall", "avoid_pole")
REQUIRED_MAX_SPEED_M_S = 2.0
FALSE_STOP_WORLDS = ("clear_path",)


@dataclass(frozen=True)
class Case:
    world: str
    sensor: str
    speed_m_s: float
    noise: bool
    seed: int

    @property
    def name(self) -> str:
        noise = "noisy" if self.noise else "clean"
        return f"{self.world}_{self.sensor}_v{self.speed_m_s:g}_{noise}_s{self.seed}"


def cases(
    worlds=WORLDS, sensors=tuple(SENSORS), speeds=SPEEDS, noise=NOISE, seeds=SEEDS
) -> list[Case]:
    return [Case(*values) for values in product(worlds, sensors, speeds, noise, seeds)]


def run_case(
    case: Case, root: Path, record: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run one case into root/<case name>; clip failures when record is given.

    record is the overlay settings used to draw the clip from the saved run.
    """
    config = avoidance_config(
        case.world, case.sensor, case.speed_m_s, seed=case.seed, noise=case.noise
    )
    output = Path(root) / case.name
    result = run_scenario(config, output=output, plots=False)
    agent = config["mission"]["agents"][0]["id"]
    positions = np.array(
        [
            [r[f"position_{a}_m"] for a in "xyz"]
            for r in result["rows"]
            if r["vehicle_id"] == agent
        ]
    )
    metrics = run_metrics(output)
    row = {
        "name": case.name,
        **asdict(case),
        **asdict(metrics),
        "completed": result["completed"],
        "ground_edge_margin_m": ground_edge_margin(config, positions),
        "clip": "",
    }
    if record is not None and (metrics.collided or metrics.false_stop):
        clip = record_replay(
            output, output / "clip.gif", record, start_s=TAKEOFF_S + HOVER_S
        )
        row["clip"] = str(clip) if clip else ""
    return row


def criteria(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pass/fail for each criterion, with the names of the failing runs."""

    def check(name: str, failing: list[dict[str, Any]], checked: int) -> dict:
        return {
            "criterion": name,
            "passed": not failing and checked > 0,
            "runs_checked": checked,
            "failures": [r["name"] for r in failing],
        }

    required = [
        r
        for r in rows
        if r["world"] in REQUIRED_WORLDS and r["speed_m_s"] <= REQUIRED_MAX_SPEED_M_S
    ]
    clear = [r for r in rows if r["world"] in FALSE_STOP_WORLDS]
    return [
        check(
            "zero collisions on avoid_wall and avoid_pole at <= 2 m/s",
            [r for r in required if r["collided"]],
            len(required),
        ),
        check(
            "zero false stops on clear_path",
            [r for r in clear if r["false_stop"]],
            len(clear),
        ),
        check(
            "every run completed with ground-edge margin >= 0",
            [r for r in rows if not r["completed"] or r["ground_edge_margin_m"] < 0],
            len(rows),
        ),
    ]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One line per world, sensor, speed and noise setting, over seeds."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["world"], row["sensor"], row["speed_m_s"], row["noise"])
        groups.setdefault(key, []).append(row)

    def stat(values: list[float], reduce) -> float:
        finite = [v for v in values if np.isfinite(v)]
        return float(reduce(finite)) if finite else float("nan")

    summary = []
    for (world, sensor, speed, noise), group in sorted(groups.items()):
        summary.append(
            {
                "world": world,
                "sensor": sensor,
                "speed_m_s": speed,
                "noise": noise,
                "runs": len(group),
                "collisions": sum(r["collided"] for r in group),
                "false_stops": sum(r["false_stop"] for r in group),
                "min_clearance_m": stat([r["min_clearance_m"] for r in group], min),
                "mean_stopping_distance_m": stat(
                    [r["stopping_distance_m"] for r in group], np.mean
                ),
                "mean_time_to_stop_s": stat(
                    [r["time_to_stop_s"] for r in group], np.mean
                ),
                "mean_reaction_speed_m_s": stat(
                    [r["reaction_speed_m_s"] for r in group], np.mean
                ),
            }
        )
    return summary
