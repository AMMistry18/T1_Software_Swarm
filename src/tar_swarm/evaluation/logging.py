"""Persist real sampled state and control data in portable CSV files."""

import csv
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from tar_swarm.control.interfaces import ControlOutput, Reference, State

AXES = "xyz"
FIELDS = [
    "time_s",
    "vehicle_id",
    *[f"position_{a}_m" for a in AXES],
    *[f"desired_{a}_m" for a in AXES],
    *[f"velocity_{a}_m_s" for a in AXES],
    *[f"desired_velocity_{a}_m_s" for a in AXES],
    "roll_rad",
    "pitch_rad",
    "yaw_rad",
    *[f"omega_{a}_rad_s" for a in AXES],
    *[f"motor_{i}_n" for i in range(4)],
    "motor_saturated",
    "acceleration_limited",
    "contacts",
]


def sample(
    time_s: float,
    name: str,
    state: State,
    desired: Reference,
    output: ControlOutput,
    contacts: int,
) -> dict[str, Any]:
    angles = Rotation.from_matrix(state.rotation).as_euler("xyz")
    values = [
        time_s,
        name,
        *state.position,
        *desired.position,
        *state.velocity,
        *desired.velocity,
        *angles,
        *state.angular_velocity,
        *output.thrusts,
        int(output.motor_saturated),
        int(output.acceleration_limited),
        contacts,
    ]
    return dict(zip(FIELDS, values, strict=True))


def metrics(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for agent in config["mission"]["agents"]:
        records = [r for r in rows if r["vehicle_id"] == agent["id"]]
        times = np.array([r["time_s"] for r in records])
        error = np.array(
            [
                [r[f"position_{a}_m"] - r[f"desired_{a}_m"] for a in AXES]
                for r in records
            ]
        )
        hover = np.zeros(len(times), dtype=bool)
        for start, end in config["mission"].get("hover_windows_s", []):
            hover |= (times >= start) & (times <= end)
        final = times >= min(times[-1], config["simulation"]["duration_s"] - 2)
        result[agent["id"]] = {
            "position_rmse_m": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
            "max_position_error_m": float(np.max(np.linalg.norm(error, axis=1))),
            "hover_altitude_rmse_m": float(np.sqrt(np.mean(error[hover, 2] ** 2)))
            if hover.any()
            else None,
            "final_hover_rmse_m": float(
                np.sqrt(np.mean(np.sum(error[final] ** 2, axis=1)))
            ),
            "motor_saturation_samples": sum(r["motor_saturated"] for r in records),
            "acceleration_limit_samples": sum(
                r["acceleration_limited"] for r in records
            ),
            "samples": len(records),
        }
    return result


def save_run(
    directory: Path,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    xml: str,
    controls: list[dict[str, Any]],
    completed: bool,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "states.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    control_fields = [
        "time_s",
        "vehicle_id",
        *[f"motor_{i}_n" for i in range(4)],
        "motor_saturated",
        "acceleration_limited",
    ]
    with (directory / "controls.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=control_fields)
        writer.writeheader()
        writer.writerows(controls)
    report = metrics(rows, config)
    report["run"] = {
        "completed": completed,
        "final_time_s": rows[-1]["time_s"],
        "control_updates": len(controls),
        "motor_saturation_updates": sum(c["motor_saturated"] for c in controls),
        "max_contacts": max(r["contacts"] for r in rows),
    }
    (directory / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (directory / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (directory / "scene.xml").write_text(xml, encoding="utf-8")
    environment = {
        name: version(name)
        for name in ("mujoco", "numpy", "scipy", "matplotlib", "pyyaml")
    }
    (directory / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    return report
