"""Rebuild viewer frames from a saved run, without re-simulating.

Reads only measurement-side files (states.csv, lidar_scans.npz,
l2_obstacles.npz, rangefinder.npz, avoidance.npz) plus scene.xml and
resolved_config.yaml; never the oracle (truth) files. Sensor mounts come from the
scene's sites, the same fixed calibration the runner uses.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from tar_swarm.mujoco.overlays import Beam, L2View, VelocityArrow
from tar_swarm.perception.obstacles import heading_rotation, to_level_frame


@dataclass
class Track:
    time_s: NDArray[np.float64]
    position: NDArray[np.float64]  # (N, 3)
    rotation: NDArray[np.float64]  # (N, 3, 3) body to world

    def index(self, time_s: float) -> int:
        """Latest sample at or before time_s (the first sample before that)."""
        found = np.searchsorted(self.time_s, time_s + 1e-9, side="right") - 1
        return int(max(found, 0))


@dataclass
class ReplayRun:
    model: mujoco.MjModel
    config: dict[str, Any]
    tracks: dict[str, Track]
    qpos_address: dict[str, int]
    mounts: dict[str, tuple[NDArray, NDArray]]  # site name -> (rotation, position)
    rangefinder: dict[str, tuple[NDArray, NDArray]] = field(default_factory=dict)
    l2: dict[str, list[tuple[float, NDArray, NDArray, NDArray]]] = field(
        default_factory=dict
    )
    avoidance: dict[str, tuple[NDArray, NDArray, NDArray, NDArray]] = field(
        default_factory=dict
    )

    @property
    def end_time_s(self) -> float:
        return max(float(track.time_s[-1]) for track in self.tracks.values())


@dataclass
class ReplayFrame:
    qpos: NDArray[np.float64]
    beams: dict[str, Beam]
    l2: dict[str, L2View]
    arrows: dict[str, VelocityArrow]


def latest(times: NDArray, time_s: float) -> int:
    """Index of the latest entry at or before time_s, or -1 when none yet."""
    return int(np.searchsorted(times, time_s + 1e-9, side="right") - 1)


def load_run(directory: Path) -> ReplayRun:
    directory = Path(directory)
    config = yaml.safe_load((directory / "resolved_config.yaml").read_text("utf-8"))
    model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    agents = [agent["id"] for agent in config["mission"]["agents"]]

    rows: dict[str, list[dict[str, str]]] = {name: [] for name in agents}
    with (directory / "states.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows[row["vehicle_id"]].append(row)
    tracks = {}
    for name, records in rows.items():
        angles = [
            [float(r[f"{a}_rad"]) for a in ("roll", "pitch", "yaw")] for r in records
        ]
        tracks[name] = Track(
            np.array([float(r["time_s"]) for r in records]),
            np.array([[float(r[f"position_{a}_m"]) for a in "xyz"] for r in records]),
            Rotation.from_euler("xyz", angles).as_matrix(),
        )

    mounts = {}
    for site in range(model.nsite):
        body = model.site_bodyid[site]
        body_rotation = data.xmat[body].reshape(3, 3)
        mounts[model.site(site).name] = (
            body_rotation.T @ data.site_xmat[site].reshape(3, 3),
            body_rotation.T @ (data.site_xpos[site] - data.xpos[body]),
        )
    run = ReplayRun(
        model,
        config,
        tracks,
        {name: int(model.joint(f"{name}_free").qposadr[0]) for name in agents},
        mounts,
    )

    if (directory / "rangefinder.npz").exists():
        readings = dict(np.load(directory / "rangefinder.npz"))
        for name in np.unique(readings["vehicle_id"]):
            mine = readings["vehicle_id"] == name
            run.rangefinder[str(name)] = (
                readings["time_s"][mine],
                readings["distance_m"][mine],
            )
    if (directory / "l2_obstacles.npz").exists():
        # Each NpzFile item access decompresses the array again: read once.
        scans = dict(np.load(directory / "lidar_scans.npz"))
        found = dict(np.load(directory / "l2_obstacles.npz"))
        points, scan_start = scans["points"], scans["scan_start"]
        mask, mask_start = found["ground_mask"], found["mask_start"]
        scan_ends = np.r_[scan_start[1:], len(points)]
        mask_ends = np.r_[mask_start[1:], len(mask)]
        for k, scan in enumerate(found["scan_index"]):
            name = str(found["vehicle_id"][k])
            run.l2.setdefault(name, []).append(
                (
                    float(found["time_s"][k]),
                    points[scan_start[scan] : scan_ends[scan]].astype(float),
                    mask[mask_start[k] : mask_ends[k]],
                    found["sector_distance_m"][k],
                )
            )
    if (directory / "avoidance.npz").exists():
        records = dict(np.load(directory / "avoidance.npz"))
        for name in np.unique(records["vehicle_id"]):
            mine = records["vehicle_id"] == name
            run.avoidance[str(name)] = (
                records["time_s"][mine],
                records["commanded_velocity"][mine],
                records["limited_velocity"][mine],
                records["status"][mine],
            )
    return run


def vehicle_type(run: ReplayRun, name: str) -> str:
    return next(a for a in run.config["mission"]["agents"] if a["id"] == name)[
        "vehicle"
    ]


def replay_frame(run: ReplayRun, time_s: float) -> ReplayFrame:
    qpos = run.model.qpos0.copy()
    beams, views, arrows = {}, {}, {}
    sensors = run.config.get("sensors", {})
    for name, track in run.tracks.items():
        i = track.index(time_s)
        position, rotation = track.position[i], track.rotation[i]
        x, y, z, w = Rotation.from_matrix(rotation).as_quat()
        address = run.qpos_address[name]
        qpos[address : address + 7] = [*position, w, x, y, z]

        if name in run.rangefinder:
            times, distances = run.rangefinder[name]
            k = latest(times, time_s)
            params = sensors["rangefinder"][vehicle_type(run, name)]
            mount_rotation, mount_position = run.mounts[f"{name}_{params['site']}"]
            if k >= 0:
                beams[name] = Beam(
                    position + rotation @ mount_position,
                    rotation @ mount_rotation[:, 2],
                    float(distances[k]),
                    params["effective_max_range_m"],
                )

        scans = run.l2.get(name, [])
        k = latest(np.array([s[0] for s in scans]), time_s) if scans else -1
        if k >= 0:
            scan_time, points, ground, sectors = scans[k]
            j = track.index(scan_time)
            params = sensors["lidar"][vehicle_type(run, name)]
            mount = run.mounts[f"{name}_{params['site']}"]
            views[name] = L2View(
                track.position[j],
                heading_rotation(track.rotation[j]),
                to_level_frame(points, *mount, track.rotation[j]),
                ground,
                sectors,
            )

        if name in run.avoidance:
            times, commanded, limited, status = run.avoidance[name]
            k = latest(times, time_s)
            if k >= 0:
                arrows[name] = VelocityArrow(
                    position, commanded[k], limited[k], str(status[k])
                )
    return ReplayFrame(qpos, beams, views, arrows)
