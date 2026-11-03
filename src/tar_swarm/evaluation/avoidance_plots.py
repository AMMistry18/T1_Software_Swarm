"""AV5 check figure for one avoidance run.

Reads states.csv, resolved_config.yaml, avoidance.npz and the sensor
measurement files (l2_obstacles.npz or rangefinder.npz); never the oracle files.
Obstacle outlines on the trajectory panels come from the world config.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Patch, Rectangle

from tar_swarm.perception.obstacles import sector_centers

PATH = "#2a78d6"
LIMITED_LINE = "#eb6834"
MUTED = "#52514e"
OBSTACLE = "#b9b8b2"
NO_RETURN = "#e9e8e4"
STATUS = {"limited": "#fab219", "stop": "#d03b3b", "stale": "#ec835a"}
STATUS["stale_hold"] = STATUS["stale"]
# Near returns dark, far returns light (sequential blue ramp).
NEARNESS = LinearSegmentedColormap.from_list(
    "nearness", ["#0d366b", "#2a78d6", "#86b6ef", "#cde2fb"]
)
HEATMAP_MAX_M = 10.0


def shade_status(axis, times: np.ndarray, statuses: np.ndarray) -> None:
    """Background bands where avoidance was limiting, stopping or stale."""
    edges = np.flatnonzero(statuses[1:] != statuses[:-1]) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(statuses) - 1]
    for start, end in zip(starts, ends):
        status = str(statuses[start])
        if status in STATUS:
            axis.axvspan(
                times[start], times[end], color=STATUS[status], alpha=0.18, lw=0
            )


def draw_obstacles(axis, obstacles: list[dict], plane: str) -> None:
    for obstacle in obstacles:
        x, y, z = obstacle["position_m"]
        sx, sy, sz = obstacle["size_m"]
        if plane == "xy" and obstacle.get("shape") == "cylinder":
            axis.add_patch(Circle((x, y), sx / 2, color=OBSTACLE))
        elif plane == "xy":
            axis.add_patch(Rectangle((x - sx / 2, y - sy / 2), sx, sy, color=OBSTACLE))
        else:
            axis.add_patch(Rectangle((x - sx / 2, z - sz / 2), sx, sz, color=OBSTACLE))


def plot_avoidance(directory: str | Path, path: str | Path | None = None) -> Path:
    directory = Path(directory)
    path = Path(path) if path else directory / "avoidance_check.png"
    config = yaml.safe_load((directory / "resolved_config.yaml").read_text("utf-8"))
    avoidance = np.load(directory / "avoidance.npz")
    agent = str(avoidance["vehicle_id"][0])
    vehicle = next(
        a["vehicle"] for a in config["mission"]["agents"] if a["id"] == agent
    )
    params = config["avoidance"][vehicle]
    with (directory / "states.csv").open(encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if r["vehicle_id"] == agent]
    position = np.array([[float(r[f"position_{a}_m"]) for a in "xyz"] for r in rows])
    mine = avoidance["vehicle_id"] == agent
    ticks, statuses = avoidance["time_s"][mine], avoidance["status"][mine]
    commanded = np.linalg.norm(avoidance["commanded_velocity"][mine][:, :2], axis=1)
    limited = np.linalg.norm(avoidance["limited_velocity"][mine][:, :2], axis=1)
    limit = avoidance["speed_limit_m_s"][mine].astype(float)
    limit[~np.isfinite(limit)] = np.nan
    obstacles = config["world"]["obstacles"]
    l2 = "l2" in params["sources"] and (directory / "l2_obstacles.npz").exists()

    # The sector panel exists only for L2 runs; the TF-Luna is a single beam.
    panels = 4 if l2 else 3
    figure = plt.figure(figsize=(12, 3.8 * panels), layout="constrained")
    grid = figure.add_gridspec(panels, 2, height_ratios=[1.1] + [1] * (panels - 1))
    top, side = figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])
    sectors = figure.add_subplot(grid[1, :]) if l2 else None
    forward = figure.add_subplot(grid[panels - 2, :], sharex=sectors)
    speed = figure.add_subplot(grid[panels - 1, :], sharex=forward)

    reaction = np.flatnonzero(np.isin(statuses, ("limited", "stop")))
    stop = int(np.argmax(position[:, 0]))
    for axis, (i, j), plane in ((top, (0, 1), "xy"), (side, (0, 2), "xz")):
        draw_obstacles(axis, obstacles, plane)
        axis.plot(position[:, i], position[:, j], color=PATH, lw=2, label="flown")
        axis.plot(*position[stop, [i, j]], "o", color=MUTED, ms=8, label="farthest")
        if len(reaction):
            at = int(
                np.searchsorted([float(r["time_s"]) for r in rows], ticks[reaction[0]])
            )
            at = min(at, len(position) - 1)
            axis.plot(
                *position[at, [i, j]],
                "D",
                color=LIMITED_LINE,
                ms=8,
                label="first reaction",
            )
        axis.set_xlabel("x (m)")
        axis.set_ylabel(f"{'xyz'[j]} (m)")
        axis.grid(alpha=0.3)
    top.set_title("Top view")
    side.set_title("Side view")
    top.set_aspect("equal", adjustable="datalim")
    top.legend(loc="upper left", fontsize=8)

    d_stop = params["d_stop_m"]
    if l2:
        found = np.load(directory / "l2_obstacles.npz")
        found_mine = found["vehicle_id"] == agent
        times = found["time_s"][found_mine]
        distance = found["sector_distance_m"][found_mine]
        angles = np.degrees(sector_centers(distance.shape[1]))
        angles = (angles + 180) % 360 - 180
        order = np.argsort(angles)
        step = 360 / distance.shape[1]
        edges_t = np.r_[
            times, times[-1] + (times[-1] - times[-2] if len(times) > 1 else 0.1)
        ]
        edges_a = np.r_[angles[order] - step / 2, angles[order][-1] + step / 2]
        cmap = NEARNESS.copy()
        cmap.set_bad(NO_RETURN)
        mesh = sectors.pcolormesh(
            edges_t,
            edges_a,
            np.ma.masked_invalid(distance[:, order]).T,
            cmap=cmap,
            vmin=0,
            vmax=HEATMAP_MAX_M,
            shading="flat",
        )
        figure.colorbar(
            mesh, ax=sectors, label=f"nearest return (m, >{HEATMAP_MAX_M:g} clipped)"
        )
        cone = params["cone_half_angle_deg"]
        for edge in (-cone, cone):
            sectors.axhline(edge, color=MUTED, lw=1, ls="--")
        sectors.set_ylabel("sector angle from heading (deg)")
        sectors.set_title("L2 sector distances (light grey = no return; dashed = cone)")
        ahead = np.abs(angles) <= cone
        with np.errstate(all="ignore"):
            nearest = np.where(
                np.isfinite(distance[:, ahead]).any(axis=1),
                np.nanmin(
                    np.where(
                        np.isfinite(distance[:, ahead]), distance[:, ahead], np.inf
                    ),
                    axis=1,
                ),
                np.nan,
            )
        forward.plot(
            times, nearest, ".", color=PATH, ms=4, label="nearest in heading cone"
        )
    else:
        readings = np.load(directory / "rangefinder.npz")
        mine_r = readings["vehicle_id"] == agent
        forward.plot(
            readings["time_s"][mine_r],
            readings["distance_m"][mine_r],
            ".",
            color=PATH,
            ms=2,
            label="TF-Luna distance",
        )
    forward.axhline(d_stop, color=STATUS["stop"], lw=1.5, ls="--", label="d_stop")
    forward.set_ylim(bottom=0)
    forward.set_ylabel("distance ahead (m)")
    forward.set_title("Forward distance (gaps = no return)")
    forward.legend(loc="upper right", fontsize=8)
    forward.grid(alpha=0.3)

    speed.plot(ticks, commanded, color=PATH, lw=3.5, alpha=0.5, label="commanded")
    speed.plot(ticks, limited, color=LIMITED_LINE, lw=1.5, label="limited")
    speed.plot(ticks, limit, color=MUTED, lw=1, ls="--", label="speed limit")
    # Scaled to the flown speed: while held, the commanded speed includes the
    # catch-up toward the mission reference running ahead, and leaves the plot.
    top_speed = max(float(np.nanmax(limited)), 0.5) * 1.5
    speed.set_ylim(0, top_speed)
    speed.set_ylabel("horizontal speed (m/s)")
    speed.set_xlabel("time (s)")
    speed.set_title(
        "Commanded vs limited velocity (commanded includes catch-up; clipped)"
    )
    for axis in (forward, speed):
        shade_status(axis, ticks, statuses)
    handles, _ = speed.get_legend_handles_labels()
    handles += [
        Patch(color=STATUS[s], alpha=0.3, label=f"status: {s}")
        for s in ("limited", "stop", "stale")
    ]
    speed.legend(handles=handles, loc="upper right", fontsize=8)
    speed.grid(alpha=0.3)

    figure.suptitle(f"{directory.name}: {agent}")
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path
