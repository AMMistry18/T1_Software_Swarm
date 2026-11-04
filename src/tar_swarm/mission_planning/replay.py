"""Visual / file replays of coverage, motion, and links."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import PathPatch
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.ops import unary_union

from tar_swarm.mission_planning.communication import (
    ConservativeDistanceLink,
    build_connectivity,
)
from tar_swarm.mission_planning.geometry import polygon_from_ring
from tar_swarm.mission_planning.schemas import PreparedScenario
from tar_swarm.mission_planning.simulator import PlanningSimulator


def save_replay(
    scenario: PreparedScenario,
    sim: PlanningSimulator,
    output_dir: str | Path,
    *,
    save_gif: bool = False,
    gif_stride: int = 2,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    with history_path.open("w", encoding="utf-8") as fh:
        for frame in sim.history:
            serial = {
                "time_s": frame["time_s"],
                "positions": {
                    k: v.tolist() for k, v in frame["positions"].items()
                },
                "modes": frame["modes"],
                "coverage_fraction": frame["coverage_fraction"],
                "metrics": frame["metrics"],
            }
            fh.write(json.dumps(serial) + "\n")

    fig_path = output_dir / "replay.png"
    _plot_replay(scenario, sim, fig_path)
    metrics_path = output_dir / "metrics.json"
    metrics = {
        "coverage_fraction": sim.metrics.coverage_fraction,
        "disconnected_mapper_s": sim.metrics.disconnected_mapper_s,
        "relay_travel_m": sim.metrics.relay_travel_m,
        "relay_repositions": sim.metrics.relay_repositions,
        "elapsed_s": sim.metrics.elapsed_s,
        "mission_complete": sim.metrics.mission_complete,
        "term_reason": sim.term_reason,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    out = {
        "history": str(history_path),
        "figure": str(fig_path),
        "metrics": str(metrics_path),
    }
    if save_gif:
        gif_path = output_dir / "replay.gif"
        animate_replay(
            scenario,
            sim,
            save_path=gif_path,
            show=False,
            stride=gif_stride,
        )
        out["gif"] = str(gif_path)
    return out


def covered_geometry_at(
    scenario: PreparedScenario,
    history: list[dict[str, Any]],
    end_index: int,
):
    """Rebuild scanned footprint through ``history[end_index]`` (survey mode only)."""
    region = polygon_from_ring(scenario.request.region_polygon_m)
    half = 0.5 * float(scenario.request.sensor.swath_m)
    mapper_ids = [m.id for m in scenario.request.mappers]
    pieces = []
    end_index = max(0, min(end_index, len(history) - 1))
    for i in range(1, end_index + 1):
        prev = history[i - 1]
        cur = history[i]
        for mid in mapper_ids:
            # Count coverage while either endpoint is surveying.
            if prev["modes"].get(mid) != "survey" and cur["modes"].get(mid) != "survey":
                continue
            a = prev["positions"][mid]
            b = cur["positions"][mid]
            start_xy = (float(a[0]), float(a[1]))
            end_xy = (float(b[0]), float(b[1]))
            if np.linalg.norm(np.asarray(end_xy) - np.asarray(start_xy)) < 1e-9:
                footprint = Point(start_xy).buffer(half)
            else:
                footprint = LineString([start_xy, end_xy]).buffer(
                    half, cap_style=2, join_style=2
                )
            piece = footprint.intersection(region)
            if not piece.is_empty:
                pieces.append(piece)
    if not pieces:
        return None
    return unary_union(pieces).intersection(region)


def _iter_polygons(geom):
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        yield from geom.geoms
    elif isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            yield from _iter_polygons(g)


def _polygon_to_pathpatch(poly: Polygon, **kwargs) -> PathPatch:
    arrays = [np.asarray(poly.exterior.coords)]
    for interior in poly.interiors:
        arrays.append(np.asarray(interior.coords))
    codes = []
    verts = []
    for ring in arrays:
        codes.append(MplPath.MOVETO)
        codes.extend([MplPath.LINETO] * (len(ring) - 2))
        codes.append(MplPath.CLOSEPOLY)
        verts.extend(ring.tolist())
    path = MplPath(verts, codes)
    return PathPatch(path, **kwargs)


def add_coverage_shade(ax, covered, *, label: str | None = "scanned"):
    """Shade scanned polygons on ``ax``. Returns artists added."""
    artists = []
    first = True
    for poly in _iter_polygons(covered):
        patch = _polygon_to_pathpatch(
            poly,
            facecolor="#4c78a8",
            edgecolor="none",
            alpha=0.35,
            zorder=1,
            label=label if first else None,
        )
        ax.add_patch(patch)
        artists.append(patch)
        first = False
    return artists


def animate_replay(
    scenario: PreparedScenario,
    sim: PlanningSimulator,
    *,
    save_path: str | Path | None = None,
    show: bool = True,
    stride: int = 2,
    interval_ms: int = 80,
):
    """Animate drone motion, scanned area, trails, and links."""
    history = sim.history
    if len(history) < 2:
        raise ValueError("Need at least two history frames to animate")
    indices = list(range(0, len(history), max(1, stride)))
    if indices[-1] != len(history) - 1:
        indices.append(len(history) - 1)

    # Precompute scanned footprints for animation frames.
    covered_by_frame = [
        covered_geometry_at(scenario, history, idx) for idx in indices
    ]

    ring = np.asarray(
        scenario.request.region_polygon_m + [scenario.request.region_polygon_m[0]]
    )
    base = np.asarray(scenario.request.base_position_m, dtype=float)
    link = ConservativeDistanceLink(
        scenario.request.communication.max_range_m,
        scenario.request.communication.connection_margin_m,
    )
    colors = {"mapper": "C0", "relay": "C1"}
    drones = list(scenario.request.drones)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax.plot(ring[:, 0], ring[:, 1], "k-", lw=1.5, zorder=3)
    ax.scatter([base[0]], [base[1]], c="black", marker="s", s=40, zorder=5)
    for cand in scenario.candidates:
        ax.scatter(
            [cand.position_m[0]],
            [cand.position_m[1]],
            c="0.75",
            s=10,
            marker="x",
            zorder=2,
        )
    xs = [p[0] for p in scenario.request.region_polygon_m] + [base[0]]
    ys = [p[1] for p in scenario.request.region_polygon_m] + [base[1]]
    pad = 8.0
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title("Mission animation")

    shade_artists: list = []
    # Legend proxy for scanned area.
    ax.add_patch(
        MplPolygon(
            [[0, 0]],
            closed=True,
            facecolor="#4c78a8",
            edgecolor="none",
            alpha=0.35,
            label="scanned",
            visible=False,
        )
    )

    trails = {}
    nodes = {}
    for drone in drones:
        color = colors.get(drone.role.value, "C2")
        (trail,) = ax.plot(
            [], [], color=color, alpha=0.85, lw=1.5, label=drone.id, zorder=4
        )
        (node,) = ax.plot(
            [],
            [],
            marker="o",
            color=color,
            linestyle="none",
            markersize=8,
            zorder=5,
        )
        trails[drone.id] = trail
        nodes[drone.id] = node
    edge_lines = []
    for _ in range(32):
        (line,) = ax.plot([], [], color="green", alpha=0.45, lw=1.2, zorder=3)
        edge_lines.append(line)
    title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(fontsize=7, loc="best")

    times = [f["time_s"] for f in history]
    cov = [f["coverage_fraction"] for f in history]
    (cov_line,) = ax2.plot([], [], label="coverage fraction")
    ax2.set_xlim(0, max(times[-1], 1e-6))
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("time s")
    ax2.set_title("Progress")
    ax2.legend(fontsize=8)
    time_mark = ax2.axvline(0.0, color="red", lw=1, alpha=0.7)

    def _update(frame_i: int):
        nonlocal shade_artists
        idx = indices[frame_i]
        frame = history[idx]
        for artist in shade_artists:
            artist.remove()
        shade_artists = add_coverage_shade(
            ax, covered_by_frame[frame_i], label=None
        )
        for drone in drones:
            trail_xy = np.asarray(
                [history[j]["positions"][drone.id][:2] for j in range(idx + 1)]
            )
            trails[drone.id].set_data(trail_xy[:, 0], trail_xy[:, 1])
            pos = frame["positions"][drone.id]
            nodes[drone.id].set_data([pos[0]], [pos[1]])
        positions = {k: np.asarray(v) for k, v in frame["positions"].items()}
        positions["ground"] = base.copy()
        active = {k: True for k in positions}
        graph, _ = build_connectivity(
            positions, active, frame["time_s"], link, "ground"
        )
        for line in edge_lines:
            line.set_data([], [])
        for e_i, (a, b) in enumerate(graph.edges):
            if e_i >= len(edge_lines):
                break
            pa, pb = positions[a], positions[b]
            edge_lines[e_i].set_data([pa[0], pb[0]], [pa[1], pb[1]])
        cov_line.set_data(times[: idx + 1], cov[: idx + 1])
        time_mark.set_xdata([frame["time_s"], frame["time_s"]])
        title.set_text(
            f"t = {frame['time_s']:.1f}s  coverage = {frame['coverage_fraction']:.2f}"
        )
        return (
            *shade_artists,
            *trails.values(),
            *nodes.values(),
            *edge_lines,
            cov_line,
            time_mark,
            title,
        )

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=len(indices),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(save_path, writer="pillow", fps=max(1, int(1000 / interval_ms)))
    if show:
        plt.show()
    else:
        plt.close(fig)
    return anim


def _plot_replay(scenario: PreparedScenario, sim: PlanningSimulator, path: Path):
    ring = scenario.request.region_polygon_m
    poly = np.asarray(ring + [ring[0]])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    ax = axes[0]
    covered = covered_geometry_at(scenario, sim.history, len(sim.history) - 1)
    add_coverage_shade(ax, covered, label="scanned")
    ax.plot(poly[:, 0], poly[:, 1], "k-", lw=1.5, label="region", zorder=3)
    base = scenario.request.base_position_m
    ax.scatter(
        [base[0]], [base[1]], c="black", marker="s", s=40, label="base", zorder=5
    )
    for cand in scenario.candidates:
        ax.scatter(
            [cand.position_m[0]],
            [cand.position_m[1]],
            c="0.7",
            s=10,
            marker="x",
            zorder=2,
        )

    colors = {"mapper": "C0", "relay": "C1"}
    for drone in scenario.request.drones:
        trail = np.asarray(
            [
                f["positions"][drone.id][:2]
                for f in sim.history
                if drone.id in f["positions"]
            ]
        )
        if len(trail):
            ax.plot(
                trail[:, 0],
                trail[:, 1],
                color=colors.get(drone.role.value, "C2"),
                alpha=0.85,
                label=drone.id,
                zorder=4,
            )
            ax.scatter(
                [trail[-1, 0]],
                [trail[-1, 1]],
                color=colors.get(drone.role.value, "C2"),
                zorder=5,
            )

    if sim.history:
        last = sim.history[-1]
        positions = {k: np.asarray(v) for k, v in last["positions"].items()}
        positions["ground"] = np.asarray(base, dtype=float)
        active = {k: True for k in positions}
        link = ConservativeDistanceLink(
            scenario.request.communication.max_range_m,
            scenario.request.communication.connection_margin_m,
        )
        graph, _ = build_connectivity(positions, active, last["time_s"], link, "ground")
        for a, b in graph.edges:
            pa, pb = positions[a], positions[b]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], "g-", alpha=0.4, lw=1, zorder=3)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Coverage / motion / links")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")

    ax2 = axes[1]
    times = [f["time_s"] for f in sim.history]
    cov = [f["coverage_fraction"] for f in sim.history]
    disc = [f["metrics"]["disconnected_mapper_s"] for f in sim.history]
    ax2.plot(times, cov, label="coverage fraction")
    ax2.plot(
        times,
        np.asarray(disc) / max(times[-1], 1e-6),
        label="cum disconnect frac",
    )
    ax2.set_xlabel("time s")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.set_title("Progress")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def replay_history_file(history_jsonl: str | Path, output_png: str | Path) -> None:
    """Minimal replot from saved history without full scenario (positions only)."""
    frames: list[dict[str, Any]] = []
    with Path(history_jsonl).open(encoding="utf-8") as fh:
        for line in fh:
            frames.append(json.loads(line))
    fig, ax = plt.subplots(figsize=(5, 5))
    ids = sorted(frames[0]["positions"])
    for did in ids:
        trail = np.asarray([f["positions"][did][:2] for f in frames])
        ax.plot(trail[:, 0], trail[:, 1], label=did)
    ax.legend(fontsize=7)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_png, dpi=120)
    plt.close(fig)
