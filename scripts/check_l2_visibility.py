"""AV2 pre-check: can the downward L2 see a wall straight ahead at flight height?

Places the mapper at a fixed hover pose in front of a wall (no flight), scans
with max_polar_deg 90 and 96 at several pitch angles, runs obstacle extraction,
and reports forward-sector returns and the visible part of the height band.
Also measures airframe self-occlusion, which the main simulation ignores
because the scanning drone's own body is excluded from ray casts.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tar_swarm.common.config import load_config  # noqa: E402
from tar_swarm.mujoco.flight import FlightSimulation  # noqa: E402
from tar_swarm.mujoco.overlays import L2View, draw_sensor_overlays  # noqa: E402
from tar_swarm.perception.obstacles import (  # noqa: E402
    extract_obstacles,
    heading_rotation,
    load_obstacle_params,
)
from tar_swarm.sensors.lidar import (  # noqa: E402
    apply_noise,
    load_lidar_params,
    scan_directions,
    to_points,
)

ROOT = Path(__file__).resolve().parents[1]
NAME = "mapper_0"
POLARS = (90.0, 96.0)
PITCHES = (0.0, 10.0, -10.0)  # positive = nose down, as when flying forward
FORWARD = np.arange(-2, 3)  # sectors within +/-12.5 deg of the heading


def check_config(polar: float, altitude: float, wall_x: float) -> dict:
    config = load_config(ROOT / "configs/simulation.yaml")
    config["world"] = {
        "name": "l2_wall_check",
        "ground_half_size_m": 60.0,  # beyond the 30 m L2 range in every direction
        "ground_station": {"position_m": [-40, -40, 0.4], "size_m": [0.8, 0.6, 0.8]},
        "obstacles": [
            {
                "name": "wall",
                "position_m": [wall_x + 0.1, 0, 4],
                "size_m": [0.2, 20, 8],
            }
        ],
    }
    start = [0.0, 0.0, altitude]
    config["mission"]["agents"] = [
        {
            "id": NAME,
            "vehicle": "mapping_drone",
            "spawn_m": start,
            "yaw_rad": 0.0,
            "waypoints": [{"time_s": 0, "position_m": start}],
        }
    ]
    config["sensors"]["lidar"]["mapping_drone"]["max_polar_deg"] = polar
    config["sensors"].pop("rangefinder", None)
    return config


def place(sim: FlightSimulation, altitude: float, pitch_deg: float) -> None:
    address = sim.model.joint(f"{NAME}_free").qposadr[0]
    half = np.radians(pitch_deg) / 2
    sim.data.qpos[address : address + 7] = [
        0,
        0,
        altitude,
        np.cos(half),
        0,
        np.sin(half),
        0,
    ]
    sim.data.qvel[:] = 0
    mujoco.mj_forward(sim.model, sim.data)


def run_case(polar, pitch, args, rng):
    config = check_config(polar, args.altitude, args.distance)
    sim = FlightSimulation(config)
    place(sim, args.altitude, pitch)
    lidar = load_lidar_params(config["sensors"], "mapping_drone")
    obstacles = load_obstacle_params(config["perception"], "mapping_drone")
    site = f"{NAME}_{lidar.site}"
    body = sim.state(NAME)
    site_position, site_rotation = sim.site_pose(site)
    mount = (
        body.rotation.T @ site_rotation,
        body.rotation.T @ (site_position - body.position),
    )
    returns, distances, counts, wall_z = [], [], [], []
    for _ in range(args.scans):
        directions = scan_directions(lidar, rng)
        truth, _, _ = sim.cast_rays(NAME, site, directions, lidar.max_range_m)
        points = to_points(directions, apply_noise(truth, lidar, rng))
        found = extract_obstacles(points, *mount, body.rotation, obstacles, rng)
        forward = found.sector_distance_m[FORWARD % obstacles.num_sectors]
        returns.append(np.isfinite(forward))
        distances.append(forward)
        counts.append(found.sector_count[FORWARD % obstacles.num_sectors].sum())
        level = found.points_level[~found.ground_mask]
        # Wall points inside the forward sectors only (not the wall's far ends).
        half_width = np.radians(2.5) * (len(FORWARD))
        on_wall = (np.abs(level[:, 0] - args.distance) < 0.3) & (
            np.abs(np.arctan2(level[:, 1], level[:, 0])) <= half_width
        )
        wall_z.append(level[on_wall, 2])
    wall_z = np.concatenate(wall_z)
    band = (wall_z >= -obstacles.band_below_m) & (wall_z <= obstacles.band_above_m)
    result = {
        "max_polar_deg": polar,
        "pitch_deg": pitch,
        "forward_return_rate": float(np.mean(returns)),
        "sector0_distance_m": float(np.nanmean(column))
        if np.isfinite(column := np.array(distances)[:, 2]).any()
        else None,
        "forward_points_per_scan": float(np.mean(counts)),
        "wall_band_top_seen_m": float(wall_z[band].max()) if band.any() else None,
        "wall_band_bottom_seen_m": float(wall_z[band].min()) if band.any() else None,
        "sensor_z_in_body_m": float(mount[1][2]),
    }
    view = L2View(
        body.position,
        heading_rotation(body.rotation),
        found.points_level,
        found.ground_mask,
        found.sector_distance_m,
    )
    return result, sim, view, wall_z


def self_occlusion(args) -> dict:
    """Fraction of L2 rays blocked by the drone's own exterior geoms, by polar angle."""
    config = check_config(96.0, args.altitude, args.distance)
    sim = FlightSimulation(config)
    place(sim, args.altitude, 0.0)
    model, data = sim.model, sim.data
    sensor_geom = model.geom(f"{NAME}_unitree_l2").id
    body_id = sim.handles[NAME].body_id
    # Exterior (group 0) geoms only; the L2 housing itself is moved out of the mask.
    model.geom_group[sensor_geom] = 5
    group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    origin, rotation = sim.site_pose(f"{NAME}_lidar_down")
    lidar = load_lidar_params(config["sensors"], "mapping_drone")
    params = type(lidar)(**{**lidar.__dict__, "points_per_scan": 200_000})
    directions = scan_directions(params, np.random.default_rng(0))
    count = len(directions)
    geom_id = np.empty(count, dtype=np.int32)
    distance = np.empty(count)
    mujoco.mj_multiRay(
        model,
        data,
        origin,
        np.ascontiguousarray(directions @ rotation.T).ravel(),
        group,
        True,
        -1,
        geom_id,
        distance,
        None,
        count,
        5.0,
    )
    own = (geom_id >= 0) & (model.geom_bodyid[np.maximum(geom_id, 0)] == body_id)
    polar = np.degrees(np.arccos(np.clip(directions[:, 2], -1, 1)))
    edges = np.arange(0, 97, 6)
    which = np.digitize(polar, edges) - 1
    fraction = [float(own[which == i].mean()) for i in range(len(edges) - 1)]
    names, hits = np.unique(geom_id[own], return_counts=True)
    return {
        "polar_bin_edges_deg": edges.tolist(),
        "blocked_fraction": fraction,
        "blocked_total": float(own.mean()),
        "blocking_geoms": {
            model.geom(int(g)).name: int(h) for g, h in zip(names, hits, strict=True)
        },
    }


def render(sim, view, path):
    settings = {
        "lidar_points": {"enabled": True, "max_points": 3000},
        "sector_ring": {"enabled": True, "d_stop_m": 1.0, "no_return_length_m": 0.3},
    }
    camera = mujoco.MjvCamera()
    camera.lookat[:] = view.body_position + [2.5, 0, -0.5]
    camera.distance, camera.azimuth, camera.elevation = 8.0, 65.0, -20.0
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    with mujoco.Renderer(sim.model, height=600, width=900) as renderer:
        renderer.update_scene(sim.data, camera=camera, scene_option=option)
        draw_sensor_overlays(renderer.scene, settings, {}, {NAME: view})
        plt.imsave(path, renderer.render())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scans", type=int, default=20)
    parser.add_argument("--distance", type=float, default=5.0, help="Wall face, m")
    parser.add_argument("--altitude", type=float, default=3.0)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(0)
    cases, side = [], {}
    for polar in POLARS:
        for pitch in PITCHES:
            result, sim, view, wall_z = run_case(polar, pitch, args, rng)
            cases.append(result)
            side[(polar, pitch)] = wall_z
            if not args.no_render and pitch == 0.0:
                render(sim, view, args.output / f"wall_polar{polar:.0f}.png")
    occlusion = self_occlusion(args)
    report = {
        "wall_distance_m": args.distance,
        "altitude_m": args.altitude,
        "cases": cases,
        "self_occlusion": occlusion,
    }
    (args.output / "visibility.json").write_text(json.dumps(report, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for (polar, pitch), z in side.items():
        if pitch == 0.0 or polar == 96.0:
            axes[0].hist(
                z,
                bins=np.arange(-3.2, 1.6, 0.05),
                histtype="step",
                label=f"{polar:.0f} deg, pitch {pitch:+.0f}",
            )
    axes[0].axhspan(0, 0, color="k")
    axes[0].axvspan(-0.5, 0.5, color="tab:green", alpha=0.1, label="height band")
    axes[0].set(
        xlabel="wall point height relative to body (m)",
        ylabel="points",
        title=f"Wall points seen at {args.distance} m",
    )
    axes[0].legend(fontsize=8)
    edges = occlusion["polar_bin_edges_deg"]
    axes[1].bar(edges[:-1], occlusion["blocked_fraction"], width=5.5, align="edge")
    axes[1].set(
        xlabel="polar angle from sensor axis (deg)",
        ylim=(0, 1.05),
        ylabel="fraction blocked by own airframe",
        title="L2 self-occlusion",
    )
    fig.tight_layout()
    fig.savefig(args.output / "visibility.png", dpi=120)

    print(
        f"Wall face {args.distance} m ahead, body at {args.altitude} m, "
        f"band +/-0.5 m about body origin; sensor at "
        f"{cases[0]['sensor_z_in_body_m']:+.3f} m in body z"
    )
    print(
        f"{'polar':>5} {'pitch':>6} {'fwd rate':>8} {'d0 (m)':>7} "
        f"{'pts/scan':>8} {'band seen (m)':>16}"
    )
    for c in cases:
        seen = (
            f"{c['wall_band_bottom_seen_m']:+.2f}..{c['wall_band_top_seen_m']:+.2f}"
            if c["wall_band_top_seen_m"] is not None
            else "none"
        )
        print(
            f"{c['max_polar_deg']:5.0f} {c['pitch_deg']:+6.0f} "
            f"{c['forward_return_rate']:8.2f} {c['sector0_distance_m'] or np.nan:7.2f} "
            f"{c['forward_points_per_scan']:8.1f} {seen:>16}"
        )
    print("Self-occlusion (fraction of rays hitting own exterior geoms) by polar bin:")
    for lo, f in zip(edges[:-1], occlusion["blocked_fraction"], strict=True):
        print(f"  {lo:3d}-{lo + 6:3d} deg: {f:.2f}")
    print("Blocking geoms:", occlusion["blocking_geoms"])
    print(f"Outputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
