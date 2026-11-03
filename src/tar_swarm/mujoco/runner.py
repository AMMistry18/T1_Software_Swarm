"""Fixed-clock experiment execution, independent of viewer frame rate."""

import argparse
import json
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tar_swarm.common.config import interval_steps, load_config, validate_config
from tar_swarm.control.avoidance import (
    ReferenceShaper,
    SectorSource,
    load_avoidance_params,
    rangefinder_sector,
)
from tar_swarm.control.cascaded import CascadedController
from tar_swarm.control.mission import WaypointMission
from tar_swarm.evaluation.logging import sample, save_run
from tar_swarm.mujoco.flight import FlightSimulation
from tar_swarm.mujoco.overlays import (
    Beam,
    L2View,
    VelocityArrow,
    draw_sensor_overlays,
    load_overlay_settings,
    needs_l2,
)
from tar_swarm.mujoco.recording import GifRecorder
from tar_swarm.mujoco.renderer import configure_viewer, draw_overlays, save_snapshot
from tar_swarm.perception.obstacles import (
    extract_obstacles,
    heading_rotation,
    load_obstacle_params,
    obstacles_enabled,
    sector_centers,
)
from tar_swarm.sensors.lidar import (
    apply_noise,
    load_lidar_params,
    scan_directions,
    to_points,
)
from tar_swarm.sensors.rangefinder import (
    apply_noise as apply_range_noise,
)
from tar_swarm.sensors.rangefinder import (
    beam_directions,
    load_rangefinder_params,
    nearest_hit,
)
from tar_swarm.sensors.rng import sensor_rngs


def run_scenario(
    config: dict[str, Any],
    *,
    headless: bool = True,
    output: Path | None = None,
    plots: bool = True,
    snapshot: bool = False,
    overlay: dict[str, Any] | None = None,
    record: Path | None = None,
) -> dict[str, Any]:
    validate_config(config)
    if output and output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    simulation = FlightSimulation(config)
    recorder = GifRecorder(simulation.model, record) if record else None
    settings = config["simulation"]
    dt = settings["physics_timestep_s"]
    clocks = {
        key: interval_steps(settings[f"{key}_interval_s"], dt)
        for key in ("controller", "mission", "visualization", "log")
    }
    steps = interval_steps(settings["duration_s"], dt)
    record_steps = interval_steps(1 / recorder.fps, dt) if recorder else 0
    followed = config["mission"]["agents"][0]["id"]
    controllers = {
        a["id"]: CascadedController(
            config["vehicles"][a["vehicle"]], settings["gravity_m_s2"]
        )
        for a in config["mission"]["agents"]
    }
    missions = {a["id"]: WaypointMission(a) for a in config["mission"]["agents"]}
    targets = {name: mission.sample(0) for name, mission in missions.items()}
    trails: dict[str, list[np.ndarray]] = {name: [] for name in missions}
    lidars = {
        a["id"]: params
        for a in config["mission"]["agents"]
        if (params := load_lidar_params(config.get("sensors"), a["vehicle"]))
    }
    lidar_steps = {
        name: interval_steps(1 / params.rate_hz, dt) for name, params in lidars.items()
    }
    rngs = sensor_rngs(settings["seed"])
    lidar_rng = rngs["lidar"]
    lidar_scans: list[dict[str, Any]] = []
    lidar_truth: list[dict[str, Any]] = []
    rangefinders = {
        a["id"]: params
        for a in config["mission"]["agents"]
        if (params := load_rangefinder_params(config.get("sensors"), a["vehicle"]))
    }
    rangefinder_steps = {
        name: interval_steps(1 / params.rate_hz, dt)
        for name, params in rangefinders.items()
    }
    beam_rays = {name: beam_directions(p) for name, p in rangefinders.items()}
    range_readings: list[dict[str, Any]] = []
    range_truth: list[dict[str, Any]] = []
    beams: dict[str, Beam] = {}
    shapers = {
        a["id"]: ReferenceShaper(params)
        for a in config["mission"]["agents"]
        if (params := load_avoidance_params(config.get("avoidance"), a["vehicle"]))
    }
    logged_l2 = {
        a["id"]
        for a in config["mission"]["agents"]
        if a["id"] in lidars
        and obstacles_enabled(config.get("perception"), a["vehicle"])
    }
    obstacle_params = {
        a["id"]: params
        for a in config["mission"]["agents"]
        if a["id"] in lidars
        and (
            needs_l2(overlay)
            or a["id"] in logged_l2
            or (a["id"] in shapers and "l2" in shapers[a["id"]].params.sources)
        )
        and (params := load_obstacle_params(config.get("perception"), a["vehicle"]))
    }
    mounts = {}
    for name in obstacle_params:
        body = simulation.state(name)
        site_position, site_rotation = simulation.site_pose(
            f"{name}_{lidars[name].site}"
        )
        # Fixed mount calibration: sensor pose in the body frame.
        mounts[name] = (
            body.rotation.T @ site_rotation,
            body.rotation.T @ (site_position - body.position),
        )
    l2_views: dict[str, L2View] = {}
    l2_records: list[dict[str, Any]] = []
    # Latest measurement-side inputs for avoidance: (reading time, distances).
    latest_l2: dict[str, tuple[float, np.ndarray]] = {}
    latest_range: dict[str, tuple[float, float]] = {}
    last_valid_range: dict[str, float] = {}
    references = {}
    avoidance_rows: list[dict[str, Any]] = []
    arrows: dict[str, VelocityArrow] = {}

    def avoidance_sources(name: str, now: float) -> list[SectorSource]:
        sources = []
        if "l2" in shapers[name].params.sources:
            count = obstacle_params[name].num_sectors
            seen, distance = latest_l2.get(name, (-np.inf, np.full(count, np.nan)))
            sources.append(
                SectorSource(
                    sector_centers(count),
                    distance,
                    now - seen,
                    1 / lidars[name].rate_hz,
                )
            )
        if "rangefinder" in shapers[name].params.sources:
            seen, distance = latest_range.get(name, (-np.inf, np.nan))
            sources.append(
                SectorSource(
                    np.zeros(1),
                    np.array([distance]),
                    now - seen,
                    1 / rangefinders[name].rate_hz,
                )
            )
        return sources

    rows, control_rows = [], []
    outputs = {}
    context = nullcontext(None)
    if not headless:
        import mujoco.viewer

        context = mujoco.viewer.launch_passive(simulation.model, simulation.data)
    start = time.monotonic()
    completed = False
    with context as viewer:
        if viewer:
            configure_viewer(viewer)
        for tick in range(steps + 1):
            if viewer and not viewer.is_running():
                break
            sim_time = float(simulation.data.time)
            if tick % clocks["mission"] == 0:
                targets = {
                    name: mission.sample(sim_time) for name, mission in missions.items()
                }
            if tick % clocks["controller"] == 0:
                for name, controller in controllers.items():
                    state = simulation.state(name)
                    reference = targets[name]
                    if name in shapers:
                        reference, record = shapers[name].update(
                            reference,
                            heading_rotation(state.rotation),
                            avoidance_sources(name, sim_time),
                            clocks["controller"] * dt,
                        )
                        avoidance_rows.append(
                            {
                                "time_s": sim_time,
                                "vehicle_id": name,
                                "commanded_velocity": record.commanded_velocity,
                                "limited_velocity": record.limited_velocity,
                                "speed_limit_m_s": record.speed_limit_m_s,
                                "status": record.status,
                                "reference_position": reference.position.copy(),
                            }
                        )
                        if overlay:
                            arrows[name] = VelocityArrow(
                                state.position,
                                record.commanded_velocity,
                                record.limited_velocity,
                                record.status,
                            )
                    references[name] = reference
                    command = controller.update(state, reference)
                    outputs[name] = command
                    simulation.command(name, command.thrusts)
                    control_rows.append(
                        {
                            "time_s": sim_time,
                            "vehicle_id": name,
                            **{
                                f"motor_{i}_n": value
                                for i, value in enumerate(command.thrusts)
                            },
                            "motor_saturated": int(command.motor_saturated),
                            "acceleration_limited": int(command.acceleration_limited),
                        }
                    )
            if tick % clocks["log"] == 0 or tick == steps:
                for name in missions:
                    state = simulation.state(name)
                    trails[name].append(state.position.copy())
                    rows.append(
                        sample(
                            sim_time,
                            name,
                            state,
                            references.get(name, targets[name]),
                            outputs[name],
                            int(simulation.data.ncon),
                        )
                    )
            for name, params in lidars.items():
                if tick % lidar_steps[name] == 0:
                    directions = scan_directions(params, lidar_rng)
                    truth, geom_id, normal = simulation.cast_rays(
                        name, f"{name}_{params.site}", directions, params.max_range_m
                    )
                    measured = apply_noise(truth, params, lidar_rng)
                    valid = np.isfinite(measured)
                    points = to_points(directions, measured)
                    lidar_scans.append(
                        {"time_s": sim_time, "vehicle_id": name, "points": points}
                    )
                    if name in obstacle_params:
                        body = simulation.state(name)
                        found = extract_obstacles(
                            points,
                            *mounts[name],
                            body.rotation,
                            obstacle_params[name],
                            rngs["l2_obstacles"],
                        )
                        latest_l2[name] = (sim_time, found.sector_distance_m)
                        if name in logged_l2:
                            ground = found.ground
                            l2_records.append(
                                {
                                    "time_s": sim_time,
                                    "vehicle_id": name,
                                    "scan_index": len(lidar_scans) - 1,
                                    "sector_distance_m": found.sector_distance_m,
                                    "sector_count": found.sector_count,
                                    "ground_normal": ground.normal if ground else None,
                                    "ground_offset": ground.offset
                                    if ground
                                    else np.nan,
                                    "ground_mask": found.ground_mask,
                                }
                            )
                        l2_views[name] = L2View(
                            body.position,
                            heading_rotation(body.rotation),
                            found.points_level,
                            found.ground_mask,
                            found.sector_distance_m,
                        )
                    # Oracles for tests and checks only; estimators must not read.
                    lidar_truth.append(
                        {
                            "true_distance_m": truth[valid],
                            "geom_id": geom_id[valid],
                            "normal_world": normal[valid],
                        }
                    )
            for name, params in rangefinders.items():
                if tick % rangefinder_steps[name] == 0:
                    site = f"{name}_{params.site}"
                    truth, geom_id, _ = simulation.cast_rays(
                        name, site, beam_rays[name], params.max_range_m
                    )
                    hit = nearest_hit(truth)
                    true_distance = float(truth[hit]) if hit >= 0 else -1.0
                    measured = float(
                        apply_range_noise(
                            np.array([true_distance]), params, rngs["rangefinder"]
                        )[0]
                    )
                    range_readings.append(
                        {"time_s": sim_time, "vehicle_id": name, "distance_m": measured}
                    )
                    if name in shapers:
                        latest_range[name] = (
                            sim_time,
                            rangefinder_sector(
                                measured,
                                last_valid_range.get(name),
                                shapers[name].params.rangefinder_latch_m,
                            ),
                        )
                        if np.isfinite(measured):
                            last_valid_range[name] = measured
                    origin, rotation = simulation.site_pose(site)
                    # Oracles for tests and checks only; avoidance must not read.
                    range_truth.append(
                        {
                            "true_distance_m": true_distance,
                            "geom_id": int(geom_id[hit]) if hit >= 0 else -1,
                            "site_position_world": origin,
                            "site_rotation_world": rotation,
                        }
                    )
                    if overlay:
                        beams[name] = Beam(
                            origin,
                            rotation[:, 2],
                            measured,
                            params.effective_max_range_m,
                        )
            if viewer and tick % clocks["visualization"] == 0:
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    draw_overlays(viewer.user_scn, simulation, config, targets, trails)
                    if overlay:
                        draw_sensor_overlays(
                            viewer.user_scn, overlay, beams, l2_views, arrows
                        )
                viewer.sync()
            if recorder and tick % record_steps == 0:

                def draw(scene: Any) -> None:
                    draw_overlays(scene, simulation, config, targets, trails)
                    if overlay:
                        draw_sensor_overlays(scene, overlay, beams, l2_views, arrows)

                recorder.capture(
                    simulation.data, simulation.state(followed).position, draw
                )
            if tick == steps:
                completed = True
                break
            simulation.step()
            if viewer:
                delay = start + simulation.data.time - time.monotonic()
                if delay > 0:
                    time.sleep(min(delay, dt))
    if recorder:
        recorder.close()
    report = None
    if output:
        report = save_run(
            output,
            rows,
            config,
            simulation.xml,
            control_rows,
            completed,
            (lidar_scans, lidar_truth) if lidars else None,
            (range_readings, range_truth) if rangefinders else None,
            l2_records or None,
            avoidance_rows or None,
        )
        if plots:
            from tar_swarm.evaluation.plots import plot_run

            plot_run(output)
        if snapshot:
            save_snapshot(output / "scene.png", simulation, config, targets, trails)
    return {
        "simulation": simulation,
        "rows": rows,
        "controls": control_rows,
        "lidar": lidar_scans,
        "lidar_truth": lidar_truth,
        "rangefinder": range_readings,
        "rangefinder_truth": range_truth,
        "l2_obstacles": l2_records,
        "avoidance": avoidance_rows,
        "metrics": report,
        "completed": completed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TAR Step 0 physical quadrotor mission"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/simulation.yaml"))
    parser.add_argument("--world", type=Path, help="Override world YAML")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--headless", action="store_true")
    mode.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--duration", type=float, help="Override duration, in simulation seconds"
    )
    parser.add_argument(
        "--output", type=Path, help="New directory (existing directories are protected)"
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Draw sensor overlays in the viewer (settings: overlays.yaml)",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Render final scene to PNG; requires OpenGL",
    )
    parser.add_argument(
        "--record",
        type=Path,
        help="Save a GIF clip (offscreen, works headless); add --overlay for sensors",
    )
    args = parser.parse_args()
    config = load_config(args.config, args.world)
    if args.duration is not None:
        config["simulation"]["duration_s"] = args.duration
    headless = args.headless or (config["simulation"]["headless"] and not args.viewer)
    output = args.output or Path("experiments/results") / datetime.now(
        timezone.utc
    ).strftime("step0_%Y%m%dT%H%M%S_%fZ")
    if output.exists():
        parser.error(f"Output directory already exists: {output}")
    result = run_scenario(
        config,
        headless=headless,
        output=output,
        plots=not args.no_plots,
        snapshot=args.snapshot,
        overlay=load_overlay_settings(args.config.parent / "overlays.yaml")
        if args.overlay
        else None,
        record=args.record,
    )
    print(json.dumps(result["metrics"], indent=2))
    print(f"Outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
