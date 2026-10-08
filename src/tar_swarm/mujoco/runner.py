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
from tar_swarm.control.cascaded import CascadedController
from tar_swarm.control.mission import WaypointMission
from tar_swarm.evaluation.logging import sample, save_run
from tar_swarm.mujoco.flight import FlightSimulation
from tar_swarm.mujoco.renderer import configure_viewer, draw_overlays, save_snapshot


def run_scenario(
    config: dict[str, Any],
    *,
    headless: bool = True,
    output: Path | None = None,
    plots: bool = True,
    snapshot: bool = False,
) -> dict[str, Any]:
    validate_config(config)
    if output and output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    simulation = FlightSimulation(config)
    settings = config["simulation"]
    dt = settings["physics_timestep_s"]
    clocks = {
        key: interval_steps(settings[f"{key}_interval_s"], dt)
        for key in ("controller", "mission", "visualization", "log")
    }
    steps = interval_steps(settings["duration_s"], dt)
    controllers = {
        a["id"]: CascadedController(
            config["vehicles"][a["vehicle"]], settings["gravity_m_s2"]
        )
        for a in config["mission"]["agents"]
    }
    missions = {a["id"]: WaypointMission(a) for a in config["mission"]["agents"]}
    targets = {name: mission.sample(0) for name, mission in missions.items()}
    trails: dict[str, list[np.ndarray]] = {name: [] for name in missions}
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
                    command = controller.update(simulation.state(name), targets[name])
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
                            targets[name],
                            outputs[name],
                            int(simulation.data.ncon),
                        )
                    )
            if viewer and tick % clocks["visualization"] == 0:
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    draw_overlays(viewer.user_scn, simulation, config, targets, trails)
                viewer.sync()
            if tick == steps:
                completed = True
                break
            simulation.step()
            if viewer:
                delay = start + simulation.data.time - time.monotonic()
                if delay > 0:
                    time.sleep(min(delay, dt))
    report = None
    if output:
        report = save_run(output, rows, config, simulation.xml, control_rows, completed)
        if plots:
            from tar_swarm.evaluation.plots import plot_run

            plot_run(output)
        if snapshot:
            save_snapshot(output / "scene.png", simulation, config, targets, trails)
    return {
        "simulation": simulation,
        "rows": rows,
        "controls": control_rows,
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
        "--snapshot",
        action="store_true",
        help="Render final scene to PNG; requires OpenGL",
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
    )
    print(json.dumps(result["metrics"], indent=2))
    print(f"Outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
