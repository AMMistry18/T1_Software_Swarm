"""Play back a saved run in the MuJoCo viewer with sensor overlays.

Drones are placed from states.csv; beams, L2 points, sector rings and velocity
arrows come from the run's measurement files. Nothing is re-simulated.
With --record the run is rendered offscreen to a GIF instead of a window.
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from tar_swarm.mujoco.overlays import draw_sensor_overlays, load_overlay_settings
from tar_swarm.mujoco.recording import record_replay
from tar_swarm.mujoco.renderer import configure_viewer
from tar_swarm.mujoco.replay import load_run, replay_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Results directory of a saved run")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed")
    parser.add_argument("--start", type=float, default=0.0, help="Start time, s")
    parser.add_argument("--overlays", type=Path, default=Path("configs/overlays.yaml"))
    parser.add_argument("--end", type=float, help="End time for --record, s")
    parser.add_argument("--record", type=Path, help="Render to this GIF, no window")
    args = parser.parse_args()
    settings = load_overlay_settings(args.overlays)
    if args.record:
        path = record_replay(
            args.run, args.record, settings, start_s=args.start, end_s=args.end
        )
        print(f"Clip: {path.resolve() if path else 'no frames'}")
        return
    run = load_run(args.run)
    data = mujoco.MjData(run.model)
    print(f"Replaying {args.run} ({run.end_time_s:.1f} s); close the window to stop.")
    with mujoco.viewer.launch_passive(run.model, data) as viewer:
        configure_viewer(viewer)
        wall_start = time.monotonic()
        while viewer.is_running():
            sim_time = min(
                args.start + (time.monotonic() - wall_start) * args.speed,
                run.end_time_s,
            )
            frame = replay_frame(run, sim_time)
            with viewer.lock():
                data.qpos[:] = frame.qpos
                data.time = sim_time
                mujoco.mj_forward(run.model, data)
                viewer.user_scn.ngeom = 0
                draw_sensor_overlays(
                    viewer.user_scn, settings, frame.beams, frame.l2, frame.arrows
                )
            viewer.sync()
            time.sleep(1 / 30)


if __name__ == "__main__":
    main()
