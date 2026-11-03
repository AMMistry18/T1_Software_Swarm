"""Run one AV4 avoidance scenario: a single drone flies a straight lane into a world.

Example (relay with TF-Luna into the wall at 2 m/s peak, viewer with overlays):
    python scripts/run_avoidance.py --world avoid_wall --sensor rangefinder
        --viewer --overlay
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tar_swarm.evaluation.scenarios import (
    SENSORS,
    WORLDS,
    avoidance_config,
    ground_edge_margin,
)
from tar_swarm.mujoco.overlays import load_overlay_settings
from tar_swarm.mujoco.runner import run_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", choices=WORLDS, required=True)
    parser.add_argument("--sensor", choices=sorted(SENSORS), required=True)
    parser.add_argument("--speed", type=float, default=2.0, help="Peak speed, m/s")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-noise", action="store_true", help="Noise and dropout off")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--record", type=Path, help="Save a GIF clip with sensor overlays (headless)"
    )
    args = parser.parse_args()
    config = avoidance_config(
        args.world, args.sensor, args.speed, seed=args.seed, noise=not args.no_noise
    )
    output = args.output or Path("experiments/results") / datetime.now(
        timezone.utc
    ).strftime(f"{args.world}_{args.sensor}_%Y%m%dT%H%M%S_%fZ")
    if output.exists():
        parser.error(f"Output directory already exists: {output}")
    overlay = (
        load_overlay_settings(Path("configs/overlays.yaml"))
        if args.overlay or args.record
        else None
    )
    result = run_scenario(
        config,
        headless=not args.viewer,
        output=output,
        plots=False,
        overlay=overlay,
        record=args.record,
    )
    agent = config["mission"]["agents"][0]["id"]
    rows = [r for r in result["rows"] if r["vehicle_id"] == agent]
    positions = np.array([[r[f"position_{a}_m"] for a in "xyz"] for r in rows])
    speeds = np.linalg.norm(
        [[r[f"velocity_{a}_m_s"] for a in "xyz"] for r in rows], axis=1
    )
    statuses = [a["status"] for a in result["avoidance"]]
    summary = {
        "world": args.world,
        "sensor": args.sensor,
        "peak_speed_request_m_s": args.speed,
        "seed": args.seed,
        "noise": not args.no_noise,
        "completed": result["completed"],
        "max_x_m": float(positions[:, 0].max()),
        "peak_speed_m_s": float(speeds.max()),
        "contacts_after_takeoff": int(max(r["contacts"] for r in rows[50:])),
        "status_counts": {s: statuses.count(s) for s in sorted(set(statuses))},
        "ground_edge_margin_m": ground_edge_margin(config, positions),
    }
    (output / "scenario.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Outputs: {output.resolve()}")
    if summary["ground_edge_margin_m"] < 0:
        print("FAIL: rays could have been lost off the ground edge", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
