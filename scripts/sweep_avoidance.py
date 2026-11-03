"""AV5 sweep: worlds x sensors x speeds x noise x seeds, with pass criteria.

Writes one run directory per case, sweep.csv (one row per run), summary.json
(criteria and per-group summary). Exits 1 if a criterion fails.

Example (full sweep, 4 processes, GIF clips of collisions and false stops):
    python scripts/sweep_avoidance.py --jobs 4 --record-failures
"""

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tar_swarm.evaluation.avoidance_study import (
    NOISE,
    SEEDS,
    SPEEDS,
    cases,
    criteria,
    run_case,
    summarize,
)
from tar_swarm.evaluation.scenarios import SENSORS, WORLDS
from tar_swarm.mujoco.overlays import load_overlay_settings

NOISE_CHOICES = {"on": (True,), "off": (False,), "both": NOISE}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", nargs="+", choices=WORLDS, default=list(WORLDS))
    parser.add_argument(
        "--sensors", nargs="+", choices=sorted(SENSORS), default=sorted(SENSORS)
    )
    parser.add_argument("--speeds", nargs="+", type=float, default=list(SPEEDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--noise", choices=NOISE_CHOICES, default="both")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel processes")
    parser.add_argument(
        "--record-failures",
        action="store_true",
        help="Save clip.gif (from the saved run) for collisions and false stops",
    )
    parser.add_argument("--overlays", type=Path, default=Path("configs/overlays.yaml"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.output or Path("experiments/results") / datetime.now(
        timezone.utc
    ).strftime("av5_sweep_%Y%m%dT%H%M%S_%fZ")
    if root.exists():
        parser.error(f"Output directory already exists: {root}")
    root.mkdir(parents=True)
    record = load_overlay_settings(args.overlays) if args.record_failures else None
    todo = cases(
        args.worlds, args.sensors, args.speeds, NOISE_CHOICES[args.noise], args.seeds
    )
    # Longest runs (L2, slow) first so parallel workers finish together.
    todo.sort(key=lambda c: (c.sensor != "l2", c.speed_m_s))
    print(f"{len(todo)} runs into {root.resolve()}", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_case, case, root, record): case for case in todo}
        for done, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            flag = "COLLISION" if row["collided"] else ""
            flag += " FALSE STOP" if row["false_stop"] else ""
            print(
                f"[{done}/{len(todo)}] {row['name']}: clearance "
                f"{row['min_clearance_m']:.2f} m, stop x {row['stop_x_m']:.2f} {flag}",
                flush=True,
            )
    rows.sort(key=lambda r: r["name"])
    with (root / "sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    results = criteria(rows)
    (root / "summary.json").write_text(
        json.dumps({"criteria": results, "groups": summarize(rows)}, indent=2)
    )
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status}: {result['criterion']} ({result['runs_checked']} runs)")
        for name in result["failures"]:
            print(f"    {name}")
    print(f"Outputs: {root.resolve()}")
    if not all(r["passed"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
