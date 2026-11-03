"""Plot one avoidance run: trajectory, sector distances, commanded vs limited velocity.

Example:
    python scripts/check_avoidance.py experiments/results/<run>
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tar_swarm.evaluation.avoidance_metrics import run_metrics
from tar_swarm.evaluation.avoidance_plots import plot_avoidance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Results directory of an avoidance run")
    parser.add_argument("--output", type=Path, help="PNG path (default: in the run)")
    args = parser.parse_args()
    path = plot_avoidance(args.run, args.output)
    print(json.dumps(asdict(run_metrics(args.run)), indent=2))
    print(f"Figure: {path.resolve()}")


if __name__ == "__main__":
    main()
