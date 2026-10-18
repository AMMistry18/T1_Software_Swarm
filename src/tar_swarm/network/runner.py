"""Command-line entry point for the standalone network foundation."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from tar_swarm.network.config import load_config
from tar_swarm.network.results import save_results
from tar_swarm.network.simulator import Simulator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/network.yaml"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--headless", action="store_true", help="Run without a GUI (default)"
    )
    mode.add_argument("--animate", action="store_true", help="Show top-down animation")
    parser.add_argument("--output", type=Path, help="New output directory")
    args = parser.parse_args(argv)
    output = args.output or Path("experiments/results") / (
        "network_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    if output.exists():
        parser.error(f"Output directory already exists: {output}")
    simulator = Simulator(load_config(args.config))
    # Plot imports occur after selecting the backend, including in headless mode.
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/tar-swarm-matplotlib")))
    if not args.animate:
        import matplotlib

        matplotlib.use("Agg")
        simulator.run()
    else:
        from tar_swarm.network.visualization import animate

        animate(simulator)
    summary = save_results(simulator, output)
    from tar_swarm.network.visualization import save_plot

    save_plot(simulator, output / "trajectory_connectivity.png")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
