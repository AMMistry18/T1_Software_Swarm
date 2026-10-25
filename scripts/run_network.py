"""Run the network demo directly from a checkout, without installing MuJoCo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tar_swarm.network.runner import main  # noqa: E402

"""Calls main function of runner.py"""
if __name__ == "__main__":
    main()
