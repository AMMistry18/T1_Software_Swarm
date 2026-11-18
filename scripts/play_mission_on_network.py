#!/usr/bin/env python3
"""Replay a mission-planning result inside the packet/network simulator.

Examples:
  # Baseline plan → live network window
  python scripts/play_mission_on_network.py \\
    --scenario one_mapper_small --fallback search --animate \\
    --output experiments/results/network_play_search

  # RL checkpoint → headless packet logs + PNG
  python scripts/play_mission_on_network.py \\
    --scenario one_mapper_small \\
    --checkpoint experiments/checkpoints/incoming/best_model.zip \\
    --output experiments/results/network_play_rl
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tar_swarm.mission_planning.cli import main  # noqa: E402


if __name__ == "__main__":
    # Reuse the CLI entry so flags stay in one place.
    argv = ["network-play", *sys.argv[1:]]
    main(argv)
