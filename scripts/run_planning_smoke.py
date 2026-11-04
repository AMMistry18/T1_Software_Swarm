#!/usr/bin/env python3
"""Convenience wrapper for the mission-planning end-to-end smoke run."""

from tar_swarm.mission_planning.cli import main

if __name__ == "__main__":
    main(["smoke", "--output", "experiments/results/plan_smoke", "--timesteps", "512"])
