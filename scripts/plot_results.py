"""Regenerate analysis figures from a saved Step 0 run."""

import argparse

from tar_swarm.evaluation.plots import plot_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    arguments = parser.parse_args()
    for path in plot_run(arguments.directory):
        print(path)
