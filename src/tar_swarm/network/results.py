"""Persist sampled positions, actual graph updates, config, and summary."""

import csv
import json
import platform
from importlib.metadata import version
from pathlib import Path

import yaml

from tar_swarm.metrics.network import connectivity_metrics


def save_results(simulator, output: str | Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = simulator.config  # how the run started (hand edits aren't in it)
    # Every drone that was ever in the run, including ones removed by hand, so
    # their older rows still get a role.
    roles = simulator.roles
    fields = [
        "time_s",
        "agent_id",
        "role",
        "active",
        "x_m",
        "y_m",
        "z_m",
        "target_x_m",
        "target_y_m",
        "target_z_m",
        "graph_time_s",
        "ground_reachable",
    ]
    with (output / "states.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for frame in simulator.history:
            for name in sorted(frame.positions):
                writer.writerow(
                    [
                        frame.time_s,
                        name,
                        roles[name],
                        int(frame.active[name]),
                        *frame.positions[name],
                        *frame.targets[name],
                        frame.graph.time_s,
                        int(name in frame.reachable),
                    ]
                )
    with (output / "graphs.jsonl").open("w", encoding="utf-8") as stream:
        for frame in simulator.history:
            if frame.graph_updated:
                stream.write(
                    json.dumps(
                        {
                            "time_s": frame.graph.time_s,
                            "nodes": frame.graph.nodes,
                            "active_nodes": sorted(
                                name for name, active in frame.active.items() if active
                            ),
                            "edges": frame.graph.edges,
                            "ground_reachable": sorted(frame.reachable),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    summary = {
        "backend": "speed_limited_kinematics",
        "link_model": "toy_two_way_3d_range_per_role",
        "seed": config["seed"],
        # Reached the config's duration_s (the live window may run past it).
        "completed": simulator.tick >= simulator.total_steps,
        # True if anything was changed by hand during the run (drones added,
        # removed, moved, or given goals; ranges changed). Then
        # resolved_config.yaml describes only how the run started.
        "edited_by_hand": simulator.edited,
        # The ranges in use when the run ended (sliders may have changed them).
        "radio_range_m_at_end": dict(getattr(simulator.link_model, "range_m", {})),
        "end_time_s": simulator.time_s,
        "dt_s": simulator.dt_s,
        "graph_interval_s": config["graph_interval_s"],
        "sampling_convention": (
            "Reachability held between graph updates; sample k describes "
            "[t[k], t[k+1]); terminal sample has zero weight. "
            "Outages are half-open intervals clipped to the run."
        ),
        "mappers": connectivity_metrics(
            simulator.history, [name for name in roles if roles[name] == "mapper"]
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    environment = {"python": platform.python_version()}
    environment.update(
        {name: version(name) for name in ("numpy", "matplotlib", "pyyaml")}
    )
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
