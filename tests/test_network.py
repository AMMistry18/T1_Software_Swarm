"""Focused checks for the standalone kinematic/radius-graph foundation."""

import csv
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from tar_swarm.metrics.network import connectivity_metrics
from tar_swarm.network.config import load_config, validate_config
from tar_swarm.network.connectivity import reachable_from
from tar_swarm.network.graph import build_graph
from tar_swarm.network.radio import RadiusLink
from tar_swarm.network.simulator import Simulator

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return load_config(ROOT / "configs/network.yaml")


def test_direct_multihop_disconnected_and_inactive():
    positions = {
        "ground": np.array([0.0, 0.0, 0.0]),
        "relay": np.array([3.0, 0.0, 0.0]),
        "mapper": np.array([6.0, 0.0, 0.0]),
        "far": np.array([0.0, 0.0, 10.0]),
        "inactive": np.array([1.0, 0.0, 0.0]),
    }
    active = dict.fromkeys(positions, True)
    active["inactive"] = False
    graph = build_graph(positions, active, 0.0, RadiusLink(3.0))
    assert graph.edges == (("ground", "relay"), ("mapper", "relay"))
    assert reachable_from(graph, "ground") == {"ground", "relay", "mapper"}
    assert ("ground", "mapper") not in graph.edges
    active["relay"] = False
    assert reachable_from(
        build_graph(positions, active, 1.0, RadiusLink(3.0)), "ground"
    ) == {"ground"}


def test_3d_radius_boundary():
    link = RadiusLink(5)
    origin = np.zeros(3)
    assert link.usable(origin, np.array([3, 0, 4]))
    assert not link.usable(origin, np.array([3, 0, 4.001]))


def test_waypoints_no_overshoot_final_hold_and_stationary_ground(config):
    config["dt_s"] = 1.0
    config["graph_interval_s"] = 1.0
    config["duration_s"] = 6.0
    mapper = config["agents"][1]
    mapper.update(
        position_m=[0, 0, 10],
        speed_m_s=2.0,
        waypoints_m=[[3, 0, 10], [3, 2, 10]],
    )
    sim = Simulator(config)
    frames = sim.run()
    np.testing.assert_array_equal(
        [f.positions["mapper_1"] for f in frames],
        [
            [0, 0, 10],
            [2, 0, 10],
            [3, 0, 10],
            [3, 2, 10],
            [3, 2, 10],
            [3, 2, 10],
            [3, 2, 10],
        ],
    )
    for first, second in zip(frames, frames[1:]):
        assert (
            np.linalg.norm(second.positions["mapper_1"] - first.positions["mapper_1"])
            <= 2.0
        )
    for frame in frames:
        np.testing.assert_array_equal(frame.positions["ground"], [0, 0, 0])
    with pytest.raises(StopIteration):
        sim.step()


def test_graph_clock_holds_between_updates(config):
    config["duration_s"] = 1.0
    sim = Simulator(config)
    initial_graph = sim.graph
    for _ in range(4):
        frame = sim.step()
        assert frame.graph is initial_graph
        assert not frame.graph_updated
    frame = sim.step()
    assert frame.graph_updated
    assert frame.graph.time_s == 0.5


def test_default_outage_reconnection_and_reproducibility(config):
    first, second = Simulator(config), Simulator(config)
    first.run()
    second.run()
    assert len(first.history) == 241
    for a, b in zip(first.history, second.history):
        assert a.time_s == b.time_s
        assert a.graph == b.graph
        assert a.reachable == b.reachable
        for name in a.positions:
            np.testing.assert_array_equal(a.positions[name], b.positions[name])
    metrics = connectivity_metrics(first.history, ["mapper_1"])["mapper_1"]
    assert metrics["connected_time_fraction"] == pytest.approx(0.375)
    assert metrics["outage_intervals_s"] == [[2.5, 17.5]]
    assert "mapper_1" in first.history[0].reachable
    assert "mapper_1" in first.history[-1].reachable


def test_multiple_mappers_zero_relays_and_policy_extension(config):
    config["agents"] = config["agents"][:2]
    extra = deepcopy(config["agents"][1])
    extra["id"] = "mapper_2"
    config["agents"].append(extra)
    config["duration_s"] = 1.0

    class HoldPolicy:
        def target(self, agent_id, time_s, positions):
            return positions[agent_id]

    sim = Simulator(config, policy=HoldPolicy())
    sim.run()
    metrics = connectivity_metrics(sim.history, ["mapper_1", "mapper_2"])
    for name in metrics:
        np.testing.assert_array_equal(sim.positions[name], [70, 0, 10])
        assert metrics[name]["outage_intervals_s"] == [[0.0, 1.0]]
        assert metrics[name]["connected_time_fraction"] == 0.0


def test_terminal_reconnection_has_zero_metric_weight(config):
    # Reachable only at the terminal graph update: the entire run is an outage.
    config.update(dt_s=1.0, duration_s=1.0, graph_interval_s=1.0)
    config["agents"] = config["agents"][:2]
    config["agents"][1].update(
        position_m=[70, 0, 10],
        speed_m_s=70,
        waypoints_m=[[0, 0, 10]],
    )
    sim = Simulator(config)
    sim.run()
    assert "mapper_1" in sim.history[-1].reachable
    metrics = connectivity_metrics(sim.history, ["mapper_1"])["mapper_1"]
    assert metrics["outage_intervals_s"] == [[0.0, 1.0]]
    assert metrics["connected_time_fraction"] == 0.0


@pytest.mark.parametrize(
    "key,value",
    [
        ("dt_s", 0),
        ("dt_s", float("nan")),
        ("duration_s", 0.15),
        ("graph_interval_s", 0.15),
        ("communication_radius_m", -1),
        ("plot_bounds_m", [[10, 0], [0, 10]]),
    ],
)
def test_invalid_settings(config, key, value):
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_invalid_agents(config):
    invalid = deepcopy(config)
    invalid["agents"][1]["id"] = "ground"
    with pytest.raises(ValueError):
        validate_config(invalid)
    invalid = deepcopy(config)
    invalid["agents"][1]["waypoints_m"][0][2] = 20
    with pytest.raises(ValueError):
        validate_config(invalid)
    invalid = deepcopy(config)
    invalid["agents"][0]["active"] = False
    with pytest.raises(ValueError):
        validate_config(invalid)


def test_headless_cli_logs_repeatable_without_mujoco(tmp_path):
    # Running the public script also exercises checkout bootstrapping and plotting.
    env = dict(os.environ, MPLCONFIGDIR=str(tmp_path / "mpl"))
    outputs = [tmp_path / "first", tmp_path / "second"]
    for output in outputs:
        subprocess.run(
            [
                sys.executable,
                "scripts/run_network.py",
                "--headless",
                "--output",
                str(output),
            ],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert (output / "trajectory_connectivity.png").read_bytes()[:8] == (
            b"\x89PNG\r\n\x1a\n"
        )
        with (output / "states.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 241 * 4
        graphs = [
            json.loads(line)
            for line in (output / "graphs.jsonl").read_text().splitlines()
        ]
        assert len(graphs) == 49
        assert graphs[0]["time_s"] == 0.0
        assert "mapper_1" in graphs[0]["ground_reachable"]
        assert ["ground", "mapper_1"] not in graphs[0]["edges"]
        assert json.loads((output / "summary.json").read_text())["completed"]
    for name in ("states.csv", "graphs.jsonl", "summary.json", "resolved_config.yaml"):
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()
    rerun = subprocess.run(
        [sys.executable, "scripts/run_network.py", "--output", str(outputs[0])],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert rerun.returncode != 0
    assert "already exists" in rerun.stderr


def test_animation_pause_and_rendering_preserve_clock(config, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation_module
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import KeyEvent

    from tar_swarm.network.visualization import NetworkView, animate

    config["duration_s"] = 0.5
    animated = Simulator(config)
    expected = Simulator(config)
    expected.run()
    callbacks = {}

    class FakeTimer:
        stopped = False

        def stop(self):
            self.stopped = True

    class FakeAnimation:
        def __init__(self, fig, update, init_func, **kwargs):
            self.event_source = FakeTimer()
            callbacks.update(fig=fig, update=update, timer=self.event_source)
            init_func()

    def exercise_window():
        fig, update = callbacks["fig"], callbacks["update"]
        assert animated.time_s == 0
        # Repeated drawing cannot advance time or invoke movement policies.
        view = NetworkView(fig.axes[0], animated)
        view.draw()
        view.draw()
        assert animated.time_s == 0
        fig.canvas.callbacks.process(
            "key_press_event", KeyEvent("key_press_event", fig.canvas, key=" ")
        )
        update(0)
        update(1)
        assert animated.time_s == 0  # paused callbacks do not advance the engine
        fig.canvas.callbacks.process(
            "key_press_event", KeyEvent("key_press_event", fig.canvas, key=" ")
        )
        for tick in range(animated.total_steps):
            update(tick)
        assert animated.done
        assert callbacks["timer"].stopped
        fig.canvas.draw()
        plt.close(fig)

    monkeypatch.setattr(animation_module, "FuncAnimation", FakeAnimation)
    monkeypatch.setattr(plt, "show", exercise_window)
    animate(animated)
    for first, second in zip(animated.history, expected.history):
        assert first.time_s == second.time_s
        assert first.graph == second.graph
        for name in first.positions:
            np.testing.assert_array_equal(first.positions[name], second.positions[name])
