"""Focused checks for the standalone kinematic/radius-graph foundation."""

import csv
import json
import math
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
from tar_swarm.network.radio import RangeLink
from tar_swarm.network.simulator import Simulator
from tar_swarm.strategies.fixed_relay import (
    chain_slots,
    hop_limits,
    match_relays_to_slots,
    split_relays,
)

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("ground_station", "mapper", "relay")
# In the small demo the drones fly 10 m above the base, so a hop from the base
# also climbs 10 m. With 30 m ranges, Pythagoras leaves sqrt(30^2 - 10^2) =
# 28.3 m of it on the map (the plan works on the flat map).
BASE_HOP_M = math.sqrt(30**2 - 10**2)


def same_range(range_m):
    """A RangeLink where every kind of drone has the same radio range."""
    return RangeLink(dict.fromkeys(ROLES, range_m))


# Robert's original 24 s demo scene: a 135 m strip with 30 m radio ranges. Many tests
# check its exact, hand-verified numbers (for example the [2.5, 17.5) s mapper
# outage), so it lives here with the tests instead of in configs/. The default 1 km
# scene (configs/network.yaml) has its own test.
SMALL_DEMO = {
    "seed": 7,
    "dt_s": 0.1,
    "duration_s": 24.0,
    "graph_interval_s": 0.5,
    "radio_range_m": {"ground_station": 30.0, "relay": 30.0, "mapper": 30.0},
    "plot_bounds_m": [[-10, 125], [-25, 30]],
    "grid_step_m": 10.0,
    "new_relay_altitude_m": 10.0,
    "new_relay_speed_m_s": 1.0,
    "new_mapper_altitude_m": 10.0,
    "new_mapper_speed_m_s": 4.0,
    "obstacles_m": [],
    "agents": [
        {"id": "ground", "role": "ground_station", "position_m": [0, 0, 0]},
        {
            "id": "mapper_1",
            "role": "mapper",
            "position_m": [70, 0, 10],
            "speed_m_s": 4.0,
            "waypoints_m": [[110, 0, 10], [70, 0, 10]],
        },
        {
            "id": "relay_1",
            "role": "relay",
            "position_m": [25, 0, 10],
            "speed_m_s": 1.0,
            "waypoints_m": [[25, 8, 10], [25, 0, 10]],
        },
        {
            "id": "relay_2",
            "role": "relay",
            "position_m": [50, 0, 10],
            "speed_m_s": 1.0,
            "waypoints_m": [[50, -8, 10], [50, 0, 10]],
        },
    ],
}


@pytest.fixture
def config():
    # validate_config makes its own copy, so no test can change SMALL_DEMO itself.
    return validate_config(SMALL_DEMO)


def write_small_demo(folder):
    """Save SMALL_DEMO as a YAML file in `folder` (a throwaway test folder) and
    return its path, for tests that run the program from the command line."""
    import yaml

    path = folder / "small_demo.yaml"
    path.write_text(yaml.safe_dump(SMALL_DEMO, sort_keys=False))
    return path


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
    roles = {"ground": "ground_station", "mapper": "mapper"}
    roles.update(relay="relay", far="relay", inactive="relay")
    graph = build_graph(positions, active, 0.0, same_range(3.0), roles)
    assert graph.edges == (("ground", "relay"), ("mapper", "relay"))
    assert reachable_from(graph, "ground") == {"ground", "relay", "mapper"}
    assert ("ground", "mapper") not in graph.edges
    active["relay"] = False
    assert reachable_from(
        build_graph(positions, active, 1.0, same_range(3.0), roles), "ground"
    ) == {"ground"}


def test_3d_radius_boundary():
    link = same_range(5)
    origin = np.zeros(3)
    assert link.usable("relay", "relay", origin, np.array([3, 0, 4]))
    assert not link.usable("relay", "relay", origin, np.array([3, 0, 4.001]))


def test_link_needs_both_ends_in_range():
    """A link works only up to the SMALLER of its two ranges: a base with a
    600 m range can't link with a 300 m network drone 400 m away."""
    link = RangeLink({"ground_station": 600.0, "relay": 300.0, "mapper": 300.0})
    origin = np.zeros(3)
    assert not link.usable("ground_station", "relay", origin, np.array([400, 0, 0]))
    assert link.usable("ground_station", "relay", origin, np.array([300, 0, 0]))
    with pytest.raises(ValueError):
        RangeLink({"relay": 0.0})


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
        ("radio_range_m", {"ground_station": 30, "mapper": 30, "relay": -1}),
        ("radio_range_m", {"relay": 30}),  # every role needs a range
        ("plot_bounds_m", [[10, 0], [0, 10]]),
        ("grid_step_m", 0),
        ("new_relay_altitude_m", float("nan")),
        ("new_relay_speed_m_s", -1),
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
    scene = str(write_small_demo(tmp_path))
    outputs = [tmp_path / "first", tmp_path / "second"]
    for output in outputs:
        subprocess.run(
            [
                sys.executable,
                "scripts/run_network.py",
                "--headless",
                "--config",
                scene,
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
        [
            sys.executable,
            "scripts/run_network.py",
            "--config",
            scene,
            "--output",
            str(outputs[0]),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert rerun.returncode != 0
    assert "already exists" in rerun.stderr


def test_animation_pause_and_rendering_preserve_clock(config):
    """Drawing never moves the clock, pausing stops it, the live window keeps
    running past duration_s, and its steps match a headless run exactly."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import KeyEvent

    from tar_swarm.network.visualization import InteractiveWindow, NetworkView

    config["duration_s"] = 0.5
    animated = Simulator(config)
    expected = Simulator(config)
    expected.run()
    window = InteractiveWindow(animated)
    fig = window.fig
    # A pretend clock that moves 0.05 s each time the window reads it, so every
    # frame covers 0.05 real seconds (0.25 simulated seconds at 5x).
    fake_now = [0.0]

    def fake_clock():
        fake_now[0] += 0.05
        return fake_now[0]

    window.clock = fake_clock
    window.last_frame_s = 0.0
    # Repeated drawing cannot advance time or invoke movement policies.
    view = NetworkView(fig.axes[0], animated)
    view.draw()
    view.draw()
    window.redraw()
    assert animated.time_s == 0

    def press(key):
        """Send a fake key press to the window, as if from a user."""
        fig.canvas.callbacks.process(
            "key_press_event", KeyEvent("key_press_event", fig.canvas, key=key)
        )

    press(" ")  # pause
    window.update()
    window.update()
    assert animated.time_s == 0  # paused frames do not advance the engine
    press(" ")  # resume
    for _ in range(6):  # 6 frames of 0.05 real s at 5x: 1.5 simulated seconds
        window.update()
    # The live window runs past duration_s (0.5 s), so you can keep editing.
    assert animated.time_s == pytest.approx(1.5)
    assert not animated.done
    fig.canvas.draw()
    plt.close(fig)
    # Its first 0.5 s match a headless run step for step.
    assert len(expected.history) == 6
    for first, second in zip(animated.history, expected.history):
        assert first.time_s == second.time_s
        assert first.graph == second.graph
        for name in first.positions:
            np.testing.assert_array_equal(first.positions[name], second.positions[name])


def test_default_scene_is_a_square_km_map():
    """The default config is the 1 km x 1 km map with the base at its corner,
    and its relay chain starts out connected."""
    config = load_config(ROOT / "configs/network.yaml")
    assert config["plot_bounds_m"] == [[0, 1000], [0, 1000]]
    assert config["grid_step_m"] == 100
    sim = Simulator(config)
    np.testing.assert_array_equal(sim.positions["ground"], [0, 0, 0])
    assert "mapper_1" in sim.history[0].reachable


def test_hand_edits_add_move_and_remove(config):
    """Hand edits change the links at once and never move the clock. Names are
    never reused, and a dragged drone holds its new spot instead of its route."""
    sim = Simulator(config)
    sim.remove_drone("relay_2")
    assert "mapper_1" not in sim.history[-1].reachable  # the chain is broken
    name = sim.add_drone("relay", (50, 0))
    assert name == "relay_3"  # not relay_2 again, so saved logs stay unambiguous
    np.testing.assert_array_equal(sim.positions[name], [50, 0, 10])
    assert "mapper_1" in sim.history[-1].reachable  # the new drone fixes it at once
    assert sim.time_s == 0
    sim.move_agent("mapper_1", (60, 5))
    for _ in range(20):
        sim.step()
    np.testing.assert_array_equal(sim.positions["mapper_1"], [60, 5, 10])
    np.testing.assert_array_equal(sim.positions[name], [50, 0, 10])
    with pytest.raises(ValueError):
        sim.remove_drone("mapper_1")  # the run keeps at least one mapper
    with pytest.raises(ValueError):
        sim.move_agent("ground", (5, 5))  # the base station never moves


def test_saved_results_keep_drones_removed_by_hand(config, tmp_path):
    """A drone removed partway through keeps its earlier rows in states.csv, and
    the summary says the run was edited by hand."""
    from tar_swarm.network.results import save_results

    config["duration_s"] = 1.0
    sim = Simulator(config)
    for _ in range(5):
        sim.step()
    sim.remove_drone("relay_2")  # at t = 0.5 s; this replaces the 0.5 s snapshot
    sim.run()
    summary = save_results(sim, tmp_path / "run")
    assert summary["edited_by_hand"]
    with (tmp_path / "run" / "states.csv").open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["agent_id"] == "relay_2"]
    assert [float(row["time_s"]) for row in rows] == pytest.approx(
        [0.0, 0.1, 0.2, 0.3, 0.4]
    )
    assert {row["role"] for row in rows} == {"relay"}


def test_window_mouse_edits_and_range_toggles(config):
    """In the live window, right-click adds and removes a network drone, dragging
    moves one, and a checkbox hides a role's range circles. None of it moves the
    clock."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import MouseEvent

    from tar_swarm.network.visualization import InteractiveWindow

    sim = Simulator(config)
    window = InteractiveWindow(sim)
    window.fig.canvas.draw()  # lay out the window so meters-to-pixels math works
    ax = window.view.ax

    def mouse(kind, x_m, y_m, button=1):
        """Send a fake mouse event at map point (x_m, y_m), as if from a user."""
        x_px, y_px = ax.transData.transform((x_m, y_m))
        event = MouseEvent(kind, window.fig.canvas, x_px, y_px, button=button)
        window.fig.canvas.callbacks.process(kind, event)

    mouse("button_press_event", 40, 20, button=3)  # right-click empty map
    np.testing.assert_allclose(sim.positions["relay_3"], [40, 20, 10])
    mouse("button_press_event", 25, 0)  # grab relay_1 ...
    mouse("motion_notify_event", 30, -10)  # ... drag it ...
    mouse("button_release_event", 30, -10)  # ... and let go
    np.testing.assert_allclose(sim.positions["relay_1"], [30, -10, 10])
    mouse("button_press_event", 40, 20, button=3)  # right-click relay_3
    assert "relay_3" not in sim.positions
    window.on_toggle("network")
    window.view.draw()
    assert not window.view.range_circles["relay_1"].get_visible()
    assert window.view.range_circles["ground"].get_visible()
    assert sim.time_s == 0
    plt.close(window.fig)


def test_even_spacing_matches_hand_computed_positions():
    """(Required test.) Hand math: base (0, 0) to mapper (600, 450) is 750 m. With
    every range 300 m and 2 relays there are 3 hops limited to 300 m each, so each
    hop is 750 / 3 = 250 m: relays at (200, 150) and (400, 300), and every hop uses
    250 / 300 = 83.3% of its limit."""
    limits = hop_limits(2, dict.fromkeys(ROLES, 300.0))
    assert limits == [300.0, 300.0, 300.0]
    slots, stretch = chain_slots((0, 0), (600, 450), limits)
    np.testing.assert_allclose(slots, [(200, 150), (400, 300)])
    assert stretch == pytest.approx(750 / 900)


def test_uneven_ranges_and_a_chain_that_cannot_reach():
    """Hand math with different ranges: base 600 m, relays 300 m, mapper 150 m, and
    1 relay give hop limits min(600, 300) = 300 and min(300, 150) = 150 (sum 450).
    - Mapper 400 m out: both hops at 400 / 450 = 88.9% of their limits, so the
      relay sits 300 x 400 / 450 = 266.7 m out.
    - Mapper 900 m out: too far (900 / 450 = 200%). Choice (b): the relay sits at
      its full 300 m from the base, staying linked; the gap is at the mapper's end.
    """
    ranges = {"ground_station": 600.0, "relay": 300.0, "mapper": 150.0}
    limits = hop_limits(1, ranges)
    assert limits == [300.0, 150.0]
    slots, stretch = chain_slots((0, 0), (400, 0), limits)
    np.testing.assert_allclose(slots, [(300 * 400 / 450, 0)])
    assert stretch == pytest.approx(400 / 450)
    slots, stretch = chain_slots((0, 0), (900, 0), limits)
    np.testing.assert_allclose(slots, [(300, 0)])
    assert stretch == pytest.approx(2.0)
    assert hop_limits(0, ranges) == [150.0]  # no relays: one direct hop


def test_split_and_hungarian_pairing():
    """Two mappers, 300 m ranges: one 750 m out (needs relays), one 250 m out (the
    base reaches it directly). 2 relays both go to the far chain: it is always the
    most stretched. Then the Hungarian pairing, hand-checked: relays at x = 0 and
    x = 10, spots at x = 9 and x = 20. Nearest-first would pair 10 -> 9 (1 m) and
    0 -> 20 (20 m): 21 m in all. The best pairing is 0 -> 9 and 10 -> 20: 19 m."""
    ranges = dict.fromkeys(ROLES, 300.0)
    assert split_relays(2, [750.0, 250.0], ranges) == [2, 0]
    assert split_relays(4, [750.0, 750.0], ranges) == [2, 2]
    pairs = match_relays_to_slots({"a": (0, 0), "b": (10, 0)}, [(9, 0), (20, 0)])
    assert pairs == {"a": (9, 0), "b": (20, 0)}


def test_goals_fly_at_speed_and_ranges_change_links_at_once(config):
    """A goal makes a drone fly there at its own speed (no jumping), then hold.
    Changing a range re-checks the links right away, without moving the clock."""
    sim = Simulator(config)
    sim.set_goal("relay_2", (50, 3))  # 3 m north of relay_2, which flies 1 m/s
    for _ in range(10):  # 1 simulated second
        sim.step()
    np.testing.assert_allclose(sim.positions["relay_2"], [50, 1, 10])
    for _ in range(40):  # 4 more seconds: it arrives at 3 s, then holds
        sim.step()
    np.testing.assert_allclose(sim.positions["relay_2"], [50, 3, 10])

    sim = Simulator(config)
    assert "mapper_1" in sim.history[-1].reachable
    sim.set_range("relay", 20.0)  # the relays are 25 m apart: too far now
    assert "mapper_1" not in sim.history[-1].reachable
    assert sim.time_s == 0
    assert sim.config["radio_range_m"]["relay"] == 30  # the start stays on record


def test_window_click_goals_auto_toggle_sliders_and_save_box(config):
    """Clicking a drone and then the map sends it there; checking "keep network
    drones optimal" sends the network drones to the best spots; a slider changes a
    range; the save box can be unchecked. None of it moves the clock."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import MouseEvent

    from tar_swarm.network.visualization import InteractiveWindow

    sim = Simulator(config)
    window = InteractiveWindow(sim)
    window.fig.canvas.draw()  # lay out the window so meters-to-pixels math works
    ax = window.view.ax

    def click(x_m, y_m):
        """Press and release the left button at map point (x_m, y_m)."""
        x_px, y_px = ax.transData.transform((x_m, y_m))
        for kind in ("button_press_event", "button_release_event"):
            event = MouseEvent(kind, window.fig.canvas, x_px, y_px, button=1)
            window.fig.canvas.callbacks.process(kind, event)

    click(50, 0)  # click relay_2: selects it
    assert window.selected == "relay_2"
    click(60, 20)  # click the map: relay_2's new goal
    np.testing.assert_allclose(sim.manual_targets["relay_2"], [60, 20, 10])
    window.auto_box.set_active(0)  # check "auto optimal positions"
    # The mapper is 70 m away; the 3 hops' map limits are 28.3, 30, and 30 m
    # (88.3 m in all), so every hop gets 70 / 88.3 = 79% of its limit: the relays
    # head for 22.4 m and 46.2 m out. The plan wins over relay_2's hand-set goal.
    total_m = BASE_HOP_M + 60
    targets = sim.history[-1].targets
    np.testing.assert_allclose(targets["relay_1"], [70 * BASE_HOP_M / total_m, 0, 10])
    np.testing.assert_allclose(
        targets["relay_2"], [70 * (BASE_HOP_M + 30) / total_m, 0, 10]
    )
    window.view.draw()
    assert "planned hops up to 79%" in ax.get_title()
    window.auto_box.set_active(0)  # uncheck: the network drones hold where they are
    np.testing.assert_allclose(sim.history[-1].targets["relay_2"], [50, 0, 10])
    window.sliders["relay"].set_val(20)  # the relays are 25 m apart
    assert sim.link_model.range_m["relay"] < 25
    assert "mapper_1" not in sim.history[-1].reachable
    window.save_box.set_active(0)  # uncheck "save results when closed"
    assert not window.save_on_close
    assert sim.time_s == 0
    plt.close(window.fig)


def test_relay_planner_follows_a_moving_mapper(config):
    """With automatic placement on, every step re-plans from where the mapper is
    now, so the network drones' targets stay on the best spots as it flies."""
    from tar_swarm.strategies.fixed_relay import optimal_chain_plan

    sim = Simulator(config)
    sim.set_relay_planner(optimal_chain_plan)
    for _ in range(20):  # 2 s: the mapper flies 4 m/s east, from x = 70 to 78
        sim.step()
    frame = sim.history[-1]
    mapper_x = frame.positions["mapper_1"][0]
    assert mapper_x == pytest.approx(78)
    # The 3 hops' map limits are 28.3, 30, and 30 m: each gets its share of 78 m.
    total_m = BASE_HOP_M + 60
    relay_1_x = mapper_x * BASE_HOP_M / total_m
    relay_2_x = mapper_x * (BASE_HOP_M + 30) / total_m
    np.testing.assert_allclose(frame.targets["relay_1"], [relay_1_x, 0, 10])
    np.testing.assert_allclose(frame.targets["relay_2"], [relay_2_x, 0, 10])
    assert sim.plan_stretch == pytest.approx(mapper_x / total_m)


def test_endless_run_goes_past_duration_and_counts_as_completed(config, tmp_path):
    """The live window's endless mode never stops at duration_s; a run that got
    past duration_s is saved as completed, with its real end time."""
    from tar_swarm.network.results import save_results

    config["duration_s"] = 0.5
    sim = Simulator(config)
    sim.endless = True
    for _ in range(10):  # 1 s: twice the configured length
        sim.step()
    assert not sim.done
    summary = save_results(sim, tmp_path / "run")
    assert summary["completed"]
    assert summary["end_time_s"] == pytest.approx(1.0)


def test_unchecked_save_box_skips_saving(monkeypatch, tmp_path):
    """If "save results when closed" was unchecked, closing the window saves
    nothing: no results folder is made."""
    import tar_swarm.network.visualization as visualization
    from tar_swarm.network import runner

    # Pretend the window was opened, the box unchecked, and the window closed.
    monkeypatch.setattr(visualization, "animate", lambda simulator: False)
    output = tmp_path / "run"
    config_path = str(write_small_demo(tmp_path))
    runner.main(["--animate", "--config", config_path, "--output", str(output)])
    assert not output.exists()


def test_routing_picks_fewest_hops_and_breaks_ties_alike():
    """Routing (the batman stand-in) on hand-made link snapshots:
    - a chain: the only route, hop by hop
    - a shortcut: 2 hops through d beat 4 hops through a, b, c
    - a tie: two 2-hop routes; the one through "a" wins, the same every time
    - cut off: no route at all"""
    from tar_swarm.network.graph import GraphSnapshot
    from tar_swarm.network.routing import next_hops, route

    def snapshot(*links):
        nodes = sorted({node for link in links for node in link} | {"alone"})
        return GraphSnapshot(0.0, tuple(nodes), tuple(links))

    chain = snapshot(("ground", "r1"), ("r1", "r2"), ("r2", "m"))
    assert next_hops(chain, "ground") == {"r1": "ground", "r2": "r1", "m": "r2"}
    assert route(chain, "m", "ground") == ["m", "r2", "r1", "ground"]
    shortcut = snapshot(
        ("ground", "a"), ("a", "b"), ("b", "c"), ("c", "m"), ("ground", "d"), ("d", "m")
    )
    assert route(shortcut, "m", "ground") == ["m", "d", "ground"]
    tie = snapshot(("ground", "b"), ("b", "m"), ("ground", "a"), ("a", "m"))
    assert route(tie, "m", "ground") == ["m", "a", "ground"]
    assert route(chain, "alone", "ground") is None


def test_segment_crosses_box_by_hand():
    """The box spans x 0..10 and y 0..10. A line through its middle crosses it; a
    line above it, or one that only touches a corner, doesn't; a line starting
    inside it does."""
    from tar_swarm.common.world import segment_crosses_box

    box = (0, 0, 10, 10)
    assert segment_crosses_box((-5, 5), (15, 5), box)  # straight through
    assert segment_crosses_box((-5, -5), (15, 15), box)  # diagonally through
    assert not segment_crosses_box((-5, 15), (15, 15), box)  # passes above
    assert not segment_crosses_box((-5, 5), (5, 15), box)  # touches corner (0, 10)
    assert not segment_crosses_box((-5, 0), (15, 0), box)  # slides along an edge
    assert segment_crosses_box((5, 5), (20, 20), box)  # starts inside


def test_obstacles_block_links_and_mappers_can_be_added(config):
    """A box between relay_1 and relay_2 cuts the chain at once; removing it heals
    the chain. Mapping drones can be added (mapper_2) and removed, but never the
    last one. A bad box in the config is refused."""
    sim = Simulator(config)
    sim.add_obstacle((35, -5, 40, 5))  # straddles the line between x = 25 and 50
    assert ("relay_1", "relay_2") not in sim.history[-1].graph.edges
    assert "mapper_1" not in sim.history[-1].reachable
    sim.remove_obstacle((35, -5, 40, 5))
    assert "mapper_1" in sim.history[-1].reachable
    assert sim.add_drone("mapper", (60, 20)) == "mapper_2"
    np.testing.assert_array_equal(sim.positions["mapper_2"], [60, 20, 10])
    sim.remove_drone("mapper_1")  # allowed: mapper_2 is still there
    with pytest.raises(ValueError):
        sim.remove_drone("mapper_2")  # not allowed: it's the last mapper
    bad = deepcopy(config)
    bad["obstacles_m"] = [[10, 0, 5, 5]]  # x_min > x_max
    with pytest.raises(ValueError):
        validate_config(bad)


def test_two_mappers_get_a_chain_each(config):
    """Two mappers 40 m out in opposite directions and 2 relays: each chain gets
    one relay. Its 2 hops' map limits are 28.3 m (from the base) and 30 m; they
    share the 40 m, so the relay sits 40 x 28.3 / 58.3 = 19.4 m out and both hops
    use 40 / 58.3 = 69% of their limits."""
    from tar_swarm.strategies.fixed_relay import optimal_chain_plan

    config["agents"] = config["agents"][:2]  # just the base and mapper_1
    sim = Simulator(config)
    sim.move_agent("mapper_1", (40, 0))
    sim.add_drone("mapper", (-40, 0))
    sim.add_drone("relay", (10, 5))
    sim.add_drone("relay", (-10, 5))
    plan, stretch = optimal_chain_plan(sim, sim.history[-1].positions)
    relay_x = 40 * BASE_HOP_M / (BASE_HOP_M + 30)
    np.testing.assert_allclose(plan["relay_1"], (relay_x, 0))
    np.testing.assert_allclose(plan["relay_2"], (-relay_x, 0))
    assert stretch == pytest.approx(40 / (BASE_HOP_M + 30))


def test_window_draws_obstacles_adds_mappers_and_animates_routes(config):
    """In the live window: dragging on empty map draws an obstacle and right-
    clicking it removes it; shift + right-click adds a mapping drone; the route
    from the mapper to the base is drawn, and its dashes slide while running."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import MouseEvent

    from tar_swarm.network.visualization import InteractiveWindow

    sim = Simulator(config)
    window = InteractiveWindow(sim)
    window.fig.canvas.draw()  # lay out the window so meters-to-pixels math works
    ax = window.view.ax

    def mouse(kind, x_m, y_m, button=1, key=None):
        """Send a fake mouse event at map point (x_m, y_m), as if from a user."""
        x_px, y_px = ax.transData.transform((x_m, y_m))
        event = MouseEvent(kind, window.fig.canvas, x_px, y_px, button=button, key=key)
        window.fig.canvas.callbacks.process(kind, event)

    mouse("button_press_event", 100, 20)  # empty map, far from every drone
    mouse("motion_notify_event", 110, 25)
    mouse("button_release_event", 110, 25)
    np.testing.assert_allclose(sim.obstacles, [(100, 20, 110, 25)])
    mouse("button_press_event", 105, 22, button=3)  # right-click inside it
    assert sim.obstacles == []
    mouse("button_press_event", 100, -20, button=3, key="shift")
    assert sim.roles["mapper_2"] == "mapper"
    # The route mapper_1 -> relay_2 -> relay_1 -> ground, drawn from the base out.
    window.view.draw()
    routes = window.view.routes.get_segments()
    np.testing.assert_allclose(routes[0], [(0, 0), (25, 0), (50, 0), (70, 0)])
    window.clock = iter([0.0, 0.1]).__next__  # a pretend clock: 0.1 s per frame
    window.last_frame_s = window.clock()
    window.update()
    assert window.view.route_offset == pytest.approx(1.4)  # 14 units/s x 0.1 s
    plt.close(window.fig)
