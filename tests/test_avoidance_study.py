"""AV5 metrics, sweep criteria, check plots and GIF recording."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from PIL import Image

from tar_swarm.common.config import load_config
from tar_swarm.evaluation.avoidance_metrics import (
    first_reaction,
    interventions,
    is_false_stop,
    time_to_stop,
)
from tar_swarm.evaluation.avoidance_plots import plot_avoidance
from tar_swarm.evaluation.avoidance_study import (
    Case,
    cases,
    criteria,
    run_case,
    summarize,
)
from tar_swarm.mujoco.overlays import load_overlay_settings
from tar_swarm.mujoco.recording import GifRecorder, record_replay
from tar_swarm.mujoco.runner import run_scenario

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = load_overlay_settings(ROOT / "configs/overlays.yaml")


def test_reaction_stop_and_intervention_known_answers():
    times = np.array([0.0, 4.0, 5.0, 6.0, 7.0])
    statuses = np.array(["stale_hold", "clear", "limited", "stop", "stop"])
    assert first_reaction(times, statuses) == 2
    assert first_reaction(times, statuses, after_s=5.5) == 3
    assert first_reaction(times, np.full(5, "clear")) is None
    assert interventions(times, statuses) == 3  # the t = 0 start-up tick is skipped
    speeds = np.array([0.0, 2.0, 1.5, 0.3, 0.05])
    assert time_to_stop(times, speeds, 5.0) == pytest.approx(2.0)
    assert np.isnan(time_to_stop(times, speeds, 5.0, threshold=0.01))


def test_false_stop_only_without_obstacles():
    assert not is_false_stop(True, 100, False)
    assert not is_false_stop(False, 0, True)
    assert is_false_stop(False, 1, True)
    assert is_false_stop(False, 0, False)


def row(world, speed, collided=False, false_stop=False, sensor="l2", **extra):
    return {
        "name": f"{world}_{sensor}_{speed}",
        "world": world,
        "sensor": sensor,
        "speed_m_s": speed,
        "noise": True,
        "collided": collided,
        "false_stop": false_stop,
        "completed": True,
        "ground_edge_margin_m": 1.0,
        "min_clearance_m": 0.5,
        "stopping_distance_m": 1.0,
        "time_to_stop_s": 1.0,
        "reaction_speed_m_s": speed,
        **extra,
    }


def test_criteria_scope():
    rows = [
        row("avoid_wall", 2.0),
        row("avoid_pole", 1.0),
        row("avoid_wall", 3.0, collided=True),  # above 2 m/s: reported, not required
        row("avoid_thin", 1.0, collided=True),  # a limit, not a required pass
        row("clear_path", 5.0),
    ]
    assert [c["passed"] for c in criteria(rows)] == [True, True, True]
    rows.append(row("avoid_pole", 2.0, collided=True, sensor="rangefinder"))
    rows.append(row("clear_path", 1.0, false_stop=True))
    rows.append(row("clear_path", 2.0, ground_edge_margin_m=-0.1))
    collisions, false_stops, edge = criteria(rows)
    assert collisions["failures"] == ["avoid_pole_rangefinder_2.0"]
    assert false_stops["failures"] == ["clear_path_l2_1.0"]
    assert edge["failures"] == ["clear_path_l2_2.0"]
    assert not criteria([])[0]["passed"]  # nothing checked is not a pass


def test_cases_and_summary():
    grid = cases()
    assert len(grid) == 5 * 2 * 4 * 2 * 3
    assert len({c.name for c in grid}) == len(grid)
    assert Case("avoid_wall", "l2", 2.0, False, 42).name == "avoid_wall_l2_v2_clean_s42"
    groups = summarize(
        [
            row("avoid_wall", 2.0),
            row("avoid_wall", 2.0, collided=True, min_clearance_m=0),
        ]
    )
    assert groups[0]["runs"] == 2 and groups[0]["collisions"] == 1
    assert groups[0]["min_clearance_m"] == 0


def test_gif_recorder(tmp_path):
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><geom type='sphere' size='0.1'/></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with pytest.raises(ValueError):
        GifRecorder(model, tmp_path / "clip.mp4")
    path = tmp_path / "clip.gif"
    with GifRecorder(model, path, fps=5) as recorder:
        # Distinct frames: Pillow merges identical ones (keeping total duration).
        for k in range(3):
            recorder.capture(data, np.array([0.1 * k, 0, 0]))
    with Image.open(path) as clip:
        assert clip.n_frames == 3
        assert clip.info["duration"] == 200
    with pytest.raises(FileExistsError):
        GifRecorder(model, path)


def test_record_does_not_change_the_run(tmp_path):
    config = load_config(ROOT / "configs/simulation.yaml")
    config["simulation"]["duration_s"] = 0.3
    plain = run_scenario(config, plots=False)
    recorded = run_scenario(config, plots=False, record=tmp_path / "run.gif")
    assert recorded["rows"] == plain["rows"]
    with Image.open(tmp_path / "run.gif") as clip:
        assert clip.n_frames == 4  # t = 0, 0.1, 0.2, 0.3 at 10 fps


@pytest.fixture(scope="module")
def wall_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("sweep")
    case = Case("avoid_wall", "rangefinder", 2.0, False, 42)
    return root / case.name, run_case(case, root, record=SETTINGS)


def test_wall_metrics(wall_run):
    _, metrics = wall_run
    assert not metrics["collided"] and not metrics["false_stop"]
    assert metrics["contacts"] == 0
    # Face at x = 12; TF-Luna d_stop 1 m, relay nose and tracking lag eat ~0.3 m.
    assert 0.5 < metrics["min_clearance_m"] < 1.0
    assert 9.0 < metrics["reaction_x_m"] < metrics["stop_x_m"] < 11.5
    assert metrics["stopping_distance_m"] == pytest.approx(
        metrics["stop_x_m"] - metrics["reaction_x_m"]
    )
    assert 0 < metrics["time_to_stop_s"] < 3
    assert not metrics["reached_end"] and metrics["clip"] == ""  # no clip: passed
    assert metrics["ground_edge_margin_m"] > 0


def test_check_plot_and_replay_clip(wall_run, tmp_path):
    directory, _ = wall_run
    figure = plot_avoidance(directory, tmp_path / "check.png")
    assert figure.stat().st_size > 10_000
    clip = record_replay(
        directory, tmp_path / "clip.gif", SETTINGS, start_s=16.0, end_s=17.0
    )
    with Image.open(clip) as gif:
        assert gif.n_frames == 11


def test_clear_path_has_no_false_stop(tmp_path):
    metrics = run_case(Case("clear_path", "rangefinder", 5.0, True, 42), tmp_path)
    assert metrics["reached_end"] and not metrics["false_stop"]
    assert metrics["intervention_ticks"] == 0
    assert np.isnan(metrics["min_clearance_m"]) and not metrics["collided"]


def test_plots_and_recording_never_read_truth():
    for module in ("evaluation/avoidance_plots.py", "mujoco/recording.py"):
        source = (ROOT / "src/tar_swarm" / module).read_text(encoding="utf-8")
        assert "_truth" not in source, module
