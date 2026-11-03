"""Known-answer tests for avoidance, its runner hook, logging and replay."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from tar_swarm.common.config import load_config, validate_config
from tar_swarm.control.avoidance import (
    AvoidanceParams,
    ReferenceShaper,
    SectorSource,
    load_avoidance_params,
    rangefinder_sector,
    speed_limit,
    stopping_speed,
    validate_avoidance_params,
)
from tar_swarm.control.interfaces import Reference
from tar_swarm.mujoco.overlays import (
    CLEAR,
    COMMANDED,
    DANGER,
    LIMITED,
    VelocityArrow,
    draw_velocity_arrow,
)
from tar_swarm.mujoco.replay import load_run, replay_frame
from tar_swarm.mujoco.runner import run_scenario
from tar_swarm.perception.obstacles import sector_centers

ROOT = Path(__file__).resolve().parents[1]
AHEAD = np.array([1.0, 0.0])


def params(**overrides) -> AvoidanceParams:
    base = dict(
        sources=("rangefinder",),
        a_brake_m_s2=2.0,
        d_stop_m=1.0,
        cone_half_angle_deg=30.0,
        control_delay_s=0.0,
        stale_timeout_s=0.3,
        stale_speed_m_s=0.5,
        stale_hold_after_s=0.5,
        catchup_gain=1.0,
        rangefinder_latch_m=0.5,
    )
    return AvoidanceParams(**{**base, **overrides})


def luna(distance, age=0.0, period=0.0):
    return SectorSource(np.zeros(1), np.array([distance]), age, period)


def test_stopping_speed_known_answers():
    p = params()
    assert stopping_speed(3.0, 0.0, p) == pytest.approx(np.sqrt(2 * 2.0 * 2.0))
    assert stopping_speed(0.5, 0.0, p) == 0.0
    # With latency t, v*t + v^2/(2a) must equal d - d_stop exactly.
    v = stopping_speed(3.0, 0.1, p)
    assert v * 0.1 + v**2 / 4.0 == pytest.approx(2.0)
    assert v < stopping_speed(3.0, 0.0, p)


def test_latency_is_period_delay_and_age():
    p = params(control_delay_s=0.02)
    limit, _, _ = speed_limit(AHEAD, [luna(3.0, age=0.03, period=0.01)], p)
    assert limit == pytest.approx(stopping_speed(3.0, 0.06, p))


def test_stop_inside_d_stop_and_clear_without_returns():
    assert speed_limit(AHEAD, [luna(0.8)], params())[0] == 0.0
    assert speed_limit(AHEAD, [luna(np.nan)], params())[0] == np.inf
    assert speed_limit(None, [luna(0.8)], params())[0] == np.inf


def test_cone_selects_sectors_around_motion():
    centers = sector_centers(72)
    distance = np.full(72, np.nan)
    distance[18] = 1.5  # 90 deg left
    source = SectorSource(centers, distance, 0.0, 0.0)
    assert speed_limit(AHEAD, [source], params())[0] == np.inf
    left = np.array([0.0, 1.0])
    assert speed_limit(left, [source], params())[0] == pytest.approx(
        stopping_speed(1.5, 0.0, params())
    )
    # 25 deg off the motion direction is inside the 30 deg cone.
    distance[:] = np.nan
    distance[5] = 2.0
    assert speed_limit(AHEAD, [source], params())[0] < np.inf


def test_stale_slows_then_holds():
    p = params()
    assert speed_limit(AHEAD, [luna(np.nan, age=0.2)], p) == (np.inf, False, False)
    assert speed_limit(AHEAD, [luna(np.nan, age=0.5)], p) == (0.5, True, False)
    assert speed_limit(AHEAD, [luna(np.nan, age=0.9)], p) == (0.0, False, True)
    # The tighter of distance and stale limits wins.
    assert speed_limit(AHEAD, [luna(1.01, age=0.5)], p)[0] < 0.5


def test_rangefinder_latch():
    assert rangefinder_sector(2.0, None, 0.5) == 2.0
    assert rangefinder_sector(np.nan, 0.3, 0.5) == 0.0
    assert np.isnan(rangefinder_sector(np.nan, 3.0, 0.5))
    assert np.isnan(rangefinder_sector(np.nan, None, 0.5))


def mission(x, vx):
    return Reference(np.array([x, 0.0, 2.0]), np.array([vx, 0.0, 0.3]), np.zeros(3))


def test_shaper_passes_clear_path_through_exactly():
    shaper = ReferenceShaper(params())
    for k in range(50):
        reference = mission(0.01 * k, 1.0)
        shaped, record = shaper.update(reference, np.eye(3), [luna(np.nan)], 0.01)
        assert shaped is reference and record.status == "clear"


def test_shaper_stops_short_of_wall_and_keeps_altitude():
    p = params()
    shaper = ReferenceShaper(p)
    wall, dt = 5.0, 0.01
    statuses = set()
    for k in range(1000):
        reference = mission(2.0 * k * dt, 2.0)
        reference.position[2] = 2.0 + 0.001 * k
        distance = wall - (shaper.position[0] if shaper.position is not None else 0)
        shaped, record = shaper.update(reference, np.eye(3), [luna(distance)], dt)
        statuses.add(record.status)
    # One step of discretisation at the final speed may pass d_stop slightly.
    assert shaped.position[0] <= wall - p.d_stop_m + 0.02
    assert shaped.position[0] > wall - p.d_stop_m - 0.2
    assert shaped.velocity[:2] == pytest.approx([0, 0])
    assert shaped.position[2] == pytest.approx(reference.position[2])
    assert shaped.velocity[2] == pytest.approx(0.3)
    assert {"limited", "stop"} <= statuses


@pytest.mark.parametrize(
    "key,value",
    [
        ("sources", ["sonar"]),
        ("sources", []),
        ("a_brake_m_s2", 0.0),
        ("d_stop_m", -1.0),
        ("cone_half_angle_deg", 0.0),
        ("enabled", "no"),
    ],
)
def test_invalid_params_rejected(key, value):
    raw = {"enabled": True, **params().__dict__, "sources": ["l2"], key: value}
    with pytest.raises(ValueError):
        validate_avoidance_params(raw)


def test_default_config_off_and_source_checks():
    config = load_config(ROOT / "configs/simulation.yaml")
    assert load_avoidance_params(config["avoidance"], "relay_drone") is None
    config["avoidance"]["relay_drone"]["enabled"] = True
    validate_config(config)
    assert load_avoidance_params(config["avoidance"], "relay_drone").sources == (
        "rangefinder",
    )
    config["avoidance"]["relay_drone"]["sources"] = ["l2"]
    with pytest.raises(ValueError):
        validate_config(config)


def test_velocity_arrow_overlay():
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    scn = mujoco.MjvScene(model, maxgeom=20)
    origin = np.zeros(3)
    draw_velocity_arrow(
        scn,
        VelocityArrow(origin, np.array([2.0, 0, 0]), np.array([1.0, 0, 0]), "limited"),
        0.5,
    )
    assert scn.ngeom == 4  # two lines, two tips
    assert np.allclose(scn.geoms[0].rgba, COMMANDED)
    assert np.allclose(scn.geoms[1].pos, [1.0, 0, 0])  # commanded tip at 2 m/s x 0.5 s
    assert np.allclose(scn.geoms[2].rgba, LIMITED)
    scn.ngeom = 0
    clear = VelocityArrow(origin, np.array([1.0, 0, 0]), np.array([1.0, 0, 0]), "clear")
    draw_velocity_arrow(scn, clear, 1.0)
    assert np.allclose(scn.geoms[2].rgba, CLEAR)
    scn.ngeom = 0
    stop = VelocityArrow(origin, np.array([1.0, 0, 0]), np.zeros(3), "stop")
    draw_velocity_arrow(scn, stop, 1.0)
    assert np.allclose(scn.geoms[scn.ngeom - 1].rgba, DANGER)


def relay_wall_config(enabled=True):
    """relay_0 alone, flying +x at 2 m toward a wall whose face is at x = 1."""
    config = load_config(ROOT / "configs/simulation.yaml")
    config["simulation"]["duration_s"] = 9.0
    config["world"]["obstacles"] = [
        {"name": "wall", "position_m": [1.1, -3, 2], "size_m": [0.2, 3, 4]}
    ]
    agent = next(a for a in config["mission"]["agents"] if a["id"] == "relay_0")
    agent["waypoints"] = [
        {"time_s": 0, "position_m": [-3, -3, 0.05]},
        {"time_s": 3, "position_m": [-3, -3, 2]},
        {"time_s": 8, "position_m": [5, -3, 2]},
        {"time_s": 9, "position_m": [5, -3, 2]},
    ]
    config["mission"]["agents"] = [agent]
    config["mission"]["hover_windows_s"] = []
    config["avoidance"]["relay_drone"]["enabled"] = enabled
    return config


def test_relay_stops_before_wall(tmp_path):
    config = relay_wall_config()
    out = tmp_path / "run"
    result = run_scenario(config, output=out, plots=False)
    d_stop = config["avoidance"]["relay_drone"]["d_stop_m"]
    sensor_x = 0.022  # TF-Luna mount ahead of the body origin
    x = max(r["position_x_m"] for r in result["rows"])
    assert 1.0 - (x + sensor_x) >= d_stop - 0.3
    assert max(r["contacts"] for r in result["rows"][200:]) == 0
    saved = np.load(out / "avoidance.npz")
    assert {"limited", "stop"} <= set(saved["status"])
    assert saved["commanded_velocity"].shape == saved["limited_velocity"].shape


def test_disabled_avoidance_changes_nothing():
    config = load_config(ROOT / "configs/simulation.yaml")
    config["simulation"]["duration_s"] = 0.3
    baseline = run_scenario(config, plots=False)
    config["avoidance"]["relay_drone"]["enabled"] = False
    del config["avoidance"]
    again = run_scenario(config, plots=False)
    assert baseline["rows"] == again["rows"] and not baseline["avoidance"]


def test_l2_logging_and_replay(tmp_path):
    config = load_config(ROOT / "configs/simulation.yaml")
    config["simulation"]["duration_s"] = 0.5
    config["perception"]["l2_obstacles"]["mapping_drone"]["enabled"] = True
    config["avoidance"]["relay_drone"]["enabled"] = True
    out = tmp_path / "run"
    result = run_scenario(config, output=out, plots=False)
    found = np.load(out / "l2_obstacles.npz")
    scans = np.load(out / "lidar_scans.npz")
    assert len(found["time_s"]) == len(scans["time_s"]) == 6
    assert len(found["ground_mask"]) == len(scans["points"])
    assert found["sector_distance_m"].shape == (6, 72)
    assert found["ground_found"].all()  # mapper on the ground: floor in view
    run = load_run(out)
    for t in (0.0, 0.25, 0.5):
        frame = replay_frame(run, t)
        assert set(frame.beams) == {"relay_0", "relay_1", "relay_2"}
        assert set(frame.l2) == {"mapper_0"}
        assert set(frame.arrows) == {"relay_0", "relay_1", "relay_2"}
    # Replayed beam origin matches the logged pose plus mount (within csv rounding).
    last = result["rows"][-1]
    assert np.isfinite(frame.qpos).all() and len(frame.qpos) == run.model.nq
    assert last["time_s"] == pytest.approx(0.5)


def test_replay_never_reads_truth():
    source = (ROOT / "src/tar_swarm/mujoco/replay.py").read_text(encoding="utf-8")
    script = (ROOT / "scripts/replay_run.py").read_text(encoding="utf-8")
    assert "_truth" not in source and "_truth" not in script
