from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
import pytest

from tar_swarm.common.config import load_config, validate_config
from tar_swarm.mujoco.flight import FlightSimulation, cast_rays
from tar_swarm.mujoco.runner import run_scenario
from tar_swarm.sensors.lidar import (
    LidarParams,
    apply_noise,
    load_lidar_params,
    scan_directions,
    to_points,
    validate_lidar_params,
)

ROOT = Path(__file__).resolve().parents[1]

XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="50 50 0.1"/>
    <geom name="wall" type="box" pos="5 0 1" size="0.1 2 2"/>
    <body name="drone" pos="0 0 3">
      <freejoint/>
      <geom name="plate" type="box" pos="0 0 -0.1" size="0.5 0.5 0.005" mass="1"/>
      <site name="s" zaxis="0 0 -1" size="0.01"/>
    </body>
  </worldbody>
</mujoco>
"""


def params(**overrides) -> LidarParams:
    base = dict(
        site="lidar_down",
        rate_hz=10.0,
        points_per_scan=2000,
        max_polar_deg=90.0,
        min_range_m=0.05,
        max_range_m=30.0,
        range_noise_m=0.0,
        dropout_probability=0.0,
        pattern="random_uniform",
    )
    return LidarParams(**{**base, **overrides})


def scene(position, roll_deg=0.0):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    half = np.radians(roll_deg) / 2
    data.qpos[:3] = position
    data.qpos[3:7] = [np.cos(half), np.sin(half), 0, 0]
    mujoco.mj_forward(model, data)
    return model, data


def cast_from_site(model, data, directions_site, exclude=True, max_range=30.0):
    site = model.site("s")
    return cast_rays(
        model,
        data,
        data.site_xpos[site.id],
        directions_site @ data.site_xmat[site.id].reshape(3, 3).T,
        model.body("drone").id if exclude else -1,
        max_range,
    )


def test_directions_unit_within_cap_and_deterministic():
    p = params(points_per_scan=5000, max_polar_deg=75.0)
    a = scan_directions(p, np.random.default_rng(1))
    b = scan_directions(p, np.random.default_rng(1))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, scan_directions(p, np.random.default_rng(2)))
    assert np.allclose(np.linalg.norm(a, axis=1), 1)
    assert np.degrees(np.arccos(a[:, 2])).max() <= 75.0 + 1e-9


def test_cap_is_uniform():
    # Uniform on a cap: cos(polar) is uniform in [cos(max), 1].
    for cap in (90.0, 60.0):
        p = params(points_per_scan=100_000, max_polar_deg=cap)
        d = scan_directions(p, np.random.default_rng(0))
        expected = (1 + np.cos(np.radians(cap))) / 2
        assert d[:, 2].mean() == pytest.approx(expected, rel=0.01)


def test_level_sensor_over_flat_ground():
    model, data = scene([0, 0, 3])
    p = params(points_per_scan=5000, max_polar_deg=45)
    d = scan_directions(p, np.random.default_rng(3))
    dist, geom, normal = cast_from_site(model, data, d)
    assert (dist > 0).all()
    assert np.allclose(dist, 3.0 / d[:, 2], atol=1e-6)
    assert (geom == model.geom("floor").id).all()
    assert np.allclose(normal, [0, 0, 1], atol=1e-9)


def test_tilted_sensor_axis_and_minimum_return():
    h = 3.0
    model, data = scene([0, 0, h], roll_deg=20)
    axis, *_ = cast_from_site(model, data, np.array([[0.0, 0.0, 1.0]]))
    assert axis[0] == pytest.approx(h / np.cos(np.radians(20)), rel=1e-6)
    d = scan_directions(params(points_per_scan=100_000), np.random.default_rng(4))
    dist, *_ = cast_from_site(model, data, d)
    assert dist[dist > 0].min() == pytest.approx(h, rel=1e-3)


def test_box_face_distance_and_normal():
    model, data = scene([0, 0, 1])
    origin = data.xpos[model.body("drone").id]
    dist, geom, normal = cast_rays(
        model, data, origin, np.array([[1.0, 0, 0]]), model.body("drone").id, 30
    )
    assert dist[0] == pytest.approx(4.9)
    assert geom[0] == model.geom("wall").id
    assert np.allclose(normal[0], [-1, 0, 0])


def test_own_body_excluded():
    model, data = scene([0, 0, 3])
    down = np.array([[0.0, 0.0, 1.0]])
    near, geom, _ = cast_from_site(model, data, down, exclude=False)
    assert near[0] == pytest.approx(0.095)
    assert geom[0] == model.geom("plate").id
    far, geom, _ = cast_from_site(model, data, down)
    assert far[0] == pytest.approx(3.0)
    assert model.geom_bodyid[geom[0]] != model.body("drone").id


def test_range_limits_and_dropout():
    p = params(min_range_m=1.0, max_range_m=10.0)
    raw = np.array([-1.0, 0.5, 1.0, 5.0, 10.0, 12.0])
    out = apply_noise(raw, p, np.random.default_rng(0))
    assert np.isnan(out[[0, 1, 5]]).all()
    assert np.array_equal(out[[2, 3, 4]], [1.0, 5.0, 10.0])
    p = params(dropout_probability=0.3)
    out = apply_noise(np.full(200_000, 5.0), p, np.random.default_rng(0))
    assert np.isnan(out).mean() == pytest.approx(0.3, abs=0.01)


def test_noise_statistics_and_determinism():
    p = params(range_noise_m=0.05)
    a = apply_noise(np.full(100_000, 5.0), p, np.random.default_rng(7))
    b = apply_noise(np.full(100_000, 5.0), p, np.random.default_rng(7))
    assert np.array_equal(a, b)
    assert np.std(a) == pytest.approx(0.05, rel=0.02)


def test_to_points_valid_only():
    d = np.array([[0, 0, 1.0], [1, 0, 0], [0, 1, 0]])
    pts = to_points(d, np.array([2.0, np.nan, 3.0]))
    assert np.allclose(pts, [[0, 0, 2], [0, 3, 0]])


@pytest.mark.parametrize(
    "key,value",
    [
        ("rate_hz", 0),
        ("points_per_scan", 0),
        ("max_polar_deg", 181),
        ("min_range_m", 40),
        ("range_noise_m", -1),
        ("dropout_probability", 1.0),
        ("pattern", "spiral"),
    ],
)
def test_invalid_params_rejected(key, value):
    config = load_config(ROOT / "configs/simulation.yaml")
    raw = deepcopy(config["sensors"]["lidar"]["mapping_drone"])
    validate_lidar_params(raw)
    raw[key] = value
    with pytest.raises(ValueError):
        validate_lidar_params(raw)


def test_missing_key_rejected():
    config = load_config(ROOT / "configs/simulation.yaml")
    raw = deepcopy(config["sensors"]["lidar"]["mapping_drone"])
    raw.pop("site")
    with pytest.raises(ValueError):
        validate_lidar_params(raw)


def test_only_mapper_has_lidar_and_rate_checked():
    config = load_config(ROOT / "configs/simulation.yaml")
    assert load_lidar_params(config["sensors"], "relay_drone") is None
    assert load_lidar_params(config["sensors"], "mapping_drone").rate_hz == 10.0
    config["sensors"]["lidar"]["mapping_drone"]["rate_hz"] = 7.0
    with pytest.raises(ValueError):
        validate_config(config)


def test_flight_cast_matches_site_height():
    config = load_config(ROOT / "configs/simulation.yaml")
    sim = FlightSimulation(config)
    z = sim.site_pose("mapper_0_lidar_down")[0][2]
    d = scan_directions(params(points_per_scan=20_000), np.random.default_rng(0))
    dist, geom, _ = sim.cast_rays("mapper_0", "mapper_0_lidar_down", d, 30.0)
    assert dist[dist > 0].min() == pytest.approx(z, rel=1e-2)
    own = sim.model.geom_bodyid[geom[geom >= 0]] == sim.handles["mapper_0"].body_id
    assert not own.any()


def test_run_writes_scans(tmp_path):
    config = load_config(ROOT / "configs/simulation.yaml")
    config["simulation"]["duration_s"] = 0.5
    out = tmp_path / "run"
    result = run_scenario(config, output=out, plots=False)
    assert len(result["lidar"]) == 6
    assert {s["vehicle_id"] for s in result["lidar"]} == {"mapper_0"}
    scans = np.load(out / "lidar_scans.npz")
    truth = np.load(out / "lidar_truth.npz")
    assert np.allclose(np.diff(scans["time_s"]), 0.1)
    assert len(scans["points"]) == len(truth["geom_id"])
    assert len(scans["scan_start"]) == len(scans["time_s"])
    ranges = np.linalg.norm(scans["points"], axis=1)
    assert ranges.min() >= 0.05 - 1e-6 and ranges.max() <= 30.0 + 1e-3
