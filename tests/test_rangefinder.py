from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
import pytest

from tar_swarm.common.config import load_config, validate_config
from tar_swarm.mujoco.flight import FlightSimulation, cast_rays
from tar_swarm.mujoco.overlays import (
    CLEAR,
    DANGER,
    NO_RETURN,
    Beam,
    draw_sensor_overlays,
    load_overlay_settings,
)
from tar_swarm.mujoco.runner import run_scenario
from tar_swarm.sensors.rangefinder import (
    RangefinderParams,
    apply_noise,
    beam_directions,
    load_rangefinder_params,
    nearest_hit,
    validate_rangefinder_params,
)

ROOT = Path(__file__).resolve().parents[1]

# Drone at the origin facing +x; the box "housing" encloses the sensor site.
# Wall face at x = 3.0; optional thin pole centred 0.04 m off-axis at x = 2.
POLE = '<geom name="pole" type="cylinder" pos="2 0.04 2" size="0.02 2"/>'
XML = """
<mujoco>
  <worldbody>
    <geom name="wall" type="box" pos="3.1 0 2" size="0.1 2 2"/>
    {pole}
    <body name="drone" pos="0 0 2">
      <freejoint/>
      <geom name="housing" type="box" size="0.05 0.05 0.05" mass="1"/>
      <site name="s" zaxis="1 0 0" size="0.01"/>
    </body>
  </worldbody>
</mujoco>
"""


def params(**overrides) -> RangefinderParams:
    base = dict(
        site="luna_fwd",
        rate_hz=100.0,
        min_range_m=0.2,
        max_range_m=8.0,
        effective_max_range_m=4.0,
        noise_sigma_near_m=0.0,
        noise_sigma_far_fraction=0.0,
        noise_switch_m=3.0,
        dropout_probability=0.0,
        num_rays=1,
        beam_half_angle_deg=1.0,
    )
    return RangefinderParams(**{**base, **overrides})


def raw_params(**overrides) -> dict:
    return {"enabled": True, **params().__dict__, **overrides}


def scene(with_pole=False):
    model = mujoco.MjModel.from_xml_string(XML.format(pole=POLE if with_pole else ""))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def cast(model, data, p, exclude=True):
    site = model.site("s")
    return cast_rays(
        model,
        data,
        data.site_xpos[site.id],
        beam_directions(p) @ data.site_xmat[site.id].reshape(3, 3).T,
        model.body("drone").id if exclude else -1,
        p.max_range_m,
    )


def reading(model, data, p, rng=None):
    distance, _, _ = cast(model, data, p)
    hit = nearest_hit(distance)
    true = distance[hit] if hit >= 0 else -1.0
    return apply_noise(np.array([true]), p, rng or np.random.default_rng(0))[0]


def test_wall_at_known_distance():
    model, data = scene()
    assert reading(model, data, params()) == pytest.approx(3.0, abs=1e-9)
    _, geom, _ = cast(model, data, params())
    assert geom[0] == model.geom("wall").id


def test_own_body_excluded():
    model, data = scene()
    distance, geom, _ = cast(model, data, params(), exclude=False)
    assert geom[0] == model.geom("housing").id and distance[0] < 0.06
    distance, geom, _ = cast(model, data, params())
    assert geom[0] == model.geom("wall").id


def test_range_limits():
    p = params()
    raw = np.array([-1.0, 0.1, 0.2, 2.0, 3.99, 4.01, 7.9])
    out = apply_noise(raw, p, np.random.default_rng(0))
    assert np.isnan(out[[0, 1, 5, 6]]).all()
    assert np.allclose(out[[2, 3, 4]], [0.2, 2.0, 3.99])
    # Beyond the datasheet maximum the ray caster itself returns no hit.
    model, data = scene()
    assert np.isnan(
        reading(model, data, params(max_range_m=2.5, effective_max_range_m=2.5))
    )
    # With the outdoor cap lifted, the 3 m wall is reported.
    assert reading(model, data, params(effective_max_range_m=8.0)) == pytest.approx(3)


def test_dropout_rate():
    p = params(dropout_probability=0.3)
    out = apply_noise(np.full(200_000, 2.0), p, np.random.default_rng(0))
    assert np.isnan(out).mean() == pytest.approx(0.3, abs=0.005)


def test_noise_statistics_near_and_far():
    p = params(
        noise_sigma_near_m=0.06,
        noise_sigma_far_fraction=0.02,
        effective_max_range_m=8.0,
    )
    near = apply_noise(np.full(100_000, 2.0), p, np.random.default_rng(1))
    far = apply_noise(np.full(100_000, 6.0), p, np.random.default_rng(1))
    assert near.std() == pytest.approx(0.06, rel=0.02)
    assert far.std() == pytest.approx(0.12, rel=0.02)
    assert near.mean() == pytest.approx(2.0, abs=1e-3)


def test_off_axis_pole_missed_by_single_ray_hit_by_bundle():
    model, data = scene(with_pole=True)
    # Pole spans y = 0.02..0.06 at x = 2; the single axis ray passes it.
    assert reading(model, data, params()) == pytest.approx(3.0, abs=1e-9)
    # A 2 deg beam reaches 0.035 m off-axis at 2 m; the nearest hit is the pole.
    bundle = params(num_rays=50)
    assert reading(model, data, bundle) == pytest.approx(1.98, abs=0.02)


def test_beam_directions():
    assert np.array_equal(beam_directions(params()), [[0, 0, 1]])
    d = beam_directions(params(num_rays=40, beam_half_angle_deg=1.0))
    assert np.allclose(np.linalg.norm(d, axis=1), 1)
    assert np.allclose(d[0], [0, 0, 1])
    polar = np.degrees(np.arccos(np.clip(d[:, 2], -1, 1)))
    assert polar.max() == pytest.approx(1.0) and (polar <= 1.0 + 1e-9).all()


def test_nearest_hit():
    assert nearest_hit(np.array([-1.0, 3.0, 2.0, -1.0])) == 2
    assert nearest_hit(np.array([-1.0, -1.0])) == -1


@pytest.mark.parametrize(
    "key,value",
    [
        ("rate_hz", 0.0),
        ("min_range_m", 5.0),
        ("effective_max_range_m", 9.0),
        ("noise_sigma_near_m", -0.1),
        ("dropout_probability", 1.0),
        ("num_rays", 0),
        ("num_rays", 1.5),
        ("beam_half_angle_deg", 90.0),
        ("site", ""),
        ("enabled", "yes"),
    ],
)
def test_invalid_params_rejected(key, value):
    with pytest.raises(ValueError):
        validate_rangefinder_params(raw_params(**{key: value}))


def test_missing_key_rejected():
    raw = raw_params()
    raw.pop("noise_switch_m")
    with pytest.raises(ValueError):
        validate_rangefinder_params(raw)


def test_default_config_relay_only_and_rate_checked():
    config = load_config(ROOT / "configs/simulation.yaml")
    assert load_rangefinder_params(config["sensors"], "relay_drone").rate_hz == 100
    assert load_rangefinder_params(config["sensors"], "mapping_drone") is None
    bad = deepcopy(config)
    bad["sensors"]["rangefinder"]["relay_drone"]["rate_hz"] = 300.0
    with pytest.raises(ValueError):
        validate_config(bad)


def test_forward_sites_in_scene():
    config = load_config(ROOT / "configs/simulation.yaml")
    sim = FlightSimulation(config)
    position, rotation = sim.site_pose("relay_0_luna_fwd")
    body = sim.state("relay_0")
    assert np.allclose(rotation[:, 2], body.rotation[:, 0])
    with pytest.raises(KeyError):
        sim.model.site("mapper_0_luna_fwd")
    config["sensors"]["rangefinder"]["mapping_drone"]["enabled"] = True
    sim = FlightSimulation(config)
    _, rotation = sim.site_pose("mapper_0_luna_fwd")
    assert np.allclose(rotation[:, 2], sim.state("mapper_0").rotation[:, 0])


def test_run_writes_readings(tmp_path):
    config = load_config(ROOT / "configs/simulation.yaml")
    config["simulation"]["duration_s"] = 0.5
    out = tmp_path / "run"
    result = run_scenario(config, output=out, plots=False)
    relays = {"relay_0", "relay_1", "relay_2"}
    assert {r["vehicle_id"] for r in result["rangefinder"]} == relays
    readings = np.load(out / "rangefinder.npz")
    truth = np.load(out / "rangefinder_truth.npz")
    assert len(readings["time_s"]) == 3 * 51 == len(truth["true_distance_m"])
    one = readings["time_s"][readings["vehicle_id"] == "relay_0"]
    assert np.allclose(np.diff(one), 0.01)
    valid = readings["distance_m"][np.isfinite(readings["distance_m"])]
    assert ((valid >= 0.2) & (valid <= 4.0)).all()
    assert truth["site_rotation_world"].shape == (153, 3, 3)


def test_beam_overlay_geoms():
    model = mujoco.MjModel.from_xml_string(XML.format(pole=""))
    scn = mujoco.MjvScene(model, maxgeom=10)
    axis = np.array([1.0, 0, 0])
    beams = {
        "clear": Beam(np.zeros(3), axis, 2.0, 4.0),
        "close": Beam(np.zeros(3), axis, 0.5, 4.0),
        "none": Beam(np.zeros(3), axis, np.nan, 4.0),
    }
    settings = {"rangefinder_beam": {"enabled": True, "d_stop_m": 1.0}}
    draw_sensor_overlays(scn, settings, beams)
    assert scn.ngeom == 3
    for geom, color, length in zip(
        scn.geoms[:3], (CLEAR, DANGER, NO_RETURN), (2.0, 0.5, 4.0), strict=True
    ):
        assert np.allclose(geom.rgba, color)
        assert np.allclose(geom.pos, [length / 2, 0, 0], atol=1e-6)
    scn.ngeom = 0
    draw_sensor_overlays(scn, {"rangefinder_beam": {"enabled": False}}, beams)
    assert scn.ngeom == 0


def test_overlay_settings_file_valid():
    settings = load_overlay_settings(ROOT / "configs/overlays.yaml")
    assert settings["rangefinder_beam"]["enabled"] is True
