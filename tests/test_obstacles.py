"""Known-answer tests for L2 obstacle extraction on synthetic point sets."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tar_swarm.common.config import load_config, validate_config
from tar_swarm.perception.obstacles import (
    ObstacleParams,
    extract_obstacles,
    fit_ground_plane,
    heading_rotation,
    load_obstacle_params,
    sector_centers,
    sector_distances,
    to_level_frame,
    validate_obstacle_params,
)

ROOT = Path(__file__).resolve().parents[1]
DOWN = np.diag([1.0, -1.0, -1.0])  # L2 mount: sensor +z is body -z


def params(**overrides) -> ObstacleParams:
    base = dict(
        num_sectors=72,
        band_below_m=0.5,
        band_above_m=0.5,
        ransac_iterations=100,
        ransac_threshold_m=0.1,
        ground_max_tilt_deg=15.0,
        ground_min_points=50,
        min_range_m=0.3,
    )
    return ObstacleParams(**{**base, **overrides})


def ground(height, n=3000, half=20.0, seed=0):
    """Level-frame ground points `height` below the body origin."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-half, half, (n, 2))
    return np.column_stack([xy, np.full(n, -height)])


def test_ground_fully_removed_on_flat_ground():
    points = ground(3.0)
    scan = extract_obstacles(
        points @ DOWN, DOWN, np.zeros(3), np.eye(3), params(), np.random.default_rng(1)
    )
    assert scan.ground_mask.all()
    assert scan.ground.offset == pytest.approx(3.0, abs=1e-6)
    assert np.isnan(scan.sector_distance_m).all()


def test_ground_removed_at_low_altitude_without_true_height():
    # At 0.3 m the ground falls inside the height band; RANSAC must remove it.
    points = ground(0.3, half=10.0)
    scan = extract_obstacles(
        points @ DOWN, DOWN, np.zeros(3), np.eye(3), params(), np.random.default_rng(2)
    )
    assert scan.ground_mask.all()
    assert scan.sector_count.sum() == 0


def test_point_obstacle_sector_and_distance():
    obstacle = np.array([[0.0, 4.0, 0.1]])  # 90 deg left, inside the band
    points = np.vstack([ground(3.0), obstacle])
    scan = extract_obstacles(
        points @ DOWN, DOWN, np.zeros(3), np.eye(3), params(), np.random.default_rng(3)
    )
    assert not scan.ground_mask[-1]
    assert np.flatnonzero(np.isfinite(scan.sector_distance_m)).tolist() == [18]
    assert scan.sector_distance_m[18] == pytest.approx(4.0)
    assert scan.sector_count[18] == 1


def test_wall_ahead_nearest_per_sector():
    y = np.linspace(-1, 1, 41)
    wall = np.column_stack([np.full_like(y, 5.0), y, np.zeros_like(y)])
    distance, count = sector_distances(wall, params())
    # A 2 m wide wall at 5 m spans about +/-11.3 deg: sectors -2..2.
    seen = np.flatnonzero(np.isfinite(distance))
    assert sorted(seen.tolist()) == [0, 1, 2, 70, 71]
    assert distance[0] == pytest.approx(5.0)
    assert count.sum() == 41


def test_sector_boundaries_centred_on_heading():
    angles = np.radians([2.4, -2.4, 2.6, 177.4, 182.4])
    points = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(5)]) * 2
    distance, count = sector_distances(points, params())
    assert count[0] == 2 and count[1] == 1 and count[35] == 1 and count[36] == 1
    assert np.allclose(np.degrees(sector_centers(72)[[0, 1, 36]]), [0, 5, 180])


def test_height_band_and_min_range():
    points = np.array(
        [
            [3.0, 0, 0.6],  # above band
            [3.0, 0, -0.6],  # below band
            [0.2, 0, 0.0],  # inside the airframe radius
            [3.0, 0, 0.45],  # kept
        ]
    )
    distance, count = sector_distances(points, params())
    assert count.sum() == 1 and distance[0] == pytest.approx(3.0)


def test_level_frame_removes_roll_pitch_keeps_heading():
    yaw = Rotation.from_euler("z", 70, degrees=True).as_matrix()
    assert np.allclose(heading_rotation(yaw), yaw)
    body = Rotation.from_euler("zyx", [70, 10, -5], degrees=True).as_matrix()
    # A point 4 m ahead of the body, expressed in the sensor frame.
    ahead_level = np.array([[4.0, 0.0, 0.0]])
    tilt = heading_rotation(body).T @ body
    sensor = (ahead_level @ tilt) @ DOWN
    assert np.allclose(to_level_frame(sensor, DOWN, np.zeros(3), body), ahead_level)
    offset = np.array([0.0, 0.0, -0.14])
    level = to_level_frame(np.zeros((1, 3)), DOWN, offset, np.eye(3))
    assert np.allclose(level, [[0, 0, -0.14]])


def test_tilted_drone_ground_still_found():
    body = Rotation.from_euler("y", 12, degrees=True).as_matrix()
    points_level = ground(3.0)
    tilt = heading_rotation(body).T @ body
    sensor = (points_level @ tilt) @ DOWN
    scan = extract_obstacles(
        sensor, DOWN, np.zeros(3), body, params(), np.random.default_rng(4)
    )
    assert scan.ground_mask.all()
    assert scan.ground.normal == pytest.approx([0, 0, 1], abs=1e-6)


def test_wall_is_not_taken_as_ground():
    rng = np.random.default_rng(5)
    yz = rng.uniform([-5, -3], [5, 3], (4000, 2))
    wall = np.column_stack([np.full(4000, 4.0), yz])  # more points than ground
    points = np.vstack([wall, ground(3.0, n=500)])
    plane = fit_ground_plane(points, params(), np.random.default_rng(6))
    # Wall base points within the threshold join the fit; offset moves by mm.
    assert plane.normal[2] > 0.99 and plane.offset == pytest.approx(3.0, abs=0.02)


def test_no_ground_when_too_few_points():
    assert (
        fit_ground_plane(ground(3.0, n=20), params(), np.random.default_rng(0)) is None
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("num_sectors", 2),
        ("band_below_m", -1.0),
        ("ransac_threshold_m", 0.0),
        ("ground_max_tilt_deg", 90.0),
        ("enabled", 1),
    ],
)
def test_invalid_params_rejected(key, value):
    raw = {"enabled": False, **params().__dict__, key: value}
    with pytest.raises(ValueError):
        validate_obstacle_params(raw)


def test_default_config_mapper_only_and_off():
    config = load_config(ROOT / "configs/simulation.yaml")
    raw = config["perception"]["l2_obstacles"]["mapping_drone"]
    assert raw["enabled"] is False
    assert load_obstacle_params(config["perception"], "mapping_drone").num_sectors == 72
    assert load_obstacle_params(config["perception"], "relay_drone") is None
    config["perception"]["l2_obstacles"]["relay_drone"] = raw
    with pytest.raises(ValueError):
        validate_config(config)


def overlay_scene(maxgeom=200):
    import mujoco

    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    return mujoco.MjvScene(model, maxgeom=maxgeom)


def l2_view():
    from tar_swarm.mujoco.overlays import L2View

    distance = np.full(72, np.nan)
    distance[0], distance[18] = 4.0, 0.5
    points = np.array([[4.0, 0, 0], [0, 0, -3.0], [1, 1, -3.0], [2, 2, -3.0]])
    yaw = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    return L2View(
        np.array([1.0, 2.0, 3.0]),
        yaw,
        points,
        np.array([False, True, True, True]),
        distance,
    )


def test_sector_ring_overlay_geoms():
    from tar_swarm.mujoco.overlays import CLEAR, DANGER, NO_RETURN, draw_sector_ring

    scn = overlay_scene()
    draw_sector_ring(scn, l2_view(), d_stop_m=1.0, no_return_length_m=0.3)
    assert scn.ngeom == 72
    # Sector 0 points along the heading (world +y here): 4 m, clear.
    assert np.allclose(scn.geoms[0].rgba, CLEAR)
    assert np.allclose(scn.geoms[0].pos, [1.0, 4.0, 3.0], atol=1e-6)
    # Sector 18 (90 deg left of heading = world -x): 0.5 m, inside d_stop.
    assert np.allclose(scn.geoms[18].rgba, DANGER)
    assert np.allclose(scn.geoms[18].pos, [0.75, 2.0, 3.0], atol=1e-6)
    assert np.allclose(scn.geoms[1].rgba, NO_RETURN)


def test_lidar_points_overlay_obstacles_first_and_budget():
    from tar_swarm.mujoco.overlays import (
        GROUND_POINT,
        OBSTACLE_POINT,
        draw_lidar_points,
    )

    scn = overlay_scene()
    draw_lidar_points(scn, l2_view(), max_points=2)
    assert scn.ngeom == 2
    assert np.allclose(scn.geoms[0].rgba, OBSTACLE_POINT)
    assert np.allclose(scn.geoms[0].pos, [1.0, 6.0, 3.0], atol=1e-6)
    assert np.allclose(scn.geoms[1].rgba, GROUND_POINT)
    full = overlay_scene(maxgeom=3)
    draw_lidar_points(full, l2_view(), max_points=10)
    assert full.ngeom == 3  # never past the scene's geom capacity
    hidden = overlay_scene()
    draw_lidar_points(hidden, l2_view(), max_points=10, show_ground=False)
    assert hidden.ngeom == 1  # only the obstacle point
    assert np.allclose(hidden.geoms[0].rgba, OBSTACLE_POINT)


def test_ground_points_hidden_by_default(tmp_path):
    from tar_swarm.mujoco.overlays import (
        draw_sensor_overlays,
        load_overlay_settings,
    )

    settings = load_overlay_settings(ROOT / "configs/overlays.yaml")
    assert settings["lidar_points"]["show_ground_points"] is False
    only_points = {"lidar_points": settings["lidar_points"]}
    scn = overlay_scene()
    draw_sensor_overlays(scn, only_points, {}, {"mapper_0": l2_view()})
    assert scn.ngeom == 1
    shown = {"lidar_points": {**settings["lidar_points"], "show_ground_points": True}}
    scn = overlay_scene()
    draw_sensor_overlays(scn, shown, {}, {"mapper_0": l2_view()})
    assert scn.ngeom == 4  # one obstacle point and three ground points
    bad = tmp_path / "overlays.yaml"
    bad.write_text(
        "lidar_points:\n  enabled: true\n  max_points: 10\n  show_ground_points: 1\n"
    )
    with pytest.raises(ValueError):
        load_overlay_settings(bad)


def test_runner_overlay_computes_l2_without_changing_outputs(tmp_path):
    from tar_swarm.mujoco.overlays import load_overlay_settings
    from tar_swarm.mujoco.runner import run_scenario

    config = load_config(ROOT / "configs/simulation.yaml")
    config["simulation"]["duration_s"] = 0.3
    overlay = load_overlay_settings(ROOT / "configs/overlays.yaml")
    plain = run_scenario(config, plots=False)
    with_overlay = run_scenario(config, plots=False, overlay=overlay)
    assert plain["rows"] == with_overlay["rows"]
    for a, b in zip(plain["lidar"], with_overlay["lidar"], strict=True):
        assert np.array_equal(a["points"], b["points"])
