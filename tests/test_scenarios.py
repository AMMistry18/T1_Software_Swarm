"""AV4 test worlds, scenario builder and the ground-edge hard check."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from tar_swarm.common.config import load_config, read_yaml, validate_config
from tar_swarm.evaluation.scenarios import (
    ALTITUDE_M,
    LANE_END_X_M,
    SENSORS,
    WORLDS,
    avoidance_config,
    check_ground_edge,
    ground_edge_margin,
    lane_waypoints,
)
from tar_swarm.mujoco.flight import FlightSimulation
from tar_swarm.mujoco.runner import run_scenario

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("sensor", sorted(SENSORS))
def test_every_scenario_builds_and_passes_edge_check(world, sensor):
    config = avoidance_config(world, sensor)
    agent_id, vehicle, _ = SENSORS[sensor]
    assert [a["id"] for a in config["mission"]["agents"]] == [agent_id]
    assert config["avoidance"][vehicle]["enabled"]
    assert config["avoidance"][vehicle]["sources"] == [sensor]
    assert config["perception"]["l2_obstacles"]["mapping_drone"]["enabled"] == (
        sensor == "l2"
    )
    reach = LANE_END_X_M + 30.0 + 1.0  # lane end + L2 range + margin
    assert config["world"]["ground_half_size_m"] >= reach


def test_obstacles_sit_on_the_lane():
    for world in ("avoid_wall", "avoid_pole", "avoid_thin"):
        (obstacle,) = read_yaml(ROOT / f"configs/worlds/{world}.yaml")["obstacles"]
        face = obstacle["position_m"][0] - obstacle["size_m"][0] / 2
        assert face == pytest.approx(12.0)
        assert abs(obstacle["position_m"][1]) < 1e-9
    (pole,) = read_yaml(ROOT / "configs/worlds/avoid_offaxis.yaml")["obstacles"]
    assert pole["position_m"][1] == pytest.approx(0.3)
    (wire,) = read_yaml(ROOT / "configs/worlds/avoid_thin.yaml")["obstacles"]
    assert wire["position_m"][2] == pytest.approx(ALTITUDE_M)
    assert read_yaml(ROOT / "configs/worlds/clear_path.yaml")["obstacles"] == []


def test_cylinder_obstacle_in_scene():
    config = avoidance_config("avoid_pole", "rangefinder")
    sim = FlightSimulation(config)
    geom = sim.model.geom("pole")
    assert geom.type == 5  # mjGEOM_CYLINDER
    assert np.allclose(geom.size[:2], [0.05, 3.0])
    bad = deepcopy(config)
    bad["world"]["obstacles"][0]["size_m"] = [0.1, 0.2, 6]
    with pytest.raises(ValueError):
        validate_config(bad)
    bad["world"]["obstacles"][0]["shape"] = "cone"
    with pytest.raises(ValueError):
        validate_config(bad)


def test_default_world_obstacles_stay_boxes():
    config = load_config(ROOT / "configs/simulation.yaml")
    model = FlightSimulation(config).model
    for obstacle in config["world"]["obstacles"]:
        assert model.geom(obstacle["name"]).type == 6  # mjGEOM_BOX


def test_lane_waypoints_and_speed():
    points = lane_waypoints(0.05, 2.0)
    assert points[0]["position_m"] == [0.0, 0.0, 0.05]
    assert points[-2]["position_m"] == [LANE_END_X_M, 0.0, ALTITUDE_M]
    cruise = points[3]["time_s"] - points[2]["time_s"]
    assert 1.875 * LANE_END_X_M / cruise == pytest.approx(2.0, rel=0.01)
    assert 1.875 * LANE_END_X_M / cruise <= 2.0
    with pytest.raises(ValueError):
        avoidance_config("avoid_wall", "rangefinder", 0.0)


def test_noise_off_option():
    config = avoidance_config("clear_path", "l2", noise=False)
    assert config["sensors"]["lidar"]["mapping_drone"]["range_noise_m"] == 0.0
    luna = config["sensors"]["rangefinder"]["relay_drone"]
    assert luna["noise_sigma_near_m"] == luna["noise_sigma_far_fraction"] == 0.0


def test_ground_edge_check_known_answer():
    config = avoidance_config("clear_path", "rangefinder")
    config["world"]["ground_half_size_m"] = 50.0
    positions = np.array([[0.0, 0.0, 2.0], [15.0, -3.0, 2.0]])
    # 50 - (15 + 30 + 1) = 4
    assert ground_edge_margin(config, positions) == pytest.approx(4.0)
    check_ground_edge(config, positions)
    positions[1, 1] = -20.0  # 50 - (20 + 31) = -1: a ray could leave the ground
    with pytest.raises(ValueError):
        check_ground_edge(config, positions)


def test_small_world_rejected_before_running(tmp_path):
    world = read_yaml(ROOT / "configs/worlds/avoid_wall.yaml")
    world["ground_half_size_m"] = 40
    base = tmp_path / "configs"
    (base / "worlds").mkdir(parents=True)
    for name in ("simulation", "vehicles", "sensors", "perception", "avoidance"):
        (base / f"{name}.yaml").write_text(
            (ROOT / f"configs/{name}.yaml").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    (base / "missions").mkdir()
    (base / "missions/step0.yaml").write_text(
        (ROOT / "configs/missions/step0.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (base / "worlds/outdoor.yaml").write_text(
        (ROOT / "configs/worlds/outdoor.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    import yaml

    (base / "worlds/avoid_wall.yaml").write_text(yaml.safe_dump(world))
    with pytest.raises(ValueError, match="Ground too small"):
        avoidance_config("avoid_wall", "rangefinder", base=base / "simulation.yaml")


def test_relay_stops_at_wall_in_avoid_wall():
    config = avoidance_config("avoid_wall", "rangefinder", 2.0)
    result = run_scenario(config, plots=False)
    x = max(r["position_x_m"] for r in result["rows"])
    assert x < 12.0 - 0.5  # stopped well short of the wall face
    assert max(r["contacts"] for r in result["rows"][50:]) == 0
    positions = np.array(
        [[r[f"position_{a}_m"] for a in "xyz"] for r in result["rows"]]
    )
    assert ground_edge_margin(config, positions) > 0
