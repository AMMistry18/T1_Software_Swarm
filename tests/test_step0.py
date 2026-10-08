"""Step 0 physics and flight acceptance checks on actual MuJoCo trajectories."""

from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tar_swarm.common.config import load_config, validate_config
from tar_swarm.control.cascaded import CascadedController
from tar_swarm.control.interfaces import Reference, State
from tar_swarm.control.mission import WaypointMission
from tar_swarm.evaluation.logging import metrics
from tar_swarm.mujoco.flight import FlightSimulation
from tar_swarm.mujoco.runner import run_scenario
from tar_swarm.vehicles.model import allocation_matrix, mass_properties

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return load_config(
        ROOT / "configs/simulation.yaml", ROOT / "configs/worlds/empty.yaml"
    )


@pytest.fixture(scope="module")
def full_run():
    config = load_config(
        ROOT / "configs/simulation.yaml", ROOT / "configs/worlds/empty.yaml"
    )
    return config, run_scenario(config, plots=False)


@pytest.mark.parametrize("world", ["empty", "outdoor"])
def test_scene_load_and_initialization(world):
    config = load_config(
        ROOT / "configs/simulation.yaml", ROOT / f"configs/worlds/{world}.yaml"
    )
    sim = FlightSimulation(config)
    assert sim.model.nq == 28
    assert sim.model.nv == 24
    assert sim.model.nu == 16
    assert sim.model.geom("ground_station").id >= 0
    assert sim.data.ncon == 0
    for agent in config["mission"]["agents"]:
        name = agent["id"]
        np.testing.assert_allclose(sim.state(name).position, agent["spawn_m"])
        np.testing.assert_allclose(sim.state(name).velocity, 0)
        assert sim.model.joint(f"{name}_free").type[0] == mujoco.mjtJoint.mjJNT_FREE
        handle = sim.handles[name]
        for i, actuator_id in enumerate(handle.actuator_ids):
            assert sim.model.actuator(actuator_id).name == f"{name}_motor_{i}"
            site_id = sim.model.site(f"{name}_rotor_{i}").id
            assert sim.model.actuator_trnid[actuator_id, 0] == site_id
        assert sim.model.actuator_ctrllimited[handle.actuator_ids].all()


@pytest.mark.parametrize("vehicle_key", ["mapping_drone", "relay_drone"])
def test_mass_inertia_and_hover(config, vehicle_key):
    sim = FlightSimulation(config)
    name = "mapper_0" if vehicle_key == "mapping_drone" else "relay_0"
    vehicle = config["vehicles"][vehicle_key]
    mass, _, expected = mass_properties(vehicle)
    body_id = sim.handles[name].body_id
    assert mass == pytest.approx(
        sum(c["mass_kg"] for c in vehicle["components"])
        + 4
        * sum(vehicle[k] for k in ("motor_mass_kg", "propeller_mass_kg", "arm_mass_kg"))
    )
    assert sim.model.body_mass[body_id] == pytest.approx(mass)
    np.testing.assert_allclose(sim.model.body_ipos[body_id], 0, atol=1e-10)
    quat = sim.model.body_iquat[body_id]
    rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    compiled = rotation @ np.diag(sim.model.body_inertia[body_id]) @ rotation.T
    np.testing.assert_allclose(compiled, expected, rtol=1e-7, atol=1e-10)
    assert np.linalg.eigvalsh(compiled).min() > 0
    controller = CascadedController(vehicle, 9.81)
    assert controller.hover_thrust == pytest.approx(mass * 9.81 / 4)
    state = State(np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3))
    reference = Reference(np.zeros(3), np.zeros(3), np.zeros(3))
    output = controller.update(state, reference)
    np.testing.assert_allclose(
        allocation_matrix(vehicle) @ output.thrusts, [mass * 9.81, 0, 0, 0], atol=1e-10
    )


def test_gravity_and_unpowered_step(config):
    for agent in config["mission"]["agents"]:
        agent["spawn_m"][2] = 5
        agent["waypoints"][0]["position_m"][2] = 5
    sim = FlightSimulation(config)
    for _ in range(50):
        sim.step()
    assert sim.data.time == pytest.approx(0.1)
    for name in sim.handles:
        state = sim.state(name)
        assert state.position[2] == pytest.approx(5 - 0.5 * 9.81 * 0.1**2, abs=1e-6)
        assert state.velocity[2] == pytest.approx(-9.81 * 0.1, abs=1e-6)


@pytest.mark.parametrize("axis", range(4))
def test_rotor_wrench_direction(config, axis):
    config["mission"]["agents"] = [config["mission"]["agents"][0]]
    config["mission"]["agents"][0]["spawn_m"] = [0, 0, 5]
    sim = FlightSimulation(config)
    vehicle = config["vehicles"]["mapping_drone"]
    controller = CascadedController(vehicle, 9.81)
    wrench = np.array([controller.mass * 9.81, 0.0, 0.0, 0.0])
    wrench[axis] += 0.1 if axis else 1.0
    sim.command("mapper_0", controller.inverse_mix @ wrench)
    for _ in range(5):
        sim.step()
    state = sim.state("mapper_0")
    if axis == 0:
        assert state.velocity[2] > 0
    else:
        assert state.angular_velocity[axis - 1] > 0


@pytest.mark.parametrize("vehicle_key", ["mapping_drone", "relay_drone"])
def test_controller_and_actuator_bounds(config, vehicle_key):
    vehicle = config["vehicles"][vehicle_key]
    controller = CascadedController(vehicle, 9.81)
    state = State(np.zeros(3), np.zeros(3), np.eye(3), np.array([100, -100, 100]))
    target = Reference(np.array([100, 100, 100]), np.zeros(3), np.zeros(3))
    output = controller.update(state, target)
    assert output.motor_saturated and output.acceleration_limited
    assert np.isfinite(output.thrusts).all()
    assert min(output.thrusts) >= 0
    assert max(output.thrusts) <= vehicle["max_rotor_thrust_n"]
    sim = FlightSimulation(config)
    name = "mapper_0" if vehicle_key == "mapping_drone" else "relay_0"
    ids = sim.handles[name].actuator_ids
    sim.command(name, np.array([-100, 100, -100, 100]))
    sim.step()
    assert min(sim.data.actuator_force[ids]) >= 0
    assert max(sim.data.actuator_force[ids]) <= vehicle["max_rotor_thrust_n"]


def test_hover_and_waypoint_acceptance(full_run):
    config, result = full_run
    report = metrics(result["rows"], config)
    assert result["completed"]
    assert result["simulation"].data.time == pytest.approx(36)
    for values in report.values():
        assert values["hover_altitude_rmse_m"] < 0.05
        assert values["final_hover_rmse_m"] < 0.05
        assert values["position_rmse_m"] < 0.15
        assert values["max_position_error_m"] < 0.30
    # Test visits, not just returning to the initial position at the end.
    for point in config["mission"]["agents"][0]["waypoints"][3:]:
        row = min(
            (r for r in result["rows"] if r["vehicle_id"] == "mapper_0"),
            key=lambda r: abs(r["time_s"] - point["time_s"]),
        )
        pos = np.array([row[f"position_{a}_m"] for a in "xyz"])
        assert np.linalg.norm(pos - point["position_m"]) < 0.15
    assert all(row["contacts"] == 0 for row in result["rows"])


@pytest.mark.parametrize("vehicle_key", ["mapping_drone", "relay_drone"])
def test_hover_recovers_from_pose_velocity_disturbance(config, vehicle_key):
    agent = next(a for a in config["mission"]["agents"] if a["vehicle"] == vehicle_key)
    agent["spawn_m"] = [0, 0, 3]
    config["mission"]["agents"] = [agent]
    sim = FlightSimulation(config)
    sim.data.qpos[:3] += [0.3, -0.2, 0.2]
    quat = Rotation.from_euler("xyz", [0.12, -0.1, 0.25]).as_quat()
    sim.data.qpos[3:7] = quat[[3, 0, 1, 2]]
    sim.data.qvel[:3] = [0.2, -0.1, 0.1]
    mujoco.mj_forward(sim.model, sim.data)
    controller = CascadedController(config["vehicles"][vehicle_key], 9.81)
    target = Reference(np.array([0, 0, 3]), np.zeros(3), np.zeros(3), yaw=0.5)
    for tick in range(4000):
        if tick % 5 == 0:
            sim.command(
                agent["id"], controller.update(sim.state(agent["id"]), target).thrusts
            )
        sim.step()
    state = sim.state(agent["id"])
    assert np.linalg.norm(state.position - target.position) < 0.03
    assert np.linalg.norm(state.velocity) < 0.03
    assert np.linalg.norm(state.angular_velocity) < 0.03
    assert Rotation.from_matrix(state.rotation).as_euler("xyz")[2] == pytest.approx(
        0.5, abs=0.02
    )


def test_determinism_and_log_timing(config, tmp_path):
    config["simulation"]["duration_s"] = 2
    first = run_scenario(config, plots=False, output=tmp_path / "first")
    second = run_scenario(config, plots=False)
    np.testing.assert_array_equal(
        first["simulation"].data.qpos, second["simulation"].data.qpos
    )
    assert first["rows"] == second["rows"]
    assert first["controls"] == second["controls"]
    times = [r["time_s"] for r in first["rows"] if r["vehicle_id"] == "mapper_0"]
    np.testing.assert_allclose(np.diff(times), 0.02, atol=1e-10)
    assert (tmp_path / "first/states.csv").is_file()
    assert (tmp_path / "first/controls.csv").is_file()
    assert (tmp_path / "first/scene.xml").is_file()
    assert first["metrics"]["run"]["completed"]


def test_agent_count_configurable(config):
    extra = deepcopy(config["mission"]["agents"][0])
    extra["id"] = "mapper_1"
    extra["spawn_m"] = [-6, 0, 0.23]
    for point in extra["waypoints"]:
        point["position_m"][0] -= 6
    config["mission"]["agents"].append(extra)
    sim = FlightSimulation(config)
    assert len(sim.handles) == 5
    assert sim.model.nu == 20


def test_quintic_reference_continuity(config):
    mission = WaypointMission(config["mission"]["agents"][0])
    for t in mission.times:
        left, right = mission.sample(t - 1e-6), mission.sample(t + 1e-6)
        np.testing.assert_allclose(left.position, right.position, atol=1e-5)
        np.testing.assert_allclose(left.velocity, right.velocity, atol=1e-5)
        np.testing.assert_allclose(left.acceleration, right.acceleration, atol=1e-5)


@pytest.mark.parametrize("asset", ["mapper", "relay"])
def test_exported_vehicle_models_load(asset):
    model = mujoco.MjModel.from_xml_path(str(ROOT / f"assets/mujoco/{asset}.xml"))
    assert model.nq == 7
    assert model.nu == 4


def test_downward_acceleration_has_finite_attitude(config):
    vehicle = config["vehicles"]["relay_drone"]
    vehicle["controller"]["max_acceleration_m_s2"] = 20
    controller = CascadedController(vehicle, 9.81)
    state = State(np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3))
    reference = Reference(np.zeros(3), np.zeros(3), np.array([0, 0, -9.81]))
    output = controller.update(state, reference)
    assert np.isfinite(output.thrusts).all()
    assert output.acceleration_limited


def test_output_directory_is_protected(config, tmp_path):
    with pytest.raises(FileExistsError):
        run_scenario(config, output=tmp_path)


@pytest.mark.parametrize("invalid", ["clock", "props", "mass", "waypoints"])
def test_invalid_configuration_rejected(config, invalid):
    if invalid == "clock":
        config["simulation"]["controller_interval_s"] = 0.009
    elif invalid == "props":
        config["vehicles"]["mapping_drone"]["arm_length_m"] = 0.1
    elif invalid == "mass":
        config["vehicles"]["relay_drone"]["components"][0]["mass_kg"] = -1
    else:
        config["mission"]["agents"][0]["waypoints"][1]["time_s"] = 0
    with pytest.raises(ValueError):
        validate_config(config)
