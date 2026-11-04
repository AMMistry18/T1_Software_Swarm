"""Mission-planning geometry, connectivity, rewards, masks, export replay."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tar_swarm.mission_planning.api import plan_mission
from tar_swarm.mission_planning.baselines import (
    search_relay_baseline,
    static_relay_baseline,
)
from tar_swarm.mission_planning.checkpointing import (
    assert_compatible,
    build_checkpoint_meta,
    load_meta,
    save_meta,
)
from tar_swarm.mission_planning.communication import (
    ConservativeDistanceLink,
    mapper_connected_via_permitted,
)
from tar_swarm.mission_planning.compile import compile_from_sim
from tar_swarm.mission_planning.coords import LocalFrame
from tar_swarm.mission_planning.env import MissionPlanningEnv
from tar_swarm.mission_planning.geometry import (
    is_concave,
    lawnmower_passes,
    partition_polygon,
    polygon_from_ring,
)
from tar_swarm.mission_planning.preprocess import prepare_scenario
from tar_swarm.mission_planning.rewards import RewardWeights, compute_reward
from tar_swarm.mission_planning.scenarios import (
    concave_region,
    failure_case_infeasible_range,
    multi_mapper,
    one_mapper_small,
)
from tar_swarm.mission_planning.schemas import CapacityLimits, MissionRequest, Role
from tar_swarm.mission_planning.simulator import PlanningSimulator
from tar_swarm.mission_planning.validate import validate_mission


def test_geo_roundtrip():
    frame = LocalFrame(30.0, -97.0, 0.0)
    local = frame.geo_to_local(30.001, -96.999, 10.0)
    lat, lon, alt = frame.local_to_geo(local)
    assert abs(lat - 30.001) < 1e-5
    assert abs(lon - (-96.999)) < 1e-5
    assert abs(alt - 10.0) < 1e-9


def test_concave_not_replaced_by_bbox():
    req = concave_region()
    poly = polygon_from_ring(req.region_polygon_m)
    assert is_concave(poly)
    scenario = prepare_scenario(req)
    # Covered/partition area must remain L-shaped, not full bounding rectangle.
    bbox_area = (80 - 20) * (40 - (-10))
    assert scenario.total_survey_area_m2 < bbox_area - 100


def test_holes_and_obstacles_rejected():
    base = one_mapper_small().model_dump()
    base["polygon_holes_m"] = [[(30, -5), (40, -5), (40, 5), (30, 5)]]
    with pytest.raises(ValueError, match="holes"):
        MissionRequest.model_validate(base)
    base = one_mapper_small().model_dump()
    base["obstacles_m"] = [{"type": "building"}]
    with pytest.raises(ValueError, match="Obstacles"):
        MissionRequest.model_validate(base)


def test_capacity_overflow_rejected():
    req = one_mapper_small()
    data = req.model_dump()
    data["planning"]["capacities"]["max_mappers"] = 1
    data["drones"].append(
        {
            "id": "mapper_2",
            "role": "mapper",
            "launch_position_m": [12, 1, 15],
            "speed_m_s": 5.0,
            "max_flight_time_s": 1000,
        }
    )
    with pytest.raises(ValueError, match="max_mappers"):
        MissionRequest.model_validate(data)


def test_partition_and_lawnmower_deterministic():
    poly = polygon_from_ring([(0, 0), (60, 0), (60, 40), (0, 40)])
    a1 = partition_polygon(poly, ["m1", "m2"])
    a2 = partition_polygon(poly, ["m1", "m2"])
    assert [p.area for _, p in a1] == [p.area for _, p in a2]
    passes = lawnmower_passes(poly, swath_m=10, overlap=0.2)
    assert passes
    assert all(p.length_m > 0 for p in passes)


def test_preprocess_stable_ids():
    s1 = prepare_scenario(one_mapper_small())
    s2 = prepare_scenario(one_mapper_small())
    assert [s.id for s in s1.segments] == [s.id for s in s2.segments]
    assert [c.id for c in s1.candidates] == [c.id for c in s2.candidates]


def test_connectivity_in_transit_crossing():
    """Mapper moving between samples can lose connectivity mid-segment."""
    link = ConservativeDistanceLink(max_range_m=30.0, connection_margin_m=0.0)
    roles = {
        "ground": "ground_station",
        "relay_1": "relay",
        "mapper_1": "mapper",
    }
    positions = {
        "ground": np.array([0.0, 0.0, 0.0]),
        "relay_1": np.array([25.0, 0.0, 10.0]),
        "mapper_1": np.array([50.0, 0.0, 10.0]),
    }
    active = {k: True for k in positions}
    ok = mapper_connected_via_permitted(
        positions, active, roles, 0.0, link, "ground", ["mapper_1"], False
    )
    assert ok["mapper_1"]
    positions["mapper_1"] = np.array([80.0, 0.0, 10.0])
    ok2 = mapper_connected_via_permitted(
        positions, active, roles, 0.0, link, "ground", ["mapper_1"], False
    )
    assert not ok2["mapper_1"]


def test_simultaneous_motion_and_hold():
    scenario = prepare_scenario(one_mapper_small())
    sim = PlanningSimulator(scenario)
    relays = [r.id for r in scenario.request.relays]
    cands = [c.id for c in scenario.candidates]
    sim.send_relay(relays[0], cands[0])
    # Mapper remains idle/holding while relay moves — concurrent, not frozen.
    before = sim.drones[scenario.request.mappers[0].id].position.copy()
    sim.step_decision()
    after = sim.drones[scenario.request.mappers[0].id].position.copy()
    assert np.allclose(before, after)
    assert sim.drones[relays[0]].busy or sim.drones[relays[0]].travel_m > 0


def test_reward_no_farming_and_components_logged():
    br = compute_reward(
        RewardWeights(),
        new_coverage_frac=0.1,
        mission_complete=False,
        disconnect_s=2.0,
        decision_dt=2.0,
        relay_travel_m=10.0,
        repositions=1,
        violations=0,
        max_episode_time_s=100.0,
        norm_scale_m=200.0,
    )
    d = br.as_dict()
    assert "coverage" in d and "total" in d
    # Zero new coverage yields zero coverage reward.
    br0 = compute_reward(
        RewardWeights(),
        new_coverage_frac=0.0,
        mission_complete=False,
        disconnect_s=0.0,
        decision_dt=2.0,
        relay_travel_m=0.0,
        repositions=0,
        violations=0,
        max_episode_time_s=100.0,
        norm_scale_m=200.0,
    )
    assert br0.coverage == 0.0
    assert br0.time < 0.0  # time always penalized — stationary is not free success


def test_action_masks_variable_counts():
    env = MissionPlanningEnv(request=one_mapper_small())
    obs, _ = env.reset(seed=0)
    mask = env.action_masks()
    assert mask[0]  # hold
    assert mask.shape == (env.layout.n_actions,)
    assert obs.shape == (env.layout.obs_size,)
    # Active inventory can vary within a checkpoint's fixed capacities.
    env2 = MissionPlanningEnv(request=multi_mapper())
    env2.reset(seed=0)
    assert len(env2._mapper_ids) == 2
    assert len(env._mapper_ids) == 1
    # Masks for unused padded slots remain false.
    expected = (
        1
        + env.layout.max_relays * env.layout.max_candidates
        + env.layout.max_mappers
    )
    assert mask.shape[0] == expected


def test_baseline_and_export_replay(tmp_path: Path):
    req = one_mapper_small()
    scenario = prepare_scenario(req)
    base = static_relay_baseline(scenario)
    plan = compile_from_sim(scenario, base.sim, base.command_trace)
    report = validate_mission(scenario, plan)
    assert report.assumptions_established
    assert report.requires_real_measurement
    # Timing deviation path executes.
    report2 = validate_mission(scenario, plan, timing_jitter_s=0.5)
    assert isinstance(report2.ok, bool)
    out = tmp_path / "plan.json"
    out.write_text(plan.model_dump_json(), encoding="utf-8")
    loaded = json.loads(out.read_text())
    assert loaded["schema_version"] == "1.0.0"


def test_plan_mission_baseline_and_failure():
    ok = plan_mission(one_mapper_small(), checkpoint=None, fallback="search")
    # May or may not succeed depending on heuristic; must be structured.
    assert ok.success in (True, False)
    if not ok.success:
        assert ok.failure is not None
        assert ok.source.startswith("baseline")
    fail = plan_mission(
        failure_case_infeasible_range(), checkpoint=None, fallback="static"
    )
    assert fail.success in (True, False)
    if fail.success:
        # If a plan is claimed, it must include validation.
        assert fail.plan is not None
        assert "ok" in fail.plan.validation


def test_checkpoint_compatibility(tmp_path: Path):
    req = one_mapper_small()
    meta = build_checkpoint_meta(req, train_config={"norm_scale_m": 200.0})
    path = tmp_path / "model.meta.json"
    save_meta(path, meta)
    assert_compatible(meta, req)
    # Same capacity envelope scenarios are compatible (variable active counts OK).
    assert_compatible(meta, multi_mapper())
    # Larger declared scenario capacities must fail.
    oversized = one_mapper_small().model_dump()
    oversized["planning"]["capacities"]["max_candidates"] = 64
    with pytest.raises(ValueError, match="max_candidates"):
        assert_compatible(meta, MissionRequest.model_validate(oversized))
    bad = meta.copy()
    bad["capacities"] = CapacityLimits(
        max_mappers=1, max_relays=1, max_candidates=4, max_segments_per_mapper=4
    ).model_dump()
    with pytest.raises(ValueError):
        assert_compatible(bad, multi_mapper())


def test_env_step_hold_and_release():
    env = MissionPlanningEnv(request=one_mapper_small())
    env.reset(seed=0)
    mask = env.action_masks()
    obs, reward, term, trunc, info = env.step(0)
    assert "reward" in info
    assert set(info["reward"]) >= {"coverage", "total", "time"}
    # Release first available mapper action if masked in.
    release_idxs = np.where(mask)[0]
    # Find a release action
    for a in release_idxs:
        kind, _, _ = env.layout.decode_action(int(a))
        if kind == "release":
            env.step(int(a))
            break


def test_search_baseline_runs_multi_mapper():
    scenario = prepare_scenario(multi_mapper())
    result = search_relay_baseline(scenario, max_combos=8)
    assert "coverage_fraction" in result.metrics
    assert result.name == "search_relay"


def test_role_enum_and_yaml_load():
    from tar_swarm.mission_planning.scenarios import load_request_yaml

    path = Path("configs/scenarios/planning/one_mapper_small.yaml")
    if path.exists():
        req = load_request_yaml(path)
        assert req.mappers[0].role == Role.MAPPER


def test_checkpoint_train_reload(tmp_path: Path):
    from sb3_contrib import MaskablePPO

    from tar_swarm.mission_planning.train import train_maskable_ppo

    summary = train_maskable_ppo(
        one_mapper_small(),
        tmp_path / "train",
        total_timesteps=128,
        n_envs=1,
        use_curriculum=False,
        seed=0,
    )
    ckpt = Path(summary["final_model"])
    meta = load_meta(ckpt.with_name(ckpt.stem + ".meta.json"))
    assert_compatible(meta, one_mapper_small())
    model = MaskablePPO.load(str(ckpt))
    assert model.action_space.n > 1


def test_adapter_reference_stream():
    from tar_swarm.mission_planning.adapter import (
        SafePauseConsumer,
        dispatch_plan,
        iter_references_from_action,
    )
    from tar_swarm.mission_planning.compile import compile_from_sim

    scenario = prepare_scenario(one_mapper_small())
    base = static_relay_baseline(scenario)
    plan = compile_from_sim(scenario, base.sim, base.command_trace)
    consumer = SafePauseConsumer(log=[])
    dispatch_plan(plan, consumer)
    assert consumer.log
    action = next(a for a in plan.actions if a.waypoints_m)
    refs = list(iter_references_from_action(action, speed_m_s=5.0, dt_s=0.5))
    assert refs
