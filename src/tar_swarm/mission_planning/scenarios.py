"""Example and curriculum scenario builders."""

from __future__ import annotations

from pathlib import Path

import yaml

from tar_swarm.mission_planning.schemas import (
    CapacityLimits,
    CommunicationConfig,
    CoordinateReference,
    DroneSpec,
    MissionRequest,
    PlanningLimits,
    Role,
    SensorConfig,
)


def load_request_yaml(path: str | Path) -> MissionRequest:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return MissionRequest.model_validate(data)


def _rect(xmin, ymin, xmax, ymax):
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


# Shared capacity envelope for curriculum/checkpoints (pad unused slots).
TRAIN_CAPACITIES = CapacityLimits(
    max_mappers=2,
    max_relays=3,
    max_candidates=16,
    max_segments_per_mapper=12,
    max_waypoints_per_segment=64,
)


def one_mapper_small(**overrides) -> MissionRequest:
    """Feasible small rectangle: 1 mapper + 2 relays."""
    caps = TRAIN_CAPACITIES
    req = MissionRequest(
        region_polygon_m=_rect(20, -20, 80, 20),
        coordinate_reference=CoordinateReference(kind="local_enu_m"),
        base_position_m=(0.0, 0.0, 0.0),
        drones=[
            DroneSpec(
                id="mapper_1",
                role=Role.MAPPER,
                launch_position_m=(10.0, 0.0, 15.0),
                speed_m_s=5.0,
                max_flight_time_s=1200.0,
                clearance_radius_m=1.5,
            ),
            DroneSpec(
                id="relay_1",
                role=Role.RELAY,
                launch_position_m=(15.0, 5.0, 15.0),
                speed_m_s=4.0,
                max_flight_time_s=1200.0,
                clearance_radius_m=0.5,
            ),
            DroneSpec(
                id="relay_2",
                role=Role.RELAY,
                launch_position_m=(15.0, -5.0, 15.0),
                speed_m_s=4.0,
                max_flight_time_s=1200.0,
                clearance_radius_m=0.5,
            ),
        ],
        sensor=SensorConfig(survey_altitude_m=15.0, swath_m=20.0, overlap=0.2),
        communication=CommunicationConfig(
            max_range_m=45.0, connection_margin_m=2.0
        ),
        planning=PlanningLimits(
            decision_dt_s=2.0,
            sim_dt_s=0.5,
            max_episode_time_s=600.0,
            candidate_spacing_m=20.0,
            segment_max_length_m=100.0,
            capacities=caps,
        ),
        seed=1,
    )
    return req.model_copy(update=overrides) if overrides else req


def multi_mapper(**overrides) -> MissionRequest:
    caps = TRAIN_CAPACITIES
    req = MissionRequest(
        region_polygon_m=_rect(20, -40, 100, 40),
        base_position_m=(0.0, 0.0, 0.0),
        drones=[
            DroneSpec(
                id="mapper_1",
                role=Role.MAPPER,
                launch_position_m=(10.0, -10.0, 15.0),
                speed_m_s=5.0,
                max_flight_time_s=1800.0,
            ),
            DroneSpec(
                id="mapper_2",
                role=Role.MAPPER,
                launch_position_m=(10.0, 10.0, 15.0),
                speed_m_s=5.0,
                max_flight_time_s=1800.0,
            ),
            DroneSpec(
                id="relay_1",
                role=Role.RELAY,
                launch_position_m=(15.0, 0.0, 15.0),
                speed_m_s=4.0,
                max_flight_time_s=1800.0,
                clearance_radius_m=0.5,
            ),
            DroneSpec(
                id="relay_2",
                role=Role.RELAY,
                launch_position_m=(20.0, 8.0, 15.0),
                speed_m_s=4.0,
                max_flight_time_s=1800.0,
                clearance_radius_m=0.5,
            ),
            DroneSpec(
                id="relay_3",
                role=Role.RELAY,
                launch_position_m=(20.0, -8.0, 15.0),
                speed_m_s=4.0,
                max_flight_time_s=1800.0,
                clearance_radius_m=0.5,
            ),
        ],
        sensor=SensorConfig(survey_altitude_m=15.0, swath_m=20.0, overlap=0.2),
        communication=CommunicationConfig(max_range_m=40.0, connection_margin_m=2.0),
        planning=PlanningLimits(
            decision_dt_s=2.0,
            sim_dt_s=0.5,
            max_episode_time_s=900.0,
            candidate_spacing_m=25.0,
            segment_max_length_m=120.0,
            capacities=caps,
        ),
        seed=2,
    )
    return req.model_copy(update=overrides) if overrides else req


def concave_region(**overrides) -> MissionRequest:
    """L-shaped concave polygon (supported; not replaced by bounding box)."""
    poly = [
        (20.0, -10.0),
        (80.0, -10.0),
        (80.0, 10.0),
        (50.0, 10.0),
        (50.0, 40.0),
        (20.0, 40.0),
    ]
    caps = TRAIN_CAPACITIES
    req = MissionRequest(
        region_polygon_m=poly,
        base_position_m=(0.0, 0.0, 0.0),
        drones=[
            DroneSpec(
                id="mapper_1",
                role=Role.MAPPER,
                launch_position_m=(10.0, 0.0, 15.0),
                speed_m_s=5.0,
                max_flight_time_s=1500.0,
            ),
            DroneSpec(
                id="relay_1",
                role=Role.RELAY,
                launch_position_m=(15.0, 5.0, 15.0),
                speed_m_s=4.0,
                max_flight_time_s=1500.0,
                clearance_radius_m=0.5,
            ),
            DroneSpec(
                id="relay_2",
                role=Role.RELAY,
                launch_position_m=(15.0, -5.0, 15.0),
                speed_m_s=4.0,
                max_flight_time_s=1500.0,
                clearance_radius_m=0.5,
            ),
        ],
        sensor=SensorConfig(survey_altitude_m=15.0, swath_m=18.0, overlap=0.15),
        communication=CommunicationConfig(max_range_m=45.0, connection_margin_m=2.0),
        planning=PlanningLimits(
            decision_dt_s=2.0,
            sim_dt_s=0.5,
            max_episode_time_s=700.0,
            candidate_spacing_m=20.0,
            segment_max_length_m=90.0,
            capacities=caps,
        ),
        seed=3,
    )
    return req.model_copy(update=overrides) if overrides else req


def failure_case_infeasible_range(**overrides) -> MissionRequest:
    """Deliberately difficult/infeasible: tiny radio range vs large region.

    Failure to find a plan does NOT prove physical infeasibility under real RF;
    it only shows the distance model cannot connect the configured inventory.
    """
    caps = CapacityLimits(
        max_mappers=2, max_relays=2, max_candidates=12, max_segments_per_mapper=8
    )
    req = MissionRequest(
        region_polygon_m=_rect(50, -30, 150, 30),
        base_position_m=(0.0, 0.0, 0.0),
        drones=[
            DroneSpec(
                id="mapper_1",
                role=Role.MAPPER,
                launch_position_m=(10.0, 0.0, 15.0),
                speed_m_s=5.0,
                max_flight_time_s=600.0,
            ),
            DroneSpec(
                id="relay_1",
                role=Role.RELAY,
                launch_position_m=(12.0, 2.0, 15.0),
                speed_m_s=4.0,
                max_flight_time_s=600.0,
                clearance_radius_m=0.5,
            ),
            DroneSpec(
                id="relay_2",
                role=Role.RELAY,
                launch_position_m=(12.0, -2.0, 15.0),
                speed_m_s=4.0,
                max_flight_time_s=600.0,
                clearance_radius_m=0.5,
            ),
        ],
        sensor=SensorConfig(survey_altitude_m=15.0, swath_m=20.0, overlap=0.2),
        communication=CommunicationConfig(max_range_m=25.0, connection_margin_m=5.0),
        planning=PlanningLimits(
            decision_dt_s=2.0,
            sim_dt_s=0.5,
            max_episode_time_s=400.0,
            candidate_spacing_m=30.0,
            segment_max_length_m=100.0,
            capacities=caps,
        ),
        seed=4,
    )
    return req.model_copy(update=overrides) if overrides else req


CURRICULUM = [
    ("one_mapper_small", one_mapper_small),
    ("multi_mapper", multi_mapper),
    ("concave_region", concave_region),
    ("failure_case", failure_case_infeasible_range),
]


def build_named(name: str) -> MissionRequest:
    mapping = {k: fn for k, fn in CURRICULUM}
    if name not in mapping:
        raise KeyError(f"Unknown scenario {name}; choose from {sorted(mapping)}")
    return mapping[name]()
