"""Validated, versioned schemas for mission planning I/O."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tar_swarm.mission_planning.versions import SCHEMA_VERSION


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Role(str, Enum):
    MAPPER = "mapper"
    RELAY = "relay"
    GROUND_STATION = "ground_station"


class CoordinateReference(StrictModel):
    """Either local ENU meters or WGS84 geographic with a declared local origin."""

    kind: Literal["local_enu_m", "wgs84"]
    origin_lat_deg: float | None = None
    origin_lon_deg: float | None = None
    origin_alt_m: float = 0.0

    @model_validator(mode="after")
    def _check_origin(self):
        if self.kind == "wgs84":
            if self.origin_lat_deg is None or self.origin_lon_deg is None:
                raise ValueError("WGS84 reference requires origin_lat_deg/lon_deg")
        return self


class DroneSpec(StrictModel):
    id: str
    role: Role
    launch_position_m: tuple[float, float, float]
    speed_m_s: float = Field(gt=0)
    max_flight_time_s: float = Field(gt=0)
    clearance_radius_m: float = Field(gt=0, default=1.0)

    @field_validator("id")
    @classmethod
    def _nonempty_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("drone id must be nonempty")
        return value


class SensorConfig(StrictModel):
    survey_altitude_m: float = Field(gt=0)
    swath_m: float = Field(gt=0)
    overlap: float = Field(ge=0.0, lt=1.0, default=0.2)


class CommunicationConfig(StrictModel):
    """Conservative distance-based connectivity assumption (not measured RF)."""

    max_range_m: float = Field(gt=0)
    connection_margin_m: float = Field(ge=0.0, default=2.0)
    # Roles allowed as intermediate forwarding nodes (base always included).
    forwarding_roles: tuple[Role, ...] = (Role.RELAY,)
    allow_mapper_forwarding: bool = False

    @property
    def usable_range_m(self) -> float:
        return max(0.0, self.max_range_m - self.connection_margin_m)


class CapacityLimits(StrictModel):
    max_mappers: int = Field(ge=1, le=8, default=2)
    max_relays: int = Field(ge=0, le=8, default=3)
    max_candidates: int = Field(ge=1, le=64, default=16)
    max_segments_per_mapper: int = Field(ge=1, le=64, default=12)
    max_waypoints_per_segment: int = Field(ge=2, le=256, default=64)


class PlanningLimits(StrictModel):
    decision_dt_s: float = Field(gt=0, default=2.0)
    sim_dt_s: float = Field(gt=0, default=0.5)
    max_episode_time_s: float = Field(gt=0, default=900.0)
    arrival_tolerance_m: float = Field(gt=0, default=1.0)
    separation_m: float = Field(gt=0, default=3.0)
    require_return_to_base: bool = True
    candidate_spacing_m: float = Field(gt=0, default=25.0)
    segment_max_length_m: float = Field(gt=0, default=120.0)
    capacities: CapacityLimits = Field(default_factory=CapacityLimits)


class MissionRequest(StrictModel):
    """User-supplied pre-deployment mission configuration."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    region_polygon_m: list[tuple[float, float]]
    coordinate_reference: CoordinateReference = Field(
        default_factory=lambda: CoordinateReference(kind="local_enu_m")
    )
    base_position_m: tuple[float, float, float]
    drones: list[DroneSpec]
    sensor: SensorConfig
    communication: CommunicationConfig
    planning: PlanningLimits = Field(default_factory=PlanningLimits)
    seed: int = Field(ge=0, default=0)
    # Explicit unsupported features: holes/obstacles/terrain must not be silent.
    polygon_holes_m: list[list[tuple[float, float]]] = Field(default_factory=list)
    obstacles_m: list[Any] = Field(default_factory=list)

    @field_validator("region_polygon_m")
    @classmethod
    def _polygon_size(cls, value):
        if len(value) < 3:
            raise ValueError("region_polygon_m needs at least 3 vertices")
        return value

    @model_validator(mode="after")
    def _validate_inventory(self):
        if self.polygon_holes_m:
            raise ValueError(
                "Polygon holes are not supported in preprocess v1; omit holes "
                "or preprocess externally and supply a simple outer ring."
            )
        if self.obstacles_m:
            raise ValueError(
                "Obstacles/terrain features are not supported in preprocess v1; "
                "rejecting rather than silently ignoring them."
            )
        ids = [d.id for d in self.drones]
        if len(set(ids)) != len(ids):
            raise ValueError("drone IDs must be unique")
        mappers = [d for d in self.drones if d.role == Role.MAPPER]
        relays = [d for d in self.drones if d.role == Role.RELAY]
        if not mappers:
            raise ValueError("at least one mapper is required")
        caps = self.planning.capacities
        if len(mappers) > caps.max_mappers:
            raise ValueError(
                f"mapper count {len(mappers)} exceeds capacity max_mappers="
                f"{caps.max_mappers}; refusing to truncate"
            )
        if len(relays) > caps.max_relays:
            raise ValueError(
                f"relay count {len(relays)} exceeds capacity max_relays="
                f"{caps.max_relays}; refusing to truncate"
            )
        altitudes = {d.launch_position_m[2] for d in self.drones}
        if len(altitudes) > 1 and abs(max(altitudes) - min(altitudes)) > 1e-6:
            # Mixed launch altitudes are allowed; survey altitude is separate.
            pass
        for drone in self.drones:
            if drone.role == Role.GROUND_STATION:
                raise ValueError("ground station is supplied via base_position_m")
        return self

    @property
    def mappers(self) -> list[DroneSpec]:
        return [d for d in self.drones if d.role == Role.MAPPER]

    @property
    def relays(self) -> list[DroneSpec]:
        return [d for d in self.drones if d.role == Role.RELAY]


class SurveySegment(StrictModel):
    id: str
    mapper_id: str
    waypoints_m: list[tuple[float, float, float]]
    length_m: float
    coverage_area_m2: float
    is_transit: bool = False


class CandidateRelay(StrictModel):
    id: str
    position_m: tuple[float, float, float]


class PartitionRegion(StrictModel):
    mapper_id: str
    polygon_m: list[tuple[float, float]]
    area_m2: float


class PreparedScenario(StrictModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    preprocess_version: str
    request: MissionRequest
    partitions: list[PartitionRegion]
    segments: list[SurveySegment]
    candidates: list[CandidateRelay]
    total_survey_area_m2: float
    coverage_gap_area_m2: float
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _capacity_bounds(self):
        caps = self.request.planning.capacities
        mappers = self.request.mappers
        if len(self.candidates) > caps.max_candidates:
            raise ValueError(
                f"candidate count {len(self.candidates)} exceeds max_candidates="
                f"{caps.max_candidates}; increase capacity or spacing"
            )
        by_mapper: dict[str, int] = {}
        for seg in self.segments:
            by_mapper[seg.mapper_id] = by_mapper.get(seg.mapper_id, 0) + 1
        for mapper in mappers:
            count = by_mapper.get(mapper.id, 0)
            if count > caps.max_segments_per_mapper:
                raise ValueError(
                    f"mapper {mapper.id} has {count} segments exceeding "
                    f"max_segments_per_mapper={caps.max_segments_per_mapper}"
                )
        return self


class ActionKind(str, Enum):
    HOLD = "hold"
    TRANSIT = "transit"
    SURVEY = "survey"
    RELEASE = "release"


class WaypointAction(StrictModel):
    action_id: str
    drone_id: str
    kind: ActionKind
    waypoints_m: list[tuple[float, float, float]]
    hold_s: float = 0.0
    depends_on: tuple[str, ...] = ()
    completion_tolerance_m: float = 1.0
    timeout_s: float | None = None
    expected_duration_s: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MissionPlan(StrictModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    mission_id: str
    actions: list[WaypointAction]
    expected_duration_s: float
    expected_coverage_fraction: float
    validation: dict[str, Any] = Field(default_factory=dict)
    execution_contract: dict[str, Any] = Field(default_factory=dict)


class FailureDiagnostics(StrictModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class PlanningResult(StrictModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    success: bool
    plan: MissionPlan | None = None
    failure: FailureDiagnostics | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    source: Literal["rl", "baseline_search", "baseline_static", "unknown"] = "unknown"

    @model_validator(mode="after")
    def _xor(self):
        if self.success and self.plan is None:
            raise ValueError("successful PlanningResult requires a plan")
        if not self.success and self.failure is None:
            raise ValueError("failed PlanningResult requires failure diagnostics")
        if self.success and self.failure is not None:
            raise ValueError("successful PlanningResult must not include failure")
        return self


def as_numpy_polygon(points: list[tuple[float, float]]) -> np.ndarray:
    return np.asarray(points, dtype=float)
