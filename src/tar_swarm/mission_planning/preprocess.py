"""Deterministic preprocessing: partitions, segments, candidates."""

from __future__ import annotations

from tar_swarm.mission_planning.geometry import (
    candidate_relay_positions,
    estimate_coverage_gap,
    group_passes_into_segments,
    is_concave,
    lawnmower_passes,
    partition_polygon,
    polygon_from_ring,
)
from tar_swarm.mission_planning.schemas import (
    CandidateRelay,
    MissionRequest,
    PartitionRegion,
    PreparedScenario,
    SurveySegment,
)
from tar_swarm.mission_planning.versions import PREPROCESS_VERSION


def prepare_scenario(request: MissionRequest) -> PreparedScenario:
    """Build a PreparedScenario or raise ValueError on unsupported/malformed input."""
    # MissionRequest already rejects holes/obstacles and over-capacity inventories.
    poly = polygon_from_ring(request.region_polygon_m)
    warnings: list[str] = []
    if is_concave(poly):
        warnings.append(
            "Concave polygon accepted; strip partitioning may discard secondary "
            "connected components (recorded in coverage_gap_area_m2)."
        )

    mapper_ids = [m.id for m in request.mappers]
    partitions_raw = partition_polygon(poly, mapper_ids)
    partitions: list[PartitionRegion] = []
    all_passes = []
    segments: list[SurveySegment] = []
    altitude = request.sensor.survey_altitude_m

    for mapper_id, region in partitions_raw:
        coords = list(region.exterior.coords[:-1])
        partitions.append(
            PartitionRegion(
                mapper_id=mapper_id,
                polygon_m=[(float(x), float(y)) for x, y in coords],
                area_m2=float(region.area),
            )
        )
        passes = lawnmower_passes(
            region,
            swath_m=request.sensor.swath_m,
            overlap=request.sensor.overlap,
            angle_deg=0.0,
        )
        all_passes.extend(passes)
        if not passes:
            raise ValueError(
                f"No survey passes generated for mapper {mapper_id}; "
                "region may be smaller than effective swath"
            )
        segs = group_passes_into_segments(
            passes,
            mapper_id=mapper_id,
            altitude_m=altitude,
            max_length_m=request.planning.segment_max_length_m,
            swath_m=request.sensor.swath_m,
        )
        for seg in segs:
            max_wp = request.planning.capacities.max_waypoints_per_segment
            if len(seg["waypoints_m"]) > max_wp:
                raise ValueError(
                    f"Segment {seg['id']} has {len(seg['waypoints_m'])} waypoints "
                    f"exceeding max_waypoints_per_segment="
                    f"{request.planning.capacities.max_waypoints_per_segment}; "
                    "reduce segment_max_length_m or increase capacity"
                )
            segments.append(SurveySegment(**seg))

    # Stable ordering: by mapper request order, then segment id.
    mapper_order = {m.id: i for i, m in enumerate(request.mappers)}
    segments.sort(key=lambda s: (mapper_order[s.mapper_id], s.id))

    gap = estimate_coverage_gap(poly, all_passes, request.sensor.swath_m)
    # Also account for partition discards.
    partitioned_area = sum(p.area_m2 for p in partitions)
    partition_gap = max(0.0, float(poly.area) - partitioned_area)
    total_gap = gap + partition_gap
    if total_gap > 0.05 * float(poly.area):
        warnings.append(
            f"Coverage/partition gap approximately {total_gap:.2f} m^2 "
            f"({100 * total_gap / float(poly.area):.1f}% of region)."
        )

    base_xy = (request.base_position_m[0], request.base_position_m[1])
    cand_raw = candidate_relay_positions(
        poly,
        altitude_m=altitude,
        spacing_m=request.planning.candidate_spacing_m,
        max_candidates=request.planning.capacities.max_candidates,
        base_xy=base_xy,
    )
    candidates = [
        CandidateRelay(id=cid, position_m=pos) for cid, pos in cand_raw
    ]

    scenario = PreparedScenario(
        preprocess_version=PREPROCESS_VERSION,
        request=request,
        partitions=partitions,
        segments=segments,
        candidates=candidates,
        total_survey_area_m2=float(poly.area),
        coverage_gap_area_m2=float(total_gap),
        warnings=tuple(warnings),
    )
    return scenario
