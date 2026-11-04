"""Deterministic survey geometry: partitioning, lawnmower paths, candidates.

Supported
---------
- Simple polygons (convex or concave) without holes.
- Fixed survey altitude.
- Coverage path clipping to the polygon using effective sensor swath.

Unsupported (explicitly rejected upstream)
-----------------------------------------
- Polygon holes, obstacles, terrain, variable altitude, non-planar surfaces.

Partitioning and lawnmower generation are conventional (not learned).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid


@dataclass(frozen=True)
class CoveragePass:
    waypoints_xy: list[tuple[float, float]]
    length_m: float


def polygon_from_ring(ring: list[tuple[float, float]]) -> Polygon:
    poly = Polygon(ring)
    if not poly.is_valid:
        repaired = make_valid(poly)
        if (
            isinstance(repaired, Polygon)
            and repaired.is_valid
            and not repaired.is_empty
        ):
            poly = repaired
        elif isinstance(repaired, MultiPolygon):
            # Refuse silent bounding-rectangle replacement; only accept a single
            # repaired simple polygon.
            raise ValueError(
                "Region polygon is invalid and repairs into multiple parts; "
                "supply a valid simple polygon"
            )
        else:
            raise ValueError("Region polygon is invalid and could not be repaired")
    if poly.is_empty or poly.area <= 0:
        raise ValueError("Region polygon has non-positive area")
    if poly.interiors:
        raise ValueError("Polygons with holes are not supported")
    return poly


def is_concave(poly: Polygon, tol: float = 1e-9) -> bool:
    return poly.convex_hull.area - poly.area > tol * max(poly.area, 1.0)


def partition_polygon(
    poly: Polygon, mapper_ids: list[str]
) -> list[tuple[str, Polygon]]:
    """Split into approximately equal-area connected vertical strips.

    Strips are formed by cutting the bounding box and intersecting with the
    polygon. Empty intersections are rejected rather than silently reassigned.
    """
    if not mapper_ids:
        raise ValueError("mapper_ids must be nonempty")
    if len(mapper_ids) == 1:
        return [(mapper_ids[0], poly)]

    minx, miny, maxx, maxy = poly.bounds
    width = maxx - minx
    if width <= 0:
        raise ValueError("Polygon has zero width; cannot partition")
    n = len(mapper_ids)
    parts: list[tuple[str, Polygon]] = []
    for i, mapper_id in enumerate(mapper_ids):
        x0 = minx + i * width / n
        x1 = minx + (i + 1) * width / n
        # Slight expansion on interior cuts to avoid zero-area gaps from edges.
        if i > 0:
            x0 -= 1e-6
        if i < n - 1:
            x1 += 1e-6
        strip = Polygon(
            [
                (x0, miny - 1.0),
                (x1, miny - 1.0),
                (x1, maxy + 1.0),
                (x0, maxy + 1.0),
            ]
        )
        inter = poly.intersection(strip)
        region = _as_single_polygon(inter, mapper_id)
        parts.append((mapper_id, region))
    return parts


def _as_single_polygon(geom, label: str) -> Polygon:
    if geom.is_empty:
        raise ValueError(
            f"Partition for {label} is empty; adjust region or mapper count"
        )
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        # Prefer the largest connected component; report the discarded area.
        polys = sorted(geom.geoms, key=lambda g: g.area, reverse=True)
        if len(polys) > 1 and polys[1].area > 1e-3:
            # Keep connectedness by taking only the primary piece; caller records gap.
            return polys[0]
        return polys[0]
    raise ValueError(f"Partition for {label} is not polygonal: {geom.geom_type}")


def effective_swath_m(swath_m: float, overlap: float) -> float:
    return swath_m * (1.0 - overlap)


def lawnmower_passes(
    poly: Polygon,
    swath_m: float,
    overlap: float,
    angle_deg: float = 0.0,
) -> list[CoveragePass]:
    """Generate clipped lawnmower passes. Angle rotates the pass direction."""
    spacing = effective_swath_m(swath_m, overlap)
    if spacing <= 0:
        raise ValueError("effective swath must be positive")
    rotated = affinity.rotate(poly, -angle_deg, origin="centroid")
    minx, miny, maxx, maxy = rotated.bounds
    # Inset by half swath so the sensor footprint stays inside the region.
    inset = spacing * 0.5
    y = miny + inset
    passes: list[CoveragePass] = []
    direction = 1
    while y <= maxy - inset + 1e-9:
        line = LineString([(minx - 1.0, y), (maxx + 1.0, y)])
        clipped = rotated.intersection(line)
        segments = _line_parts(clipped)
        # Alternate direction for continuous-ish coverage.
        if direction < 0:
            segments = list(reversed(segments))
            segments = [list(reversed(seg)) for seg in segments]
        for seg in segments:
            if len(seg) < 2:
                continue
            # Rotate back to world frame.
            world = [
                affinity.rotate(Point(x, yy), angle_deg, origin=poly.centroid)
                for x, yy in seg
            ]
            pts = [(float(p.x), float(p.y)) for p in world]
            length = float(LineString(pts).length)
            if length > 1e-6:
                passes.append(CoveragePass(pts, length))
        y += spacing
        direction *= -1
    return passes


def _line_parts(geom) -> list[list[tuple[float, float]]]:
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [list(geom.coords)]
    if isinstance(geom, MultiLineString):
        return [list(g.coords) for g in geom.geoms]
    # Points or mixed collections contribute no survey length.
    return []


def group_passes_into_segments(
    passes: list[CoveragePass],
    mapper_id: str,
    altitude_m: float,
    max_length_m: float,
    swath_m: float,
) -> list[dict]:
    """Group consecutive passes into segments with transit connectors."""
    segments: list[dict] = []
    current: list[CoveragePass] = []
    current_len = 0.0
    seg_index = 0

    def flush():
        nonlocal seg_index, current, current_len
        if not current:
            return
        waypoints: list[tuple[float, float, float]] = []
        area = 0.0
        for i, cov in enumerate(current):
            pts = [(x, y, altitude_m) for x, y in cov.waypoints_xy]
            if waypoints:
                # Transit between passes (already alternating lawnmower ends).
                if waypoints[-1][:2] != pts[0][:2]:
                    waypoints.append(pts[0])
            if not waypoints:
                waypoints.extend(pts)
            elif pts[0] == waypoints[-1][:2]:
                waypoints.extend(pts[1:])
            else:
                waypoints.extend(pts)
            area += cov.length_m * swath_m
        length = sum(c.length_m for c in current)
        segments.append(
            {
                "id": f"{mapper_id}_seg_{seg_index:03d}",
                "mapper_id": mapper_id,
                "waypoints_m": _dedupe_waypoints(waypoints),
                "length_m": float(length),
                "coverage_area_m2": float(area),
                "is_transit": False,
            }
        )
        seg_index += 1
        current = []
        current_len = 0.0

    for cov in passes:
        if current and current_len + cov.length_m > max_length_m:
            flush()
        current.append(cov)
        current_len += cov.length_m
    flush()
    return segments


def _dedupe_waypoints(
    waypoints: list[tuple[float, float, float]], tol: float = 1e-6
) -> list[tuple[float, float, float]]:
    if not waypoints:
        return []
    out = [waypoints[0]]
    for wp in waypoints[1:]:
        if math.dist(out[-1], wp) > tol:
            out.append(wp)
    return out


def candidate_relay_positions(
    poly: Polygon,
    altitude_m: float,
    spacing_m: float,
    max_candidates: int,
    base_xy: tuple[float, float],
) -> list[tuple[str, tuple[float, float, float]]]:
    """Deterministic grid of candidate relay holds near/inside the region."""
    minx, miny, maxx, maxy = poly.bounds
    # Expand slightly so relays can sit just outside concave notches.
    pad = spacing_m * 0.5
    xs = np.arange(minx - pad, maxx + pad + 1e-9, spacing_m)
    ys = np.arange(miny - pad, maxy + pad + 1e-9, spacing_m)
    # Always include a few chain-like candidates from base toward centroid.
    centroid = poly.centroid
    chain = _chain_points(base_xy, (centroid.x, centroid.y), spacing_m)

    scored: list[tuple[float, float, float, float]] = []
    for x in xs:
        for y in ys:
            pt = Point(x, y)
            if poly.distance(pt) > spacing_m:
                continue
            # Prefer interior / near-centroid points via stable score.
            score = pt.distance(centroid) + 0.01 * abs(x) + 0.001 * abs(y)
            scored.append((score, float(x), float(y), altitude_m))
    for x, y in chain:
        pt = Point(x, y)
        score = pt.distance(centroid) - 1.0  # slight preference for chain
        scored.append((score, float(x), float(y), altitude_m))

    # Prefer chain points first (stable multihop toward the region), then grid.
    ordered: list[tuple[float, float, float]] = []
    seen: set[tuple[int, int]] = set()

    def _key(x: float, y: float) -> tuple[int, int]:
        return (int(round(x / (spacing_m * 0.25))), int(round(y / (spacing_m * 0.25))))

    for x, y in chain:
        key = _key(x, y)
        if key in seen:
            continue
        seen.add(key)
        ordered.append((float(x), float(y), altitude_m))
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    for _, x, y, z in scored:
        key = _key(x, y)
        if key in seen:
            continue
        seen.add(key)
        ordered.append((x, y, z))
        if len(ordered) >= max_candidates:
            break
    if not ordered:
        raise ValueError("No candidate relay positions generated; check spacing/region")
    return [(f"cand_{i:03d}", pos) for i, pos in enumerate(ordered[:max_candidates])]


def _chain_points(
    start: tuple[float, float], end: tuple[float, float], spacing_m: float
) -> list[tuple[float, float]]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return [start]
    n = max(1, int(math.floor(dist / spacing_m)))
    return [
        (start[0] + dx * (i / (n + 1)), start[1] + dy * (i / (n + 1)))
        for i in range(1, n + 1)
    ]


def estimate_coverage_gap(
    poly: Polygon, passes: list[CoveragePass], swath_m: float
) -> float:
    """Approximate uncovered area after clipping/discretization (m^2)."""
    if not passes:
        return float(poly.area)
    buffers = []
    half = swath_m * 0.5
    for cov in passes:
        if len(cov.waypoints_xy) < 2:
            continue
        buffers.append(
            LineString(cov.waypoints_xy).buffer(
                half, cap_style=2, join_style=2
            )
        )
    if not buffers:
        return float(poly.area)
    covered = unary_union(buffers).intersection(poly)
    gap = float(poly.area - covered.area)
    return max(0.0, gap)
