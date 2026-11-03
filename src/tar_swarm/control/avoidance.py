"""Speed-limit / stop-and-hold obstacle avoidance on sector distances.

Works on any source of horizontal sector distances in the level frame (see
perception.obstacles): 72 L2 sectors, or the TF-Luna as one forward sector.
No path planning: the vehicle slows, stops before d_stop and holds.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tar_swarm.control.interfaces import Reference

SOURCES = ("l2", "rangefinder")
# Rejoin the mission reference exactly once the held reference is this close and
# the command is no longer limited; clear paths then pass through unchanged.
REJOIN_TOLERANCE_M = 0.1


@dataclass(frozen=True)
class AvoidanceParams:
    sources: tuple[str, ...]
    a_brake_m_s2: float
    d_stop_m: float
    cone_half_angle_deg: float
    control_delay_s: float
    stale_timeout_s: float
    stale_speed_m_s: float
    stale_hold_after_s: float
    catchup_gain: float
    rangefinder_latch_m: float


@dataclass
class SectorSource:
    centers_rad: NDArray[np.float64]  # level frame, 0 = heading, CCW positive
    distance_m: NDArray[np.float64]  # NaN = no return
    age_s: float  # time since the reading; inf = no reading yet
    period_s: float  # sensor update period


@dataclass
class AvoidanceRecord:
    commanded_velocity: NDArray[np.float64]  # world frame
    limited_velocity: NDArray[np.float64]  # world frame
    speed_limit_m_s: float  # horizontal; inf = no limit
    status: str  # clear, limited, stop, stale, stale_hold


def validate_avoidance_params(params: dict[str, Any]) -> None:
    for key in ("enabled", *AvoidanceParams.__annotations__):
        if key not in params:
            raise ValueError(f"Missing avoidance parameter: {key}")
    if not isinstance(params["enabled"], bool):
        raise ValueError("Avoidance enabled must be true or false")
    sources = params["sources"]
    if (
        not isinstance(sources, list)
        or not sources
        or any(s not in SOURCES for s in sources)
        or len(set(sources)) != len(sources)
    ):
        raise ValueError(f"Avoidance sources must be a nonempty subset of {SOURCES}")
    for key in ("a_brake_m_s2", "stale_timeout_s", "catchup_gain"):
        if not np.isfinite(params[key]) or params[key] <= 0:
            raise ValueError(f"Avoidance {key} must be positive")
    for key in (
        "d_stop_m",
        "control_delay_s",
        "stale_speed_m_s",
        "stale_hold_after_s",
        "rangefinder_latch_m",
    ):
        if not np.isfinite(params[key]) or params[key] < 0:
            raise ValueError(f"Avoidance {key} must be non-negative")
    if not 0 < params["cone_half_angle_deg"] <= 180:
        raise ValueError("Avoidance cone_half_angle_deg must be in (0, 180]")


def load_avoidance_params(
    avoidance: dict[str, Any] | None, vehicle_type: str
) -> AvoidanceParams | None:
    """Parameters for a vehicle type, or None when absent or disabled."""
    raw = (avoidance or {}).get(vehicle_type)
    if raw is None:
        return None
    validate_avoidance_params(raw)
    if not raw["enabled"]:
        return None
    values = {key: raw[key] for key in AvoidanceParams.__annotations__}
    values["sources"] = tuple(raw["sources"])
    return AvoidanceParams(**values)


def stopping_speed(distance: float, latency_s: float, params: AvoidanceParams) -> float:
    """Fastest speed that can still stop d_stop short of an obstacle.

    Solves v * latency + v^2 / (2 a) = max(d - d_stop, 0); with zero latency this
    is sqrt(2 a (d - d_stop)).
    """
    a, room = params.a_brake_m_s2, max(distance - params.d_stop_m, 0.0)
    return float(-a * latency_s + np.sqrt((a * latency_s) ** 2 + 2 * a * room))


def speed_limit(
    direction: NDArray | None, sources: list[SectorSource], params: AvoidanceParams
) -> tuple[float, bool, bool]:
    """Horizontal speed limit, and flags (any source stale, any source held).

    direction is the unit horizontal command direction in the level frame, or
    None when there is no horizontal motion (only staleness then applies).
    """
    limit, stale, hold = np.inf, False, False
    cone = np.radians(params.cone_half_angle_deg)
    for source in sources:
        if source.age_s > params.stale_timeout_s + params.stale_hold_after_s:
            hold = True
        elif source.age_s > params.stale_timeout_s:
            stale = True
            limit = min(limit, params.stale_speed_m_s)
        if direction is None or not np.isfinite(source.age_s):
            continue
        heading = np.arctan2(direction[1], direction[0])
        offset = np.angle(np.exp(1j * (source.centers_rad - heading)))
        ahead = (np.abs(offset) <= cone) & np.isfinite(source.distance_m)
        latency = source.period_s + params.control_delay_s + source.age_s
        for distance in source.distance_m[ahead]:
            limit = min(limit, stopping_speed(distance, latency, params))
    return (0.0 if hold else limit), stale, hold


def rangefinder_sector(
    distance: float, last_valid: float | None, latch_m: float
) -> float:
    """TF-Luna reading for avoidance; NaN = clear.

    A reading below the sensor's minimum range is reported as no return. If the
    last valid reading was already closer than latch_m, a no-return is taken as
    "too close" (0 m) rather than clear.
    """
    if np.isfinite(distance):
        return float(distance)
    if last_valid is not None and last_valid < latch_m:
        return 0.0
    return float("nan")


class ReferenceShaper:
    """Limits how fast the position reference advances toward the mission.

    The flight controller tracks a position reference, so clipping only the
    reference velocity would not stop the vehicle. The shaper holds its own
    reference position and advances it at the limited horizontal velocity;
    altitude follows the mission. With nothing limiting it returns the mission
    reference unchanged.
    """

    def __init__(self, params: AvoidanceParams):
        self.params = params
        self.position: NDArray | None = None
        self.velocity = np.zeros(2)  # horizontal velocity of the last reference

    def update(
        self,
        reference: Reference,
        heading: NDArray,
        sources: list[SectorSource],
        dt: float,
    ) -> tuple[Reference, AvoidanceRecord]:
        if self.position is None:
            self.position = reference.position.copy()
        command = reference.velocity.copy()
        command[:2] += self.params.catchup_gain * (
            reference.position[:2] - self.position[:2]
        )
        level = heading.T @ command
        speed = float(np.hypot(level[0], level[1]))
        direction = level[:2] / speed if speed > 1e-6 else None
        limit, stale, hold = speed_limit(direction, sources, self.params)
        limited = command.copy()
        if speed > limit:
            limited[:2] *= limit / speed
        if hold:
            status = "stale_hold"
        elif limit <= 0 and speed > 0:
            status = "stop"
        elif stale:
            status = "stale"
        elif speed > limit:
            status = "limited"
        else:
            status = "clear"
        record = AvoidanceRecord(command, limited, limit, status)
        gap = np.linalg.norm(reference.position[:2] - self.position[:2])
        if speed <= limit and gap < REJOIN_TOLERANCE_M:
            self.position = reference.position.copy()
            self.velocity = reference.velocity[:2].copy()
            return reference, record
        # Feed the reference's own braking forward so the position controller
        # does not lag (and overshoot) the slowing reference; bounded by a_brake.
        acceleration = (limited[:2] - self.velocity) / dt
        norm = np.linalg.norm(acceleration)
        if norm > self.params.a_brake_m_s2:
            acceleration *= self.params.a_brake_m_s2 / norm
        self.velocity = self.velocity + acceleration * dt
        self.position[:2] += self.velocity * dt
        self.position[2] = reference.position[2]
        shaped = Reference(
            self.position.copy(),
            np.array([self.velocity[0], self.velocity[1], reference.velocity[2]]),
            np.array([acceleration[0], acceleration[1], reference.acceleration[2]]),
            reference.yaw,
        )
        return shaped, record
