"""World ENU-like position; body FLU orientation and angular velocity."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass
class State:
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    rotation: NDArray[np.float64]
    angular_velocity: NDArray[np.float64]


@dataclass
class Reference:
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    acceleration: NDArray[np.float64]
    yaw: float = 0.0


@dataclass
class ControlOutput:
    thrusts: NDArray[np.float64]
    motor_saturated: bool
    acceleration_limited: bool


class FlightController(Protocol):
    def update(self, state: State, reference: Reference) -> ControlOutput: ...
