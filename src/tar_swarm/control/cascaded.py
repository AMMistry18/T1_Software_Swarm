"""Position PD, geometric attitude PD, and bounded four-motor allocation."""

from typing import Any

import numpy as np

from tar_swarm.control.interfaces import ControlOutput, Reference, State
from tar_swarm.vehicles.model import allocation_matrix, mass_properties


class CascadedController:
    def __init__(self, vehicle: dict[str, Any], gravity: float):
        self.mass, _, self.inertia = mass_properties(vehicle)
        self.gravity = gravity
        self.gains = vehicle["controller"]
        self.max_thrust = vehicle["max_rotor_thrust_n"]
        self.mix = allocation_matrix(vehicle)
        self.inverse_mix = np.linalg.inv(self.mix)

    @property
    def hover_thrust(self) -> float:
        return self.mass * self.gravity / 4

    def update(self, state: State, reference: Reference) -> ControlOutput:
        g = self.gains
        acceleration = (
            reference.acceleration
            + np.array(g["position_kp"]) * (reference.position - state.position)
            + np.array(g["velocity_kd"]) * (reference.velocity - state.velocity)
        )
        original = acceleration.copy()
        limit = g["max_acceleration_m_s2"]
        acceleration *= min(1.0, limit / max(np.linalg.norm(acceleration), 1e-12))
        force = self.mass * (acceleration + [0, 0, self.gravity])
        # Positive vertical force avoids an undefined desired attitude in free fall
        # when users request downward acceleration equal to or greater than g.
        force[2] = max(force[2], 0.05 * self.mass * self.gravity)
        # Bound tilt while keeping the gravity-compensated vertical force.
        horizontal_limit = max(force[2], 0.01) * np.tan(np.deg2rad(g["max_tilt_deg"]))
        force[:2] *= min(1.0, horizontal_limit / max(np.linalg.norm(force[:2]), 1e-12))
        limited = not np.allclose(original, force / self.mass - [0, 0, self.gravity])
        z_des = force / np.linalg.norm(force)
        heading = np.array([np.cos(reference.yaw), np.sin(reference.yaw), 0])
        y_des = np.cross(z_des, heading)
        y_des /= np.linalg.norm(y_des)
        desired = np.column_stack([np.cross(y_des, z_des), y_des, z_des])
        skew = (desired.T @ state.rotation - state.rotation.T @ desired) / 2
        error = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
        omega = state.angular_velocity
        torque = self.inertia @ (
            -np.array(g["attitude_kp"]) * error - np.array(g["rate_kd"]) * omega
        )
        torque += np.cross(omega, self.inertia @ omega)
        collective = max(0.0, float(force @ state.rotation[:, 2]))
        raw = self.inverse_mix @ np.r_[collective, torque]
        thrusts = np.clip(raw, 0, self.max_thrust)
        return ControlOutput(
            thrusts, bool(np.any(np.abs(raw - thrusts) > 1e-10)), limited
        )
