"""Mass distribution and rotor geometry, independent of MuJoCo."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class Component:
    name: str
    mass: float
    position: NDArray[np.float64]
    size: NDArray[np.float64]
    kind: str = "box"
    yaw: float = 0.0


def rotor_positions(vehicle: dict[str, Any]) -> NDArray[np.float64]:
    a = vehicle["arm_length_m"] / np.sqrt(2)
    return np.array([[a, a, 0], [-a, a, 0], [-a, -a, 0], [a, -a, 0]])


def components(vehicle: dict[str, Any]) -> list[Component]:
    result = [
        Component(
            c["name"],
            c["mass_kg"],
            np.array(c["pos_m"], dtype=float),
            np.array(c["size_m"], dtype=float),
        )
        for c in vehicle["components"]
    ]
    radius = vehicle["propeller_diameter_in"] * 0.0254 / 2
    for i, pos in enumerate(rotor_positions(vehicle)):
        yaw = float(np.arctan2(pos[1], pos[0]))
        result.extend(
            [
                Component(
                    f"arm_{i}",
                    vehicle["arm_mass_kg"],
                    pos / 2,
                    np.array(
                        [
                            vehicle["arm_length_m"],
                            vehicle["arm_width_m"],
                            vehicle["arm_width_m"],
                        ]
                    ),
                    yaw=yaw,
                ),
                Component(
                    f"motor_{i}",
                    vehicle["motor_mass_kg"],
                    pos + [0, 0, vehicle["motor_height_m"] / 2],
                    np.array(
                        [vehicle["motor_radius_m"], vehicle["motor_height_m"] / 2]
                    ),
                    "cylinder",
                ),
                Component(
                    f"propeller_{i}",
                    vehicle["propeller_mass_kg"],
                    pos + [0, 0, vehicle["motor_height_m"]],
                    np.array([radius, 0.001]),
                    "cylinder",
                ),
            ]
        )
    return result


def mass_properties(vehicle: dict[str, Any]) -> tuple[float, NDArray, NDArray]:
    """Return mass, COM in design frame, inertia about COM in body axes."""
    parts = components(vehicle)
    mass = sum(c.mass for c in parts)
    center = sum((c.mass * c.position for c in parts), np.zeros(3)) / mass
    inertia = np.zeros((3, 3))
    for c in parts:
        if c.kind == "box":
            x, y, z = c.size
            local = c.mass / 12 * np.diag([y * y + z * z, x * x + z * z, x * x + y * y])
        else:
            r, half_height = c.size
            transverse = c.mass * (3 * r * r + 4 * half_height**2) / 12
            local = np.diag([transverse, transverse, c.mass * r * r / 2])
        co, si = np.cos(c.yaw), np.sin(c.yaw)
        rotation = np.array([[co, -si, 0], [si, co, 0], [0, 0, 1]])
        d = c.position - center
        inertia += rotation @ local @ rotation.T + c.mass * (
            np.dot(d, d) * np.eye(3) - np.outer(d, d)
        )
    return mass, center, inertia


def allocation_matrix(vehicle: dict[str, Any]) -> NDArray[np.float64]:
    """Body wrench [collective, roll, pitch, yaw] from rotor forces in N."""
    _, center, _ = mass_properties(vehicle)
    rotors = rotor_positions(vehicle) - center
    return np.array(
        [
            np.ones(4),
            rotors[:, 1],
            -rotors[:, 0],
            np.array([1, -1, 1, -1]) * vehicle["reaction_torque_ratio_m"],
        ]
    )
