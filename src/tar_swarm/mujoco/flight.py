"""Ground-truth physics adapter; only this layer imports MuJoCo."""

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from tar_swarm.control.interfaces import State
from tar_swarm.mujoco.model import build_scene


@dataclass
class VehicleHandle:
    body_id: int
    actuator_ids: NDArray[np.int64]


class FlightSimulation:
    def __init__(self, config: dict[str, Any]):
        self.xml = build_scene(config)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.handles = {
            agent["id"]: VehicleHandle(
                self.model.body(agent["id"]).id,
                np.array(
                    [
                        self.model.actuator(f"{agent['id']}_motor_{i}").id
                        for i in range(4)
                    ]
                ),
            )
            for agent in config["mission"]["agents"]
        }
        mujoco.mj_forward(self.model, self.data)

    def state(self, vehicle_id: str) -> State:
        body_id = self.handles[vehicle_id].body_id
        velocity = np.empty(6)
        # mj_objectVelocity returns angular then linear, in world axes when local=0.
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0
        )
        rotation = self.data.xmat[body_id].reshape(3, 3).copy()
        return State(
            self.data.xpos[body_id].copy(),
            velocity[3:].copy(),
            rotation,
            rotation.T @ velocity[:3],
        )

    def command(self, vehicle_id: str, thrusts: NDArray[np.float64]) -> None:
        if np.asarray(thrusts).shape != (4,) or not np.isfinite(thrusts).all():
            raise ValueError("Rotor commands must be four finite thrust values")
        ids = self.handles[vehicle_id].actuator_ids
        self.data.ctrl[ids] = np.clip(
            thrusts,
            self.model.actuator_ctrlrange[ids, 0],
            self.model.actuator_ctrlrange[ids, 1],
        )

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)
        # mj_step's derived xpos/xmat may describe the preceding integration stage.
        # Refresh so logged pose/velocity and contacts match the current qpos/time.
        mujoco.mj_forward(self.model, self.data)
        if (
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
        ):
            raise RuntimeError("Nonfinite physics state")
