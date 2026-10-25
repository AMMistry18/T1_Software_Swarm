"""Ground-truth physics adapter; only this layer imports MuJoCo."""

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from tar_swarm.control.interfaces import State
from tar_swarm.mujoco.model import build_scene


def cast_rays(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    origin: NDArray[np.float64],
    directions_world: NDArray[np.float64],
    body_exclude: int,
    max_range_m: float,
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.float64]]:
    """True ray hits: distance (-1 = no hit), geom id, and world surface normal."""
    count = len(directions_world)
    distance = np.empty(count)
    geom_id = np.empty(count, dtype=np.int32)
    normal = np.zeros(3 * count)
    mujoco.mj_multiRay(
        model,
        data,
        np.asarray(origin, dtype=np.float64),
        np.ascontiguousarray(directions_world, dtype=np.float64).ravel(),
        None,
        True,
        body_exclude,
        geom_id,
        distance,
        normal,
        count,
        max_range_m,
    )
    return distance, geom_id, normal.reshape(count, 3)


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

    def site_pose(self, site_name: str) -> tuple[NDArray, NDArray]:
        """World position and site-to-world rotation of a named site."""
        site = self.model.site(site_name)
        return (
            self.data.site_xpos[site.id].copy(),
            self.data.site_xmat[site.id].reshape(3, 3).copy(),
        )

    def cast_rays(
        self,
        vehicle_id: str,
        site_name: str,
        directions_site: NDArray[np.float64],
        max_range_m: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.float64]]:
        """Noise-free ray hits from a site; the vehicle's own body is excluded."""
        origin, rotation = self.site_pose(site_name)
        return cast_rays(
            self.model,
            self.data,
            origin,
            directions_site @ rotation.T,
            self.handles[vehicle_id].body_id,
            max_range_m,
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
