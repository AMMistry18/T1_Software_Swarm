"""Targets, IDs, attitude axes, desired paths, and actual flight trails."""

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from tar_swarm.control.interfaces import Reference
from tar_swarm.mujoco.flight import FlightSimulation

COLORS = [
    (0.1, 0.45, 1.0, 1.0),
    (1.0, 0.45, 0.05, 1.0),
    (0.8, 0.2, 0.7, 1.0),
    (0.1, 0.8, 0.6, 1.0),
]


def draw_overlays(
    scene: mujoco.MjvScene,
    simulation: FlightSimulation,
    config: dict[str, Any],
    targets: dict[str, Reference],
    trails: dict[str, list[np.ndarray]],
) -> None:
    def sphere(position: np.ndarray, radius: float, rgba: Any, label: str = "") -> None:
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0, 0]),
            position,
            np.eye(3).ravel(),
            np.array(rgba),
        )
        geom.label = label
        scene.ngeom += 1

    def line(start: np.ndarray, end: np.ndarray, width: float, rgba: Any) -> None:
        if scene.ngeom >= scene.maxgeom or np.linalg.norm(end - start) < 1e-7:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).ravel(),
            np.array(rgba),
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, start, end)
        scene.ngeom += 1

    for index, agent in enumerate(config["mission"]["agents"]):
        name = agent["id"]
        color = COLORS[index % len(COLORS)]
        state = simulation.state(name)
        sphere(state.position + [0, 0, 0.3], 0.025, color, name)
        sphere(targets[name].position, 0.05, (*color[:3], 0.35))
        # Show the next discrete waypoint as well as the moving flight reference.
        goal = next(
            (
                w["position_m"]
                for w in agent["waypoints"]
                if w["time_s"] > simulation.data.time + 1e-8
            ),
            agent["waypoints"][-1]["position_m"],
        )
        label = (
            f"{name} waypoint"
            if np.linalg.norm(np.array(goal) - state.position) > 0.3
            else ""
        )
        sphere(np.array(goal, dtype=float), 0.09, (*color[:3], 0.25), label)
        positions = np.array([w["position_m"] for w in agent["waypoints"]])
        for start, end in zip(positions, positions[1:]):
            # Dashed desired geometric path; actual trail is continuous.
            for u in np.arange(0, 1, 0.1):
                line(
                    start + u * (end - start),
                    start + min(u + 0.05, 1) * (end - start),
                    0.009,
                    (*color[:3], 0.5),
                )
        trail = trails[name]
        stride = max(1, len(trail) // 180)
        reduced = trail[::stride]
        for start, end in zip(reduced, reduced[1:]):
            line(start, end, 0.012, color)
        for axis, axis_color in enumerate([(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]):
            line(
                state.position,
                state.position + 0.25 * state.rotation[:, axis],
                0.008,
                axis_color,
            )


def configure_viewer(viewer: Any) -> None:
    viewer.cam.lookat[:] = [0, 0, 2]
    viewer.cam.distance = 18
    viewer.cam.azimuth = 130
    viewer.cam.elevation = -30
    viewer.opt.geomgroup[3] = 0
    viewer.opt.label = mujoco.mjtLabel.mjLABEL_SELECTION


def save_snapshot(
    path: Path,
    simulation: FlightSimulation,
    config: dict[str, Any],
    targets: dict[str, Reference],
    trails: dict[str, list[np.ndarray]],
) -> None:
    import matplotlib.pyplot as plt

    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 1.8]
    camera.distance = 18
    camera.azimuth = 130
    camera.elevation = -30
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    option.label = mujoco.mjtLabel.mjLABEL_SELECTION
    with mujoco.Renderer(simulation.model, height=900, width=1200) as renderer:
        renderer.update_scene(simulation.data, camera=camera, scene_option=option)
        draw_overlays(renderer.scene, simulation, config, targets, trails)
        plt.imsave(path, renderer.render())
