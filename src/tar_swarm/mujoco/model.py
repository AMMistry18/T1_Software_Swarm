"""Generate an inspectable MJCF scene from component and world configurations."""

from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

from tar_swarm.vehicles.model import components, mass_properties, rotor_positions


def vector(values: Any) -> str:
    return " ".join(f"{v:.12g}" for v in values)


def build_scene(config: dict[str, Any]) -> str:
    sim, world = config["simulation"], config["world"]
    root = ET.Element("mujoco", model=f"tar_step0_{world['name']}")
    ET.SubElement(
        root, "compiler", angle="radian", autolimits="true", inertiafromgeom="true"
    )
    ET.SubElement(
        root,
        "option",
        timestep=str(sim["physics_timestep_s"]),
        gravity=f"0 0 {-sim['gravity_m_s2']}",
        integrator="RK4",
    )
    ET.SubElement(root, "statistic", center="0 0 2", extent="15")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        name="sky",
        type="skybox",
        builtin="gradient",
        rgb1="0.35 0.55 0.8",
        rgb2="0.85 0.9 0.95",
        width="512",
        height="512",
    )
    ET.SubElement(
        asset,
        "texture",
        name="grid",
        type="2d",
        builtin="checker",
        rgb1="0.24 0.32 0.24",
        rgb2="0.32 0.4 0.32",
        width="256",
        height="256",
    )
    ET.SubElement(asset, "material", name="ground", texture="grid", texrepeat="30 30")
    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", pos="0 0 15", dir="-0.2 -0.3 -1", directional="true")
    ET.SubElement(
        wb,
        "geom",
        name="floor",
        type="plane",
        size=f"{world['ground_half_size_m']} {world['ground_half_size_m']} 0.1",
        material="ground",
        friction="0.8 0.01 0.001",
    )
    station = world["ground_station"]
    ET.SubElement(
        wb,
        "geom",
        name="ground_station",
        type="box",
        pos=vector(station["position_m"]),
        size=vector(np.array(station["size_m"]) / 2),
        rgba="0.9 0.7 0.1 1",
    )
    for obstacle in world["obstacles"]:
        ET.SubElement(
            wb,
            "geom",
            name=obstacle["name"],
            type="box",
            pos=vector(obstacle["position_m"]),
            size=vector(np.array(obstacle["size_m"]) / 2),
            rgba="0.55 0.52 0.49 1",
        )
    actuators = ET.SubElement(root, "actuator")
    for agent in config["mission"]["agents"]:
        name = agent["id"]
        vehicle = config["vehicles"][agent["vehicle"]]
        _, center, _ = mass_properties(vehicle)
        yaw = agent.get("yaw_rad", 0.0)
        body = ET.SubElement(
            wb,
            "body",
            name=name,
            pos=vector(agent["spawn_m"]),
            quat=vector([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]),
        )
        ET.SubElement(body, "freejoint", name=f"{name}_free")
        mapper = agent["vehicle"] == "mapping_drone"
        color = "0.1 0.4 0.9 1" if mapper else "0.95 0.45 0.06 1"
        for component in components(vehicle):
            propeller = component.name.startswith("propeller")
            rgba = "0.15 0.15 0.18 0.5" if propeller else color
            if "battery" in component.name:
                rgba = "0.12 0.12 0.15 1"
            if component.name == "unitree_l2":
                rgba = "0.15 0.8 0.7 1"
            # Harness/electronics overlaps are intentional mass allowances, hidden
            # in geom group 3; no component self-collisions within a rigid body.
            visible = component.name.startswith(
                ("arm", "motor", "propeller", "battery")
            ) or component.name in (
                "center_frame",
                "landing_gear",
                "unitree_l2",
                "jetson_carrier_allowance",
                "cube_carrier",
                "flight_pcb",
                "tf_luna",
                "thermal_breakout",
                "vocore_antenna_assembly",
                "gnss_antenna_allowance",
            )
            # Only exterior components are collision shapes; prop disks conservatively
            # represent the swept blade volume, with no aerodynamic blade simulation.
            collision = visible and component.name not in ("flight_pcb", "cube_carrier")
            size = component.size / 2 if component.kind == "box" else component.size
            ET.SubElement(
                body,
                "geom",
                name=f"{name}_{component.name}",
                type=component.kind,
                pos=vector(component.position - center),
                size=vector(size),
                quat=vector(
                    [np.cos(component.yaw / 2), 0, 0, np.sin(component.yaw / 2)]
                ),
                mass=str(component.mass),
                rgba=rgba,
                group="0" if visible else "3",
                contype="1" if collision else "0",
                conaffinity="1" if collision else "0",
            )
        for i, pos in enumerate(rotor_positions(vehicle)):
            site = f"{name}_rotor_{i}"
            ET.SubElement(
                body,
                "site",
                name=site,
                pos=vector(pos + [0, 0, vehicle["motor_height_m"]] - center),
                size="0.006" if mapper else "0.002",
                rgba="0 0 0 0",
            )
            sign = 1 if i % 2 == 0 else -1
            ET.SubElement(
                actuators,
                "motor",
                name=f"{name}_motor_{i}",
                site=site,
                gear=f"0 0 1 0 0 {sign * vehicle['reaction_torque_ratio_m']}",
                ctrllimited="true",
                ctrlrange=f"0 {vehicle['max_rotor_thrust_n']}",
                forcelimited="true",
                forcerange=f"0 {vehicle['max_rotor_thrust_n']}",
            )
        # Axes and downward sensor mount are visualization sites only.
        ET.SubElement(
            body,
            "site",
            name=f"{name}_heading",
            type="box",
            pos=vector([vehicle["arm_length_m"] * 0.25, 0, 0.04]),
            size=vector([vehicle["arm_length_m"] * 0.15, 0.005, 0.005]),
            rgba="1 0.1 0.1 1",
        )
        if mapper:
            lidar = next(c for c in components(vehicle) if c.name == "unitree_l2")
            ET.SubElement(
                body,
                "site",
                name=f"{name}_lidar_down",
                pos=vector(lidar.position - center),
                zaxis="0 0 -1",
                size="0.01",
                rgba="0 1 1 1",
            )
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")
