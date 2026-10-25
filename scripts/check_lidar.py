"""Plot one LiDAR scan in world frame and report hit statistics for a run."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from tar_swarm.vehicles.model import components, mass_properties  # noqa: E402


def sensor_offset(config: dict, vehicle_id: str) -> np.ndarray:
    """Site position in the body frame (same formula as the MJCF generator)."""
    agent = next(a for a in config["mission"]["agents"] if a["id"] == vehicle_id)
    vehicle = config["vehicles"][agent["vehicle"]]
    _, center, _ = mass_properties(vehicle)
    lidar = next(c for c in components(vehicle) if c.name == "unitree_l2")
    return np.asarray(lidar.position - center)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--scan", type=int, default=-1, help="Scan index")
    args = parser.parse_args()
    scans = np.load(args.run / "lidar_scans.npz")
    truth = np.load(args.run / "lidar_truth.npz")
    config = yaml.safe_load((args.run / "resolved_config.yaml").read_text("utf-8"))
    index = args.scan % len(scans["time_s"])
    vehicle = str(scans["vehicle_id"][index])
    time_s = float(scans["time_s"][index])
    start = scans["scan_start"][index]
    end = (
        scans["scan_start"][index + 1]
        if index + 1 < len(scans["scan_start"])
        else len(scans["points"])
    )
    points = scans["points"][start:end].astype(float)
    with (args.run / "states.csv").open(encoding="utf-8") as stream:
        rows = [
            r
            for r in csv.DictReader(stream)
            if r["vehicle_id"] == vehicle and np.isclose(float(r["time_s"]), time_s)
        ]
    row = rows[0]
    position = np.array([float(row[f"position_{a}_m"]) for a in "xyz"])
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_euler(
        "xyz", [float(row[f"{n}_rad"]) for n in ("roll", "pitch", "yaw")]
    ).as_matrix()
    # Site frame: z axis down (zaxis 0 0 -1), x axis chosen by MuJoCo; the scan is
    # rotationally uniform about the sensor axis, so a pure flip about x is exact
    # for the statistics here. Use the body pose plus flipped sensor z.
    site_to_body = np.diag([1.0, -1.0, -1.0])
    world = (points @ site_to_body.T) @ rotation.T + (
        position + rotation @ sensor_offset(config, vehicle)
    )
    dist = np.linalg.norm(points, axis=1)
    print(f"scan {index} t={time_s:.2f}s {vehicle}: {len(points)} valid points")
    print(
        f"range min/mean/max: {dist.min():.2f} / {dist.mean():.2f} / "
        f"{dist.max():.2f} m"
    )
    ids, counts = np.unique(truth["geom_id"][start:end], return_counts=True)
    print("hits per geom id:", dict(zip(ids.tolist(), counts.tolist())))
    figure = plt.figure(figsize=(8, 7))
    axes = figure.add_subplot(projection="3d")
    axes.scatter(*world.T, c=world[:, 2], s=2, cmap="viridis")
    axes.scatter(*position, c="red", s=40, label="mapper")
    axes.set_xlabel("x east (m)")
    axes.set_ylabel("y north (m)")
    axes.set_zlabel("z up (m)")
    axes.legend()
    axes.set_title(f"LiDAR scan {index} at t={time_s:.1f}s (world frame)")
    output = args.run / "lidar_scan.png"
    figure.savefig(output, dpi=150)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
