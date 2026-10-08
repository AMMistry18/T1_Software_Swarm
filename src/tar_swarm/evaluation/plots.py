"""Quantitative figures from states.csv and controls.csv only."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_run(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    with (directory / "states.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with (directory / "controls.csv").open(encoding="utf-8") as stream:
        controls = list(csv.DictReader(stream))
    output = directory / "plots"
    output.mkdir(exist_ok=True)
    paths = []
    trajectory = plt.figure(figsize=(10, 8))
    ax3 = trajectory.add_subplot(projection="3d")
    for name in dict.fromkeys(row["vehicle_id"] for row in rows):
        records = [r for r in rows if r["vehicle_id"] == name]
        commands = [r for r in controls if r["vehicle_id"] == name]

        def column(key: str) -> np.ndarray:
            return np.array([float(r[key]) for r in records])

        time = column("time_s")
        position = np.column_stack([column(f"position_{a}_m") for a in "xyz"])
        desired = np.column_stack([column(f"desired_{a}_m") for a in "xyz"])
        figure, axes = plt.subplots(6, 1, figsize=(12, 18), sharex=True)
        for i, axis in enumerate("xyz"):
            axes[0].plot(time, position[:, i], label=f"actual {axis}")
            axes[0].plot(time, desired[:, i], "--", label=f"desired {axis}")
            axes[1].plot(time, position[:, i] - desired[:, i], label=axis)
            axes[2].plot(time, column(f"velocity_{axis}_m_s"), label=axis)
        axes[1].plot(
            time,
            np.linalg.norm(position - desired, axis=1),
            color="black",
            label="norm",
        )
        for angle in ("roll", "pitch", "yaw"):
            axes[3].plot(time, np.rad2deg(column(f"{angle}_rad")), label=angle)
        command_time = np.array([float(r["time_s"]) for r in commands])
        for motor in range(4):
            axes[4].plot(
                command_time,
                [float(r[f"motor_{motor}_n"]) for r in commands],
                label=f"rotor {motor}",
            )
        for flag in ("motor_saturated", "acceleration_limited"):
            axes[5].step(
                command_time,
                [float(r[flag]) for r in commands],
                where="post",
                label=flag,
            )
        for ax, label in zip(
            axes,
            [
                "Position [m]",
                "Error [m]",
                "Velocity [m/s]",
                "Attitude [deg]",
                "Thrust [N]",
                "Limit flags",
            ],
        ):
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
            ax.legend(ncol=3, fontsize=8)
        axes[-1].set_xlabel("MuJoCo simulation time [s]")
        figure.suptitle(f"{name}: simulation state and applied controller commands")
        figure.tight_layout(rect=(0, 0, 1, 0.98))
        path = output / f"{name}_control.png"
        figure.savefig(path, dpi=130)
        plt.close(figure)
        paths.append(path)
        ax3.plot(*position.T, label=f"{name} actual")
        ax3.plot(*desired.T, "--", label=f"{name} desired")
        ax3.scatter(*position[-1], s=25)
        ax3.text(*position[-1], name, fontsize=8)
    ax3.set(
        xlabel="East / x [m]",
        ylabel="North / y [m]",
        zlabel="Up / z [m]",
        title="Actual and desired trajectories from MuJoCo logs",
    )
    ax3.legend(fontsize=8)
    trajectory.tight_layout()
    path = output / "trajectories_3d.png"
    trajectory.savefig(path, dpi=150)
    plt.close(trajectory)
    paths.append(path)
    return paths
