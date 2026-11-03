"""Headless GIF clips from MuJoCo's offscreen renderer.

Frames are written as an animated GIF with Pillow, so no video dependency is
needed. The camera follows one drone. Clips can be recorded live (runner
`--record`) or rebuilt from a saved run's measurement files (replay), so a
failed sweep run can be watched without re-simulating.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from tar_swarm.mujoco.overlays import draw_sensor_overlays
from tar_swarm.mujoco.replay import load_run, replay_frame

# Display choices, not physical parameters.
FPS = 10.0
WIDTH, HEIGHT = 480, 360
CAMERA_DISTANCE_M = 4.0
CAMERA_AZIMUTH_DEG = 45.0  # behind-left of a drone flying +x
CAMERA_ELEVATION_DEG = -20.0


class GifRecorder:
    """Collect frames from a following camera and save them as one GIF on close."""

    def __init__(
        self,
        model: mujoco.MjModel,
        path: Path,
        fps: float = FPS,
        width: int = WIDTH,
        height: int = HEIGHT,
    ) -> None:
        path = Path(path)
        if path.suffix.lower() != ".gif":
            raise ValueError(f"Recording must be a .gif file: {path}")
        if path.exists():
            raise FileExistsError(f"Recording already exists: {path}")
        if fps <= 0:
            raise ValueError("Recording frame rate must be positive")
        self.path, self.fps = path, fps
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        self.camera.distance = CAMERA_DISTANCE_M
        self.camera.azimuth = CAMERA_AZIMUTH_DEG
        self.camera.elevation = CAMERA_ELEVATION_DEG
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3] = 0  # same hidden group as the viewer
        self.frames: list[Image.Image] = []

    def capture(
        self,
        data: mujoco.MjData,
        lookat: np.ndarray,
        draw: Callable[[mujoco.MjvScene], None] | None = None,
    ) -> None:
        """Render one frame centred on lookat; draw appends overlay geoms."""
        self.camera.lookat[:] = lookat
        self.renderer.update_scene(data, camera=self.camera, scene_option=self.option)
        if draw:
            draw(self.renderer.scene)
        image = Image.fromarray(self.renderer.render())
        self.frames.append(image.quantize(256, method=Image.Quantize.FASTOCTREE))

    def close(self) -> Path | None:
        """Write the GIF (if any frames were captured) and free the renderer."""
        self.renderer.close()
        if not self.frames:
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frames[0].save(
            self.path,
            save_all=True,
            append_images=self.frames[1:],
            duration=round(1000 / self.fps),
            loop=0,
        )
        return self.path

    def __enter__(self) -> "GifRecorder":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def record_replay(
    directory: Path,
    path: Path,
    settings: dict[str, Any],
    *,
    start_s: float = 0.0,
    end_s: float | None = None,
    fps: float = FPS,
) -> Path | None:
    """Render a saved run (measurement-side files only) to a GIF with overlays."""
    run = load_run(directory)
    data = mujoco.MjData(run.model)
    follow = run.tracks[next(iter(run.tracks))]
    end_s = run.end_time_s if end_s is None else min(end_s, run.end_time_s)
    with GifRecorder(run.model, path, fps) as recorder:
        for time_s in np.arange(start_s, end_s + 1e-9, 1 / fps):
            frame = replay_frame(run, float(time_s))
            data.qpos[:] = frame.qpos
            data.time = time_s
            mujoco.mj_forward(run.model, data)
            recorder.capture(
                data,
                follow.position[follow.index(float(time_s))],
                lambda scene, f=frame: draw_sensor_overlays(
                    scene, settings, f.beams, f.l2, f.arrows
                ),
            )
    return path if recorder.frames else None
