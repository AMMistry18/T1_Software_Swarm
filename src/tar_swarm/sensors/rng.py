"""Independent random streams per sensor kind, derived from one run seed."""

import numpy as np

# Append only: a kind's stream is fixed by its index, so adding a sensor kind
# never changes another sensor's noise. Perception stages that need randomness
# (RANSAC) get their own stream here too.
SENSOR_STREAMS = ("lidar", "rangefinder", "l2_obstacles")


def sensor_rngs(
    seed: int, streams: tuple[str, ...] = SENSOR_STREAMS
) -> dict[str, np.random.Generator]:
    children = np.random.SeedSequence(seed).spawn(len(streams))
    return {
        name: np.random.default_rng(child)
        for name, child in zip(streams, children, strict=True)
    }
