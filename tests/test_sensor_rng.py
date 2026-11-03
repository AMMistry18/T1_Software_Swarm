import numpy as np

from tar_swarm.sensors.rng import SENSOR_STREAMS, sensor_rngs


def test_streams_deterministic_per_seed():
    a = sensor_rngs(42)["lidar"].uniform(size=100)
    b = sensor_rngs(42)["lidar"].uniform(size=100)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, sensor_rngs(43)["lidar"].uniform(size=100))


def test_adding_a_stream_leaves_earlier_streams_unchanged():
    alone = sensor_rngs(42, ("lidar",))["lidar"].uniform(size=100)
    together = sensor_rngs(42, (*SENSOR_STREAMS, "future"))["lidar"].uniform(size=100)
    assert np.array_equal(alone, together)


def test_streams_are_distinct():
    rngs = sensor_rngs(42)
    assert set(rngs) == set(SENSOR_STREAMS)
    assert not np.array_equal(
        rngs["lidar"].uniform(size=100), rngs["rangefinder"].uniform(size=100)
    )
