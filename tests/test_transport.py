"""Checks for packet transport: airtime, LiDAR chunks, the priority score, one
talker at a time, dropping packets with no route, and the counts in a full run."""

from pathlib import Path

import pytest

from tar_swarm.network.config import load_config
from tar_swarm.network.graph import GraphSnapshot
from tar_swarm.network.simulator import Simulator
from tar_swarm.network.transport import (
    HEALTH,
    LIDAR,
    REPOSITION,
    Transport,
    keyframe_chunk_sizes,
)

ROOT = Path(__file__).resolve().parents[1]
# 1 Mb/s (the slowest speed the team plans for) and 0.3 ms per packet.
SETTINGS = {"link_rate_mbps": 1.0, "per_packet_overhead_s": 0.0003}
# The MVP chain: mapper_1 <-> relay_1 <-> ground. The mapper can't reach the base
# directly, so its packets take 2 hops.
CHAIN = GraphSnapshot(
    0.0,
    ("ground", "mapper_1", "relay_1"),
    (("ground", "relay_1"), ("mapper_1", "relay_1")),
)
HEALTH_AIRTIME_S = 0.0003 + (44 + 90) * 8 / 1e6  # 1.372 ms
CHUNK_AIRTIME_S = 0.0003 + (1200 + 90) * 8 / 1e6  # 10.62 ms


def test_airtime_counts_every_byte():
    """Airtime = fixed cost + (packet + lower-layer bytes) x 8 bits / link speed."""
    transport = Transport(SETTINGS, 0.1)
    assert transport.airtime_s(44) == pytest.approx(0.001372)
    assert transport.airtime_s(1200) == pytest.approx(0.01062)


def test_keyframe_chunks_fit_the_cap():
    """A 10,000-byte keyframe becomes 9 chunks, none over the 1200-byte cap, and
    together they carry every byte of map data."""
    sizes = keyframe_chunk_sizes(10_000)
    assert sizes == [1200] * 8 + [652]
    assert sum(size - 28 for size in sizes) == 10_000


def test_health_jumps_ahead_of_waiting_lidar():
    """A new Health packet goes before a LiDAR chunk that has waited 0.5 s, because
    Health's 2 s head start beats 0.5 s of waiting."""
    transport = Transport(SETTINGS, 0.1)
    transport.add_to_outbox(LIDAR, "relay_1", "ground", 1200, 0.0)
    transport.add_to_outbox(HEALTH, "relay_1", "ground", 44, 0.5)
    transport.move_packets(0.5, 0.6, CHAIN)
    assert transport.delays_s[HEALTH] == [pytest.approx(HEALTH_AIRTIME_S)]
    assert transport.delays_s[LIDAR] == [
        pytest.approx(0.5 + HEALTH_AIRTIME_S + CHUNK_AIRTIME_S)
    ]


def test_lidar_that_waited_long_enough_goes_first():
    """No starvation: after 2.5 s of waiting, the LiDAR chunk's score (2.5) beats a
    new Health packet's head start (2.0), so the chunk goes first."""
    transport = Transport(SETTINGS, 0.1)
    transport.add_to_outbox(LIDAR, "relay_1", "ground", 1200, 0.0)
    transport.add_to_outbox(HEALTH, "relay_1", "ground", 44, 2.5)
    transport.move_packets(2.5, 2.6, CHAIN)
    assert transport.delays_s[HEALTH] == [
        pytest.approx(CHUNK_AIRTIME_S + HEALTH_AIRTIME_S)
    ]


def test_two_hops_use_twice_the_airtime():
    """One talker at a time: a 2-hop packet takes 2 airtimes, and 10 chunks over 2
    hops take as long as 20 chunks over 1 hop (the team's "2 hops halve the
    speed")."""
    transport = Transport(SETTINGS, 0.1)
    transport.add_to_outbox(HEALTH, "mapper_1", "ground", 44, 0.0)
    transport.move_packets(0.0, 0.1, CHAIN)
    assert transport.delays_s[HEALTH] == [pytest.approx(2 * HEALTH_AIRTIME_S)]

    transport = Transport(SETTINGS, 0.1)
    for _ in range(10):
        transport.add_to_outbox(LIDAR, "mapper_1", "ground", 1200, 0.0)
    transport.move_packets(0.0, 1.0, CHAIN)
    assert len(transport.delays_s[LIDAR]) == 10
    assert max(transport.delays_s[LIDAR]) == pytest.approx(20 * CHUNK_AIRTIME_S)


def test_packets_without_a_route_are_dropped():
    """With the mapper-relay link down, the mapper's packet is dropped and
    counted, like batman-adv drops packets it has no route for."""
    transport = Transport(SETTINGS, 0.1)
    broken = GraphSnapshot(0.0, CHAIN.nodes, (("ground", "relay_1"),))
    transport.add_to_outbox(HEALTH, "mapper_1", "ground", 44, 0.0)
    transport.move_packets(0.0, 0.1, broken)
    assert transport.dropped[HEALTH] == 1
    assert transport.summary()[HEALTH]["waiting"] == 0


def test_simulator_makes_and_accounts_for_every_packet():
    """10 s of the default scene (3 drones, connected the whole time): the right
    number of packets is made, and each is delivered or still on its way."""
    config = load_config(ROOT / "configs/network.yaml")
    config["duration_s"] = 10.0
    sim = Simulator(config)
    sim.run()
    summary = sim.transport.summary()
    # 100 steps of 0.1 s. Health: 3 drones x 10 Hz. Reposition: 2 relays x 5 Hz.
    # LiDAR: 2 keyframes per second x 9 chunks.
    assert summary[HEALTH]["made"] == 300
    assert summary[REPOSITION]["made"] == 100
    assert summary[LIDAR]["made"] == 180
    for counts in summary.values():
        assert counts["dropped_no_route"] == 0
        assert counts["made"] == counts["delivered"] + counts["waiting"]
