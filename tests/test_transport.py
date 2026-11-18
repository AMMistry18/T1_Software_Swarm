"""Checks for packet transport: airtime, link speed by distance, LiDAR chunks, the
priority score, Wi-Fi's random turn-taking and collisions, command ACKs and
resends, dropping packets with no route (whole keyframes at a time), and the
counts in a full run."""

from pathlib import Path

import numpy as np
import pytest

from tar_swarm.network.config import load_config
from tar_swarm.network.graph import GraphSnapshot
from tar_swarm.network.simulator import Simulator
from tar_swarm.network.transport import (
    COMMAND,
    COMMAND_ACK,
    DEGENERACY,
    HEAD_START_S,
    HEALTH,
    LIDAR,
    POSE_CORRECTION,
    REPOSITION,
    SLOT_S,
    Transport,
    keyframe_chunk_sizes,
)

ROOT = Path(__file__).resolve().parents[1]
# The default config's zones (30 Mb/s up to 30% of the radio range, 10 Mb/s up to
# 60%, 1 Mb/s out to the edge) and 0.3 ms per packet.
SETTINGS = {
    "link_speed_zones": [
        {"up_to_range_fraction": 0.3, "mbps": 30.0},
        {"up_to_range_fraction": 0.6, "mbps": 10.0},
        {"up_to_range_fraction": 1.0, "mbps": 1.0},
    ],
    "per_packet_overhead_s": 0.0003,
}
# The MVP chain: mapper_1 <-> relay_1 <-> ground. The mapper can't reach the base
# directly, so its packets take 2 hops.
CHAIN = GraphSnapshot(
    0.0,
    ("ground", "mapper_1", "relay_1"),
    (("ground", "relay_1"), ("mapper_1", "relay_1")),
)
# The same drones with the mapper-relay link down: the mapper is cut off.
BROKEN = GraphSnapshot(0.0, CHAIN.nodes, (("ground", "relay_1"),))
ROLES = {"ground": "ground_station", "relay_1": "relay", "mapper_1": "mapper"}
# Both CHAIN links at 1 Mb/s (the slowest speed the team plans for), both ways.
ONE_MBPS = {
    ("ground", "relay_1"): 1e6,
    ("relay_1", "ground"): 1e6,
    ("mapper_1", "relay_1"): 1e6,
    ("relay_1", "mapper_1"): 1e6,
}
HEALTH_AIRTIME_S = 0.0003 + (44 + 90) * 8 / 1e6  # 1.372 ms at 1 Mb/s
COMMAND_AIRTIME_S = 0.0003 + (32 + 90) * 8 / 1e6  # 1.276 ms at 1 Mb/s
CHUNK_AIRTIME_S = 0.0003 + (1200 + 90) * 8 / 1e6  # 10.62 ms at 1 Mb/s


class FirstNameWins:
    """Stands in for chance in Wi-Fi's turn-taking, so a test knows who talks when:
    the drones that need a new countdown draw 0, 1, 2, ... slots in name order,
    so the first name talks at once, and nobody collides."""

    def integers(self, low, high):
        return np.arange(len(high))


class EveryoneDrawsSeven:
    """Every drone draws a countdown of 7 slots: a drone alone waits 7 slots, and
    two drones with something to send end their countdowns together."""

    def integers(self, low, high):
        return np.full(len(high), 7)


def new_transport(chance=None):
    """A Transport with the test settings and 0.1 s steps."""
    return Transport(SETTINGS, 0.1, chance or FirstNameWins())


def test_airtime_counts_every_byte():
    """Airtime = fixed cost + (packet + lower-layer bytes) x 8 bits / link speed."""
    transport = new_transport()
    assert transport.airtime_s(44, 1e6) == pytest.approx(0.001372)
    assert transport.airtime_s(1200, 1e6) == pytest.approx(0.01062)


def test_link_speed_falls_with_distance():
    """With a 300 m range: up to 90 m apart (30% of the range) a link runs at
    30 Mb/s, up to 180 m (60%) at 10 Mb/s, and out to 300 m at 1 Mb/s. A zone's
    edge counts as inside it, and a link that drifted past its range gets the
    last zone's speed."""
    transport = new_transport()
    distances_m = [50, 90, 150, 250, 305]
    speeds_mbps = [transport.speed_bps(d, 300) / 1e6 for d in distances_m]
    assert speeds_mbps == [30, 30, 10, 1, 1]


def test_each_link_gets_its_own_speed():
    """Base -> relay 80 m away -> mapper 250 m beyond it, all ranges 300 m: the
    short link runs at 30 Mb/s and the long one at 1 Mb/s, the same both ways.
    Shrink the base's range to 150 m and the short link is at 53% of its range
    (a link reaches only as far as the smaller range): 10 Mb/s."""
    transport = new_transport()
    positions = {
        "ground": np.array([0.0, 0.0, 0.0]),
        "relay_1": np.array([80.0, 0.0, 0.0]),
        "mapper_1": np.array([330.0, 0.0, 0.0]),
    }
    ranges = {"ground_station": 300.0, "relay": 300.0, "mapper": 300.0}
    assert transport.link_speeds_bps(CHAIN, ROLES, positions, ranges) == {
        ("ground", "relay_1"): 30e6,
        ("relay_1", "ground"): 30e6,
        ("mapper_1", "relay_1"): 1e6,
        ("relay_1", "mapper_1"): 1e6,
    }
    ranges["ground_station"] = 150.0
    speeds = transport.link_speeds_bps(CHAIN, ROLES, positions, ranges)
    assert speeds["ground", "relay_1"] == 10e6


def test_each_hop_uses_its_own_links_speed():
    """The mapper's Health packet crosses a 1 Mb/s link and then a 30 Mb/s link,
    so it takes one airtime at each speed."""
    transport = new_transport()
    speeds = dict(ONE_MBPS)
    speeds["ground", "relay_1"] = speeds["relay_1", "ground"] = 30e6
    transport.add_to_outbox(HEALTH, "mapper_1", "ground", 44, 0.0)
    transport.move_packets(0.0, 0.1, CHAIN, speeds)
    expected_s = transport.airtime_s(44, 1e6) + transport.airtime_s(44, 30e6)
    assert transport.delays_s[HEALTH] == [pytest.approx(expected_s)]


def test_keyframe_chunks_fit_the_cap():
    """A 10,000-byte keyframe becomes 9 chunks, none over the 1200-byte cap, and
    together they carry every byte of map data."""
    sizes = keyframe_chunk_sizes(10_000)
    assert sizes == [1200] * 8 + [652]
    assert sum(size - 28 for size in sizes) == 10_000


def test_health_jumps_ahead_of_waiting_lidar():
    """A new Health packet goes before a LiDAR chunk that has waited 0.5 s, because
    Health's 2 s head start beats 0.5 s of waiting."""
    transport = new_transport()
    transport.add_to_outbox(LIDAR, "relay_1", "ground", 1200, 0.0)
    transport.add_to_outbox(HEALTH, "relay_1", "ground", 44, 0.5)
    transport.move_packets(0.5, 0.6, CHAIN, ONE_MBPS)
    assert transport.delays_s[HEALTH] == [pytest.approx(HEALTH_AIRTIME_S)]
    assert transport.delays_s[LIDAR] == [
        pytest.approx(0.5 + HEALTH_AIRTIME_S + CHUNK_AIRTIME_S)
    ]


def test_lidar_that_waited_long_enough_goes_first():
    """No starvation: after 2.5 s of waiting, the LiDAR chunk's score (2.5) beats a
    new Health packet's head start (2.0), so the chunk goes first."""
    transport = new_transport()
    transport.add_to_outbox(LIDAR, "relay_1", "ground", 1200, 0.0)
    transport.add_to_outbox(HEALTH, "relay_1", "ground", 44, 2.5)
    transport.move_packets(2.5, 2.6, CHAIN, ONE_MBPS)
    assert transport.delays_s[HEALTH] == [
        pytest.approx(CHUNK_AIRTIME_S + HEALTH_AIRTIME_S)
    ]


def test_commands_go_first():
    """The spec's urgent kinds (P0) outrank telemetry, which outranks movement
    commands, which outrank LiDAR."""
    assert HEAD_START_S[COMMAND] > HEAD_START_S[HEALTH] > HEAD_START_S[REPOSITION]
    assert HEAD_START_S[REPOSITION] > HEAD_START_S[LIDAR]
    assert HEAD_START_S[COMMAND_ACK] == HEAD_START_S[POSE_CORRECTION] == 3.0


def test_two_hops_use_twice_the_airtime():
    """One talker at a time: a 2-hop packet takes 2 airtimes, and 10 chunks over 2
    hops take as long as 20 chunks over 1 hop (the team's "2 hops halve the
    speed"), plus 1 slot: the relay draws 1 slot while the mapper is still
    sending (FirstNameWins), and counts it down once the mapper is done. The 10
    chunks form one keyframe, complete when the last one arrives."""
    transport = new_transport()
    transport.add_to_outbox(HEALTH, "mapper_1", "ground", 44, 0.0)
    transport.move_packets(0.0, 0.1, CHAIN, ONE_MBPS)
    assert transport.delays_s[HEALTH] == [pytest.approx(2 * HEALTH_AIRTIME_S)]

    transport = new_transport()
    for _ in range(10):
        transport.add_to_outbox(LIDAR, "mapper_1", "ground", 1200, 0.0)
    transport.move_packets(0.0, 1.0, CHAIN, ONE_MBPS)
    last_s = 20 * CHUNK_AIRTIME_S + SLOT_S
    assert len(transport.delays_s[LIDAR]) == 10
    assert max(transport.delays_s[LIDAR]) == pytest.approx(last_s)
    assert transport.keyframe_delays_s == [pytest.approx(last_s)]


def test_summary_reports_health_per_drone():
    """Both drones send one Health packet at once over 1 Mb/s links. The summary
    keeps their trips apart. The mapper wins the first turn (FirstNameWins: it
    draws 0 slots, the relay 1). The relay then counts down its last slot and
    sends its own 1-hop packet (2 airtimes + 1 slot in all), and then the
    mapper's 2-hop packet arrives (3 airtimes + 1 slot)."""
    transport = new_transport()
    transport.add_to_outbox(HEALTH, "relay_1", "ground", 44, 0.0)
    transport.add_to_outbox(HEALTH, "mapper_1", "ground", 44, 0.0)
    transport.move_packets(0.0, 0.1, CHAIN, ONE_MBPS)
    by_drone = transport.summary()[HEALTH]["delay_ms_by_drone"]
    relay_s = 2 * HEALTH_AIRTIME_S + SLOT_S
    mapper_s = 3 * HEALTH_AIRTIME_S + SLOT_S
    assert by_drone["relay_1"]["max"] == pytest.approx(relay_s * 1000, abs=1e-3)
    assert by_drone["mapper_1"]["max"] == pytest.approx(mapper_s * 1000, abs=1e-3)


def test_a_lone_drone_waits_out_its_countdown():
    """Even with nobody else talking, a drone first counts down its random number
    of slots (here 7 slots of 9 microseconds), then sends."""
    transport = new_transport(EveryoneDrawsSeven())
    transport.add_to_outbox(HEALTH, "relay_1", "ground", 44, 0.0)
    transport.move_packets(0.0, 0.1, CHAIN, ONE_MBPS)
    assert transport.delays_s[HEALTH] == [pytest.approx(7 * SLOT_S + HEALTH_AIRTIME_S)]


def test_drones_that_end_their_countdowns_together_collide():
    """Two drones that draw the same countdown talk at once: the channel is busy
    for the airtime, nothing arrives, both packets stay to try again, and both
    drones double their countdown range (16 -> 32 slots)."""
    transport = new_transport(EveryoneDrawsSeven())
    transport.add_to_outbox(HEALTH, "relay_1", "ground", 44, 0.0)
    transport.add_to_outbox(HEALTH, "mapper_1", "ground", 44, 0.0)
    transport.move_packets(0.0, 1e-6, CHAIN, ONE_MBPS)  # time for one turn only
    assert transport.collisions == 1
    assert transport.delays_s[HEALTH] == []
    assert transport.summary()[HEALTH]["waiting"] == 2
    assert transport.slots["relay_1"] == transport.slots["mapper_1"] == 32
    assert transport.channel_free_s == pytest.approx(7 * SLOT_S + HEALTH_AIRTIME_S)


def test_a_command_comes_back_acknowledged():
    """A command to the mapper crosses 2 hops, the mapper's ACK crosses 2 hops
    back, so the round trip is 4 command airtimes, and nothing is resent."""
    transport = new_transport()
    transport.issue_command("ground", "mapper_1", 0.0)
    transport.move_packets(0.0, 0.1, CHAIN, ONE_MBPS)
    assert transport.round_trips_s["mapper_1"] == [pytest.approx(4 * COMMAND_AIRTIME_S)]
    assert transport.unacked == {}
    assert transport.resends == 0


def test_a_command_without_an_ack_is_resent_then_given_up():
    """With the mapper cut off, the command (and each copy) is dropped. Every
    step, the base finds the ACK more than 50 ms late and sends it again; after
    5 resends it gives up, and the command counts as failed."""
    transport = new_transport()
    transport.issue_command("ground", "mapper_1", 0.0)
    for tick in range(7):  # 0.7 s, one 0.1 s step at a time
        transport.move_packets(tick * 0.1, (tick + 1) * 0.1, BROKEN, ONE_MBPS)
    assert transport.resends == 5
    assert transport.dropped[COMMAND] == 6  # the first send and 5 resends
    assert transport.commands_failed == 1
    assert transport.unacked == {}


def test_a_lost_chunk_spoils_its_whole_keyframe():
    """3 chunks of a keyframe reach the relay, then the mapper-relay link breaks.
    The 6 chunks still at the mapper have no route and are dropped, and so are
    the 3 at the relay, although the relay could still reach the base: the base
    can't use part of a keyframe, so sending them would waste airtime."""
    transport = new_transport()
    transport.add_keyframe("mapper_1", "ground", 0.0)
    transport.move_packets(0.0, 0.025, CHAIN, ONE_MBPS)  # 3 chunks: 0, 10.6, 21.2 ms
    assert len(transport.outboxes["relay_1"]) == 3
    transport.move_packets(0.1, 0.2, BROKEN, ONE_MBPS)
    assert transport.dropped[LIDAR] == 9
    assert transport.keyframes_lost == 1
    assert transport.outboxes["relay_1"] == []
    assert transport.delays_s[LIDAR] == []


def test_packets_without_a_route_are_dropped():
    """With the mapper-relay link down, the mapper's packet is dropped and
    counted, like batman-adv drops packets it has no route for."""
    transport = new_transport()
    transport.add_to_outbox(HEALTH, "mapper_1", "ground", 44, 0.0)
    transport.move_packets(0.0, 0.1, BROKEN, ONE_MBPS)
    assert transport.dropped[HEALTH] == 1
    assert transport.summary()[HEALTH]["waiting"] == 0


def test_simulator_makes_and_accounts_for_every_packet():
    """10 s of the default scene (3 drones, connected the whole time): the right
    number of packets is made, and each is delivered or still on its way. Every
    keyframe and command is complete, or still on its way."""
    config = load_config(ROOT / "configs/network.yaml")
    config["duration_s"] = 10.0
    sim = Simulator(config)
    sim.run()
    summary = sim.transport.summary()
    # 100 steps of 0.1 s. Health: 3 drones x 10 Hz. Reposition: 2 relays x 5 Hz.
    # LiDAR: 2 keyframes per second x 9 chunks. Degeneracy: 10 Hz. Pose
    # corrections: 1 Hz. Commands: 3 drones x one every 5 s.
    assert summary[HEALTH]["made"] == 300
    assert summary[REPOSITION]["made"] == 100
    assert summary[LIDAR]["made"] == 180
    assert summary[DEGENERACY]["made"] == 100
    assert summary[POSE_CORRECTION]["made"] == 10
    assert summary[COMMAND]["round_trips"]["issued"] == 6
    for kind in HEAD_START_S:
        counts = summary[kind]
        assert counts["dropped_no_route"] == 0
        assert counts["made"] == counts["delivered"] + counts["waiting"]
    keyframes = summary[LIDAR]["keyframes"]
    assert keyframes["made"] == 20
    assert keyframes["made"] == keyframes["delivered_whole"] + keyframes["in_flight"]
    commands = summary[COMMAND]["round_trips"]
    assert commands["issued"] == (
        commands["acknowledged"] + commands["failed"] + commands["waiting_for_ack"]
    )
    assert set(summary[HEALTH]["delay_ms_by_drone"]) == {
        "mapper_1",
        "relay_1",
        "relay_2",
    }


def test_same_seed_same_packets():
    """The random countdowns come from the config's seed: the same seed repeats
    a run exactly, and another seed gives different timings."""
    config = load_config(ROOT / "configs/network.yaml")
    config["duration_s"] = 3.0
    first, second = Simulator(config), Simulator(config)
    first.run()
    second.run()
    assert first.transport.delays_s == second.transport.delays_s
    config["seed"] = 8
    other = Simulator(config)
    other.run()
    assert other.transport.delays_s != first.transport.delays_s
