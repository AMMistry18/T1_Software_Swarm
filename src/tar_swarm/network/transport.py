"""Packet transport: how packets wait their turn and hop toward their destination.

Every drone (and the base station) has an outbox: a waiting line of packets to
send. All radios share one Wi-Fi channel and only one can talk at a time. They
take turns the way real Wi-Fi does: each drone with something to send counts down
a random number of tiny time slots, and the first to reach zero talks.
On its turn, a drone sends the packet in its outbox with the highest
score/priority:

    score = head start for packet type + seconds it has waited in this outbox

Urgent commands (P0) have the biggest head start, then Health (TELEM),
then Reposition (MOV), then LiDAR. 

Each turn moves one packet one hop along the fewest-hops route (routing.py). A turn
lasts the packet's airtime: a fixed cost per packet (the gaps between frames, the
preamble, the receiver's ACK) plus the packet's bits divided by the link speed.
Nobody else talks meanwhile, so a packet that needs 2 hops uses twice the airtime:
the team's "2 hops halve the speed".

A link's speed depends on how far apart its two drones are: the farther apart, the
weaker the signal.

Two kinds of packet need more than one delivery to be useful:
- A command (C2) must be acknowledged: the drone sends back an ACK, and if none
  reaches the base within 50 ms, the base sends the command again, up to 5 times.
- A LiDAR keyframe travels as several chunks, and the base can use it only once
  every chunk has arrived. So when a chunk has to be dropped, the rest of its
  keyframe is dropped too, instead of wasting airtime.

Packet sizes are whole UDP datagrams, including their 20-byte header, from the
spec's formats in messages.py.
"""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from tar_swarm.network import messages
from tar_swarm.network.routing import next_hops

# The kinds of packet, with the spec's message type numbers.
HEALTH = "HEALTH"  # 0x01, drone -> base: position, battery, link quality
LIDAR = "LIDAR"  # 0x02, mapping drone -> base: one chunk of a LiDAR keyframe
COMMAND = "COMMAND"  # 0x03 (C2), base -> drone: launch, hold, dock, emergency stop
COMMAND_ACK = "COMMAND_ACK"  # 0x03 sent back by the drone: "got your command"
POSE_CORRECTION = "POSE_CORRECTION"  # 0x04, base -> mapping drone: SLAM's fixed pose
REPOSITION = "REPOSITION"  # 0x05, base -> network drone: where to fly
DEGENERACY = "DEGENERACY"  # 0x06, mapping drone -> base: optical flow, LiDAR failing

# Head start in the score calculation (seconds). The spec's urgent P0 kinds go
# first, then telemetry (Health, Degeneracy), which the team ranks above movement
# (Reposition), then LiDAR.
HEAD_START_S = {
    COMMAND: 3.0,
    COMMAND_ACK: 3.0,
    POSE_CORRECTION: 3.0,
    HEALTH: 2.0,
    DEGENERACY: 2.0,
    REPOSITION: 1.0,
    LIDAR: 0.0,
}

# How often each kind is made, per drone (the busiest end of the spec's ranges).
# LiDAR counts keyframes; each keyframe is cut into several chunks. The spec calls
# commands, pose corrections, and degeneracy "events" and gives no rate, so these
# are placeholders: a command for each drone every 5 s, a pose correction every
# second, and the mapping drone's LiDAR degenerate the whole time (the worst
# case). A command's ACK is made when the command arrives.
RATE_HZ = {
    HEALTH: 10.0,
    REPOSITION: 5.0,
    LIDAR: 2.0,
    COMMAND: 0.2,
    POSE_CORRECTION: 1.0,
    DEGENERACY: 10.0,
}

# Commands: the spec wants the ACK back within 50 ms; if it isn't, the base sends
# the command again, up to 5 times.
ACK_WAIT_S = 0.05
MAX_RESENDS = 5

# Sizes in bytes: whole UDP datagrams (20-byte header + payload), from the spec's
# formats in messages.py.
SIZE_BYTES = {
    HEALTH: messages.datagram_bytes(messages.HEALTH),  # 44
    COMMAND: messages.datagram_bytes(messages.COMMAND),  # 32
    COMMAND_ACK: messages.datagram_bytes(messages.COMMAND),  # 32
    POSE_CORRECTION: messages.datagram_bytes(messages.POSE_CORRECTION),  # 52
    REPOSITION: messages.datagram_bytes(messages.RELAY_REPOSITION),  # 36
    DEGENERACY: messages.datagram_bytes(messages.DEGENERACY),  # 40
}
KEYFRAME_BYTES = 10_000  # one compressed LiDAR keyframe (spec: 6-10 KB)
# Bytes the layers under our program add to every packet: UDP 8 + IP 20 +
# Ethernet 14 + batman-adv 10 + Wi-Fi 38.
LOWER_LAYER_BYTES = 90

# Wi-Fi's turn-taking, with the 802.11 standard's numbers for ordinary data: a
# drone waits a random 0 to 15 slots of 9 microseconds before it talks. After a
# collision it doubles the range (0-31, 0-63, ... up to 0-1023); after a packet
# gets through, it goes back to 0-15.
SLOT_S = 9e-6
MIN_SLOTS = 16
MAX_SLOTS = 1024


def keyframe_chunk_sizes(keyframe_bytes=KEYFRAME_BYTES):
    """Datagram sizes, in bytes, of the chunks one LiDAR keyframe is cut into.

    Each chunk holds up to 1172 bytes of map data, so with its 28 header bytes it
    fits the 1200-byte cap. 10,000 bytes -> eight chunks of 1200 and one of 652.
    """
    data_bytes = messages.MAX_CHUNK_DATA_BYTES
    full, rest = divmod(keyframe_bytes, data_bytes)  # whole chunks, leftover bytes
    sizes = [messages.MAX_DATAGRAM_BYTES] * full
    if rest:
        sizes.append(messages.CHUNK_HEADER_BYTES + rest)
    return sizes


@dataclass
class Packet:
    """One packet on its way. waiting_since_s changes at every hop into another
    drone's inbox."""

    kind: str
    source: str # name of drone that made the packet
    destination: str # name of drone the packet is trying to reach
    size_bytes: int
    created_s: float
    waiting_since_s: float  # when it joined the outbox it's in now
    frame_id: int = 0  # LiDAR: which keyframe this chunk belongs to
    cmd_seq: int = 0  # commands and ACKs: which command this is about


@dataclass
class Command:
    """A command the base has sent and is waiting to hear back about."""

    source: str  # the base
    destination: str
    created_s: float  # when the base first sent it
    sent_s: float  # when the base last sent a copy
    resends: int = 0


def score(packet, now_s):
    """The priority score: head start + seconds waited in the current outbox."""
    return HEAD_START_S[packet.kind] + now_s - packet.waiting_since_s


def delay_stats_ms(delays_s):
    """The median, 95th percentile (95% arrived at least this fast), and worst of
    a list of delays in seconds, all in milliseconds; None if the list is empty."""
    if not delays_s:
        return None
    delays_ms = np.array(delays_s) * 1000
    return {
        "median": round(float(np.percentile(delays_ms, 50)), 3),
        "p95": round(float(np.percentile(delays_ms, 95)), 3),
        "max": round(float(delays_ms.max()), 3),
    }


def by_drone_ms(delays_s_by_drone):
    """delay_stats_ms for each drone, in alphabetical order of their IDs."""
    return {
        drone: delay_stats_ms(delays_s_by_drone[drone])
        for drone in sorted(delays_s_by_drone)
    }


class Transport:
    """Every drone's outbox, the shared radio channel, and delivery statistics."""

    def __init__(self, settings, dt_s, rng):
        """settings: the config's `packets` section. dt_s: the simulator's step.
        rng: numpy's random number generator, for Wi-Fi's random countdowns."""
        self.speed_zones = settings["link_speed_zones"]  # nearest zone first
        self.overhead_s = settings["per_packet_overhead_s"]
        self.dt_s = dt_s
        self.rng = rng

        # defaultdict(list) hands out an empty list the first time a drone is used.
        self.outboxes = defaultdict(list)  # drone ID -> its packets, oldest first
        self.channel_free_s = 0.0  # when the transmission on the air ends
        # Each drone's countdown range, in slots. (lambda: a one-line function;
        # here it gives every drone the starting range.)
        self.slots = defaultdict(lambda: MIN_SLOTS)
        self.countdowns = {}  # drone ID -> slots left on its current countdown
        self.unacked = {}  # command number -> Command, until its ACK arrives
        self.chunks_missing = Counter()  # keyframe -> its chunks not at the base yet

        # Statistics; a Counter starts every count at 0.
        self.made = Counter()  # by kind
        self.dropped = Counter()  # by kind
        self.delays_s = defaultdict(list)  # by kind: seconds from made to delivered
        self.health_delays_s = defaultdict(list)  # the same for Health, by sender
        self.keyframes_made = 0
        self.keyframes_lost = 0
        self.keyframe_delays_s = []  # keyframe made -> its last chunk at the base
        self.commands_made = 0
        self.commands_failed = 0  # still no ACK after MAX_RESENDS resends
        self.resends = 0
        # By the command's drone: seconds from a command made to its ACK back.
        self.round_trips_s = defaultdict(list)
        self.collisions = 0

    def airtime_s(self, size_bytes, speed_bps):
        """Seconds the channel is busy sending one packet of size_bytes over one
        link that runs at speed_bps (bits per second)."""
        bits = (size_bytes + LOWER_LAYER_BYTES) * 8
        return self.overhead_s + bits / speed_bps

    def speed_bps(self, distance_m, range_m):
        """The speed, in bits per second, of one link whose two drones are
        distance_m apart and whose radio range (the smaller of the two drones') is
        range_m.

        The first zone that reaches that far sets the speed. With the default zones
        and a 300 m range: 50 m apart -> 30 Mb/s, 150 m -> 10 Mb/s, 250 m -> 1 Mb/s.
        """
        reach = distance_m / range_m  # 0 = side by side, 1 = at the edge of range
        for zone in self.speed_zones:
            if reach <= zone["up_to_range_fraction"]:
                return zone["mbps"] * 1e6
        # Links are checked only every graph_interval_s, so drones can drift a bit
        # past the range before their link breaks: use the last zone's speed.
        return self.speed_zones[-1]["mbps"] * 1e6

    def link_speeds_bps(self, graph, roles, positions, range_m):
        """Every working link's speed in bits per second, the same both ways.

        Returns a dict keyed by (sender, receiver); speeds[a, b] is Python's short
        way of writing speeds[(a, b)].
        """
        speeds = {}
        for a, b in graph.edges:
            distance_m = np.linalg.norm(positions[a] - positions[b])
            limit_m = min(range_m[roles[a]], range_m[roles[b]])  # as in RangeLink
            speeds[a, b] = speeds[b, a] = self.speed_bps(distance_m, limit_m)
        return speeds

    def due(self, kind, tick):
        """The simulator only moves forward in steps of 0.1 s, but each stream
        has its own rate. due() turns "packets per second" into "packets this step"

        Count the packets the stream should have made by the end of this step,
        minus those it should have made by the start. Example: LiDAR at 2 Hz with
        0.1 s steps should have made 0.8 keyframes by 0.4 s (0 whole ones) and 1.0
        by 0.5 s, so step 4 makes 1 and steps 0-3 make none.

        The + 1e-9 guards against round-off: computers store 0.3 slightly off, so
        9 * 0.3 * 10 gives 26.999999999999996, which floor (round down) makes 26.
        """
        hz = RATE_HZ[kind]
        before = math.floor(tick * self.dt_s * hz + 1e-9)
        after = math.floor((tick + 1) * self.dt_s * hz + 1e-9)
        return after - before

    # Making packets

    def add_to_outbox(
        self, kind, source, destination, size_bytes, time_s, frame_id=0, cmd_seq=0
    ):
        """Put a new packet at the back of its source's outbox."""
        packet = Packet(
            kind, source, destination, size_bytes, time_s, time_s, frame_id, cmd_seq
        )

        # dictionary that maps each drone's ID to a list of Packet objects (i.e.
        # the packets in its queue); outboxes remain small so a list is used
        # over a priority queue for now
        self.outboxes[source].append(packet)
        self.made[kind] += 1
        if kind == LIDAR:
            self.chunks_missing[frame_id] += 1

    def add_keyframe(self, source, destination, time_s):
        """Cut a new LiDAR keyframe into chunks, all into source's outbox."""
        self.keyframes_made += 1
        frame_id = self.keyframes_made  # keyframes are numbered 1, 2, 3, ...
        for size in keyframe_chunk_sizes():
            self.add_to_outbox(LIDAR, source, destination, size, time_s, frame_id)

    def issue_command(self, source, destination, time_s):
        """Send a new command, and remember it until its ACK comes back."""
        self.commands_made += 1
        cmd_seq = self.commands_made  # commands are numbered 1, 2, 3, ...
        self.unacked[cmd_seq] = Command(source, destination, time_s, time_s)
        size = SIZE_BYTES[COMMAND]
        self.add_to_outbox(COMMAND, source, destination, size, time_s, cmd_seq=cmd_seq)

    def make(self, kind, source, destination, tick, time_s):
        """Make the packets of `kind` due at step `tick` (see due)."""
        for _ in range(self.due(kind, tick)):
            if kind == LIDAR:
                self.add_keyframe(source, destination, time_s)
            elif kind == COMMAND:
                self.issue_command(source, destination, time_s)
            else:
                self.add_to_outbox(kind, source, destination, SIZE_BYTES[kind], time_s)

    def step(self, tick, graph, roles, active, positions, range_m):
        """Make the packets due at step `tick`, then let the radios talk until the
        step ends.

        graph: a GraphSnapshot of the links that work during this step.
        roles, active, positions: each drone's role, on/off state, and 3D position
        in meters, by ID. range_m: the radio range of each role, in meters.
        """
        start_s = tick * self.dt_s
        base = next(drone for drone in graph.nodes if roles[drone] == "ground_station")
        for drone in graph.nodes:
            role = roles[drone]
            if role == "ground_station" or not active[drone]:
                continue
            self.make(HEALTH, drone, base, tick, start_s)
            self.make(COMMAND, base, drone, tick, start_s)
            if role == "relay":
                self.make(REPOSITION, base, drone, tick, start_s)
            if role == "mapper":
                self.make(LIDAR, drone, base, tick, start_s)
                self.make(DEGENERACY, drone, base, tick, start_s)
                self.make(POSE_CORRECTION, base, drone, tick, start_s)
        speeds = self.link_speeds_bps(graph, roles, positions, range_m)
        self.move_packets(start_s, start_s + self.dt_s, graph, speeds)

    # Sending packets

    def move_packets(self, start_s, end_s, graph, speeds):
        """Let the drones take turns sending, from start_s until end_s.

        speeds: each link's speed, from link_speeds_bps.

        A transmission that starts before end_s may finish after it; the next call
        then waits for it to end.

        asks routing.py for each drone's next hop and drops any packet that has
        no route, the same way batman-adv would
        """
        # routes[destination][drone] = the drone's next hop toward that destination.
        routes = {d: next_hops(graph, d) for d in graph.nodes}
        now = max(self.channel_free_s, start_s)
        while now < end_s:
            # Before every turn: resend late commands, drop what can't get through.
            self.resend_late_commands(now)
            self.drop_unroutable(routes)
            talkers = sorted(drone for drone, box in self.outboxes.items() if box)
            if not talkers:
                break  # nothing to send: the channel sits idle until the next step
            wait_s, winners = self.take_turn(talkers)
            now += wait_s
            sends = []  # (talker, packet, next hop) for each drone talking now
            for talker in winners:
                # The highest score goes. max returns the first of equal scores, so
                # chunks made together keep their order. (lambda = a one-line function.)
                packet = max(self.outboxes[talker], key=lambda p: score(p, now))
                sends.append((talker, packet, routes[packet.destination][talker]))
            # The channel stays busy until the longest of these transmissions ends.
            now += max(
                self.airtime_s(packet.size_bytes, speeds[talker, next_hop])
                for talker, packet, next_hop in sends
            )
            if len(sends) > 1:
                # A collision: nothing gets through. The packets stay in their
                # outboxes, and each of these drones doubles its countdown range.
                self.collisions += 1
                for talker in winners:
                    self.slots[talker] = min(2 * self.slots[talker], MAX_SLOTS)
                continue
            talker, packet, next_hop = sends[0]
            self.slots[talker] = MIN_SLOTS
            self.outboxes[talker].remove(packet)
            if next_hop == packet.destination:
                self.arrive(packet, now)
            else:
                packet.waiting_since_s = now
                self.outboxes[next_hop].append(packet)
        self.channel_free_s = now

    def take_turn(self, talkers):
        """Wi-Fi's turn-taking. A drone starting a new try draws a random
        countdown, 0 up to its range minus 1 slots, and the lowest countdown talks
        first. The others counted down too while they waited, so they keep only
        the slots they have left: the longer a drone waits, the closer its turn.

        Returns the wait in seconds and the drones whose countdowns ended first:
        usually one, but if two or more end together, they all talk at once and
        collide.
        """
        new = [drone for drone in talkers if drone not in self.countdowns]
        if new:
            draws = self.rng.integers(0, [self.slots[drone] for drone in new])
            self.countdowns.update(zip(new, draws.tolist()))  # drone -> its draw
        lowest = min(self.countdowns[drone] for drone in talkers)
        winners = [drone for drone in talkers if self.countdowns[drone] == lowest]
        for drone in talkers:
            self.countdowns[drone] -= lowest
        for drone in winners:
            del self.countdowns[drone]  # its next try draws a new countdown
        return lowest * SLOT_S, winners

    def arrive(self, packet, now_s):
        """A packet reached its destination: record how long it took, then do what
        the destination does with it."""
        delay_s = now_s - packet.created_s
        self.delays_s[packet.kind].append(delay_s)
        if packet.kind == HEALTH:
            self.health_delays_s[packet.source].append(delay_s)
        elif packet.kind == COMMAND:  # the drone answers right away
            self.add_to_outbox(
                COMMAND_ACK,
                packet.destination,
                packet.source,
                SIZE_BYTES[COMMAND_ACK],
                now_s,
                cmd_seq=packet.cmd_seq,
            )
        elif packet.kind == COMMAND_ACK:
            # pop(key, None) gives None if an earlier copy's ACK already answered
            # the command (or the base gave up on it).
            command = self.unacked.pop(packet.cmd_seq, None)
            if command is not None:
                round_trip_s = now_s - command.created_s
                self.round_trips_s[command.destination].append(round_trip_s)
        elif packet.kind == LIDAR:
            self.chunks_missing[packet.frame_id] -= 1
            if self.chunks_missing[packet.frame_id] == 0:  # the whole keyframe is in
                del self.chunks_missing[packet.frame_id]
                self.keyframe_delays_s.append(delay_s)

    # Commands that never got an ACK, and packets that can't get through
    def resend_late_commands(self, now_s):
        """Send each command again whose ACK is more than ACK_WAIT_S late. After
        MAX_RESENDS resends, give up on it: it counts as failed."""
        for cmd_seq, command in list(self.unacked.items()):  # list(): we may delete
            if now_s - command.sent_s < ACK_WAIT_S:
                continue
            if command.resends == MAX_RESENDS:
                del self.unacked[cmd_seq]
                self.commands_failed += 1
                continue
            command.resends += 1
            command.sent_s = now_s
            self.resends += 1
            size = SIZE_BYTES[COMMAND]
            source, destination = command.source, command.destination
            self.add_to_outbox(COMMAND, source, destination, size, now_s, 0, cmd_seq)

    def drop_unroutable(self, routes):
        """Drop every packet whose drone has no route to the packet's destination,
        as batman-adv does. A dropped LiDAR chunk takes the rest of its keyframe
        with it (see drop_keyframes)."""
        spoiled = set()  # keyframes that just lost a chunk
        for drone, box in self.outboxes.items():
            no_route = [p for p in box if drone not in routes.get(p.destination, {})]
            for packet in no_route:
                box.remove(packet)
                self.dropped[packet.kind] += 1
                if packet.kind == LIDAR:
                    spoiled.add(packet.frame_id)
        if spoiled:
            self.drop_keyframes(spoiled)

    def drop_keyframes(self, frame_ids):
        """Throw away every chunk of these keyframes still waiting anywhere. The
        base can rebuild a keyframe only from all of its chunks, so once one is
        lost, sending the others would only waste airtime."""
        for box in self.outboxes.values():
            rest = [p for p in box if p.kind == LIDAR and p.frame_id in frame_ids]
            for packet in rest:
                box.remove(packet)
                self.dropped[LIDAR] += 1
        for frame_id in frame_ids:
            del self.chunks_missing[frame_id]
            self.keyframes_lost += 1

    def summary(self):
        """Per kind: packets made, delivered, dropped (no route), and still waiting,
        plus how long delivered packets took (see delay_stats_ms). Commands count
        every copy, resends included.

        Some kinds get more:
        - Health: delays per sending drone, because each drone's trip is different
          (a relay next to the base is 1 hop away, a mapper 2 or more).
        - LiDAR: whole keyframes, since the base can only use complete ones.
        - Commands: how many got their ACK, resends, failures, and the round trip
          per drone, for the same reason as Health.
        Plus the number of collisions on the channel.
        """
        waiting = Counter(p.kind for box in self.outboxes.values() for p in box)
        result = {}
        for kind in HEAD_START_S:
            result[kind] = {
                "made": self.made[kind],
                "delivered": len(self.delays_s[kind]),
                "dropped_no_route": self.dropped[kind],
                "waiting": waiting[kind],
                "delay_ms": delay_stats_ms(self.delays_s[kind]),
            }
        result[HEALTH]["delay_ms_by_drone"] = by_drone_ms(self.health_delays_s)
        result[LIDAR]["keyframes"] = {
            "made": self.keyframes_made,
            "delivered_whole": len(self.keyframe_delays_s),
            "lost": self.keyframes_lost,
            "in_flight": len(self.chunks_missing),
            "delay_ms": delay_stats_ms(self.keyframe_delays_s),
        }
        result[COMMAND]["round_trips"] = {
            "issued": self.commands_made,
            "acknowledged": sum(len(trips) for trips in self.round_trips_s.values()),
            "failed": self.commands_failed,
            "waiting_for_ack": len(self.unacked),
            "resends": self.resends,
            "round_trip_ms_by_drone": by_drone_ms(self.round_trips_s),
        }
        result["collisions"] = self.collisions
        return result
