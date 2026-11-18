"""Packet transport: how packets wait their turn and hop toward their destination.

Every drone (and the base station) has an outbox: a waiting line of packets to
send. All radios share one Wi-Fi channel and only one can talk at a time. Real
radios work out whose turn it is by themselves (listen first, then wait a random
moment); this simulation simplifies that to round robin over the drones that have
something to send. On its turn, a drone sends the packet in its outbox with the
highest score/priority:

    score = head start for packet type + seconds it has waited in this outbox

Health (TELEM) has the biggest head start, then Reposition (MOV), then LiDAR. A
waiting packet's score keeps growing, so LiDAR always gets through eventually (no
starvation).

Each turn moves one packet one hop along the fewest-hops route (routing.py). A turn
lasts the packet's airtime: a fixed cost per packet (waiting for a clear channel,
the preamble, the receiver's ACK) plus the packet's bits divided by the link speed.
Nobody else talks meanwhile, so a packet that needs 2 hops uses twice the airtime:
the team's "2 hops halve the speed".

Packet sizes are whole UDP datagrams, including its 20-byte header. Only
the MVP's steady streams are modeled so far. Also simplified for now: every link
has the same speed; Health goes to the base only; a packet
with no route is dropped, as batman-adv does; and relays use the score too, while
on real drones Linux forwards by the packet's priority tag.
"""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from tar_swarm.network.routing import next_hops

# The kinds of packet modeled so far (spec message types 0x01, 0x05, and 0x02).
HEALTH = "HEALTH"  # drone -> base: position, battery, link quality
REPOSITION = "REPOSITION"  # base -> network drone: where to fly
LIDAR = "LIDAR"  # mapping drone -> base: one chunk of a LiDAR keyframe

# Head start in the score calculation (seconds)
HEAD_START_S = {HEALTH: 2.0, REPOSITION: 1.0, LIDAR: 0.0}

# How often each stream sends (the busiest end of the range). LiDAR counts
# keyframes; each keyframe is cut into several chunks.
RATE_HZ = {HEALTH: 10.0, REPOSITION: 5.0, LIDAR: 2.0}

# Sizes in bytes, from the spec sheet.
HEALTH_BYTES = 44  # 20-byte header + 24-byte payload
REPOSITION_BYTES = 36  # 20-byte header + 16-byte payload
KEYFRAME_BYTES = 10_000  # one compressed LiDAR keyframe (spec: 6-10 KB)
CHUNK_HEADER_BYTES = 28  # 20-byte header + 8-byte chunk sub-header
MAX_DATAGRAM_BYTES = 1200  # the spec sheet's cap, so no packet gets split on the way
# Bytes the layers under our program add to every packet: UDP 8 + IP 20 +
# Ethernet 14 + batman-adv 10 + Wi-Fi 38.
LOWER_LAYER_BYTES = 90


def keyframe_chunk_sizes(keyframe_bytes=KEYFRAME_BYTES):
    """Datagram sizes, in bytes, of the chunks one LiDAR keyframe is cut into.

    Each chunk holds up to 1172 bytes of map data, so with its 28 header bytes it
    fits the 1200-byte cap. 10,000 bytes -> eight chunks of 1200 and one of 652.
    """
    data_bytes = MAX_DATAGRAM_BYTES - CHUNK_HEADER_BYTES
    full, rest = divmod(keyframe_bytes, data_bytes)  # whole chunks, leftover bytes
    sizes = [MAX_DATAGRAM_BYTES] * full
    if rest:
        sizes.append(CHUNK_HEADER_BYTES + rest)
    return sizes


@dataclass
class Packet:
    """One packet on its way. waiting_since_s changes at every step."""

    kind: str
    source: str # name of drone that made the packet 
    destination: str # name of drone the packet is trying to reach
    size_bytes: int
    created_s: float
    waiting_since_s: float  # when it joined the outbox it's in now


def score(packet, now_s):
    """The priority score: head start + seconds waited in the current outbox."""
    return HEAD_START_S[packet.kind] + now_s - packet.waiting_since_s


class Transport:
    """Every drone's outbox, the shared radio channel, and delivery statistics."""

    def __init__(self, settings, dt_s):
        """settings: the config's `packets` section. dt_s: the simulator's step."""
        self.link_rate_bps = settings["link_rate_mbps"] * 1e6
        self.overhead_s = settings["per_packet_overhead_s"]
        self.dt_s = dt_s

        # defaultdict(list) hands out an empty list the first time a drone is used.
        self.outboxes = defaultdict(list)  # drone ID -> its packets, oldest first
        self.channel_free_s = 0.0  # when the transmission on the air ends
        self.last_talker = ""  # who talked last, for taking turns

        # Statistics by kind; starts every count at 0.
        self.made = Counter()
        self.dropped = Counter()
        self.delays_s = defaultdict(list)  # seconds from made to delivered

    def airtime_s(self, size_bytes):
        """Seconds the channel is busy sending one packet of size_bytes, one hop."""
        bits = (size_bytes + LOWER_LAYER_BYTES) * 8
        return self.overhead_s + bits / self.link_rate_bps

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

    def add_to_outbox(self, kind, source, destination, size_bytes, time_s):
        """Put a new packet at the back of its source's outbox."""
        packet = Packet(kind, source, destination, size_bytes, time_s, time_s)

        # dictionary that maps each drone's ID to a list of Packet objects (i.e.
        # the packets in its queue); outboxes remain small so a list is used 
        # over a priority queue for now
        self.outboxes[source].append(packet)
        self.made[kind] += 1

    def step(self, tick, graph, roles, active):
        """Make the packets due at step `tick`, then let the radios talk until the
        step ends.

        graph: a GraphSnapshot of the links that work during this step.
        roles, active: each drone's role and on/off state, by ID.
        """
        start_s = tick * self.dt_s
        base = next(drone for drone in graph.nodes if roles[drone] == "ground_station")
        for drone in graph.nodes:
            role = roles[drone]
            if role == "ground_station" or not active[drone]:
                continue
            for _ in range(self.due(HEALTH, tick)):
                self.add_to_outbox(HEALTH, drone, base, HEALTH_BYTES, start_s)
            if role == "relay":
                for _ in range(self.due(REPOSITION, tick)):
                    self.add_to_outbox(
                        REPOSITION, base, drone, REPOSITION_BYTES, start_s
                    )
            if role == "mapper":
                for _ in range(self.due(LIDAR, tick)):
                    for size in keyframe_chunk_sizes():
                        self.add_to_outbox(LIDAR, drone, base, size, start_s)
        self.move_packets(start_s, start_s + self.dt_s, graph)

    def move_packets(self, start_s, end_s, graph):
        """Let the drones take turns sending, from start_s until end_s.

        Packets with no route are dropped first. A transmission that starts before
        end_s may finish after it; the next call then waits for it to end.

        asks routing.py for each drone's next hop and drops any packet that has 
        no route, the same way batman-adv would
        """
        # routes[destination][drone] = the drone's next hop toward that destination.
        destinations = {p.destination for box in self.outboxes.values() for p in box}
        routes = {d: next_hops(graph, d) for d in destinations if d in graph.nodes}
        for drone, box in self.outboxes.items():
            no_route = [p for p in box if drone not in routes.get(p.destination, {})]
            for packet in no_route:
                box.remove(packet)
                self.dropped[packet.kind] += 1

        now = max(self.channel_free_s, start_s)
        while now < end_s:
            talkers = sorted(drone for drone, box in self.outboxes.items() if box)
            if not talkers:
                break  # nothing to send: the channel sits idle until the next step
            # Round robin, standing in for Wi-Fi's own turn-taking: the first
            # talker after the last one, wrapping around.
            later = [drone for drone in talkers if drone > self.last_talker]
            talker = (later or talkers)[0]
            box = self.outboxes[talker]
            # The highest score goes. max returns the first of equal scores, so
            # chunks made together keep their order. (lambda = a one-line function.)
            packet = max(box, key=lambda p: score(p, now))
            box.remove(packet)
            now += self.airtime_s(packet.size_bytes)
            self.last_talker = talker
            next_hop = routes[packet.destination][talker]
            if next_hop == packet.destination:
                self.delays_s[packet.kind].append(now - packet.created_s)
            else:
                packet.waiting_since_s = now
                self.outboxes[next_hop].append(packet)
        self.channel_free_s = now

    def summary(self):
        """Per kind: packets made, delivered, dropped (no route), and still waiting,
        plus how long delivered packets took, in milliseconds: the median, the 95th
        percentile (95% arrived at least this fast), and the worst."""
        waiting = Counter(p.kind for box in self.outboxes.values() for p in box)
        result = {}
        for kind in RATE_HZ:
            delays_ms = np.array(self.delays_s[kind]) * 1000
            delay = None
            if len(delays_ms):
                delay = {
                    "median": round(float(np.percentile(delays_ms, 50)), 3),
                    "p95": round(float(np.percentile(delays_ms, 95)), 3),
                    "max": round(float(delays_ms.max()), 3),
                }
            result[kind] = {
                "made": self.made[kind],
                "delivered": len(delays_ms),
                "dropped_no_route": self.dropped[kind],
                "waiting": waiting[kind],
                "delay_ms": delay,
            }
        return result
