"""Message formats: what the base station and the drones say to each other.

Every message is a packet. The outside (`Packet`) looks the same
for every message: its ID, who sent it, who it's for, what kind it is, and when it
was sent. The inside (`contents`) depends on the kind of packet:

    MOVE   base -> one drone   "fly to this point"
    TELEM  drone -> base       "here is where I am and how fast I'm moving"
    ACK    drone -> base       "your MOVE arrived" (ACK = acknowledgment)

This file only describes messages. Sending, queuing, and routing them come in later
steps. Positions are meters measured from the base station: x east, y north, z up.
Those are the network simulator's axes, and its demo puts the base at (0, 0, 0).
"""

from dataclasses import dataclass

# The three kinds of packet
MOVE = "MOVE"
TELEM = "TELEM"
ACK = "ACK"


# @dataclass writes the routine parts of a class for us: each annotated line below
# becomes a constructor argument, and print() shows every field. frozen=True seals
# the object so it can't be changed later.
@dataclass(frozen=True)
class MoveContents:
    """Inside a MOVE: target_m = the (x, y, z) coordinate to fly to, in meters.
    """

    # three decimal numbers, in order.
    target_m: tuple[float, float, float]


@dataclass(frozen=True)
class TelemContents:
    """Inside a TELEM: the sender's latest status report.

    position_m:   (x, y, z) in meters from the base station.
    velocity_m_s: (x, y, z) speed in meters per second.
    Attitude, battery, and link scores get added when the simulator models them.
    """

    position_m: tuple[float, float, float]
    velocity_m_s: tuple[float, float, float]


@dataclass(frozen=True)
class AckContents:
    """Inside an ACK: confirms_packet_id = the packet_id of the MOVE that arrived.

    An ACK means "your MOVE arrived", not "I finished flying there"; TELEM shows
    that.
    """

    confirms_packet_id: int


# Which contents belong inside which kind of packet.
PACKET_TYPE_CONTENTS = {MOVE: MoveContents, TELEM: TelemContents, ACK: AckContents}


@dataclass(frozen=True)
class Packet:
    """One message: the label every packet carries, plus its specific contents.

    packet_id:   the sender's own counter, 1, 2, 3, ..., over all its packets. Two
                 senders can both have a packet 7, so (source, packet_id) is what
                 names exactly one packet.
    source:      who sent it, e.g. "ground" (the base), "relay_1", "mapper_1".
    destination: who it's for. On real drones, batman-adv finds the way there.
    kind:        MOVE, TELEM, or ACK. It tells the receiver how to treat the packet.
    sent_at_s:   the sender's clock when it made the packet, in seconds.
    contents:    MoveContents, TelemContents, or AckContents corresponding to 
                 the packet type
    """

    packet_id: int
    source: str
    destination: str
    kind: str
    sent_at_s: float
    contents: MoveContents | TelemContents | AckContents  # "|" means "or"

    def __post_init__(self):
        """Refuse a mislabeled envelope, such as a MOVE label with telemetry inside.

        @dataclass runs this right after filling in the fields. Raises ValueError.
        """
        expected = PACKET_TYPE_CONTENTS.get(self.kind)  # .get gives None if unknown
        # f"..." fills in each {...}; !r shows the text in quotes.
        if expected is None:
            raise ValueError(f"unknown kind {self.kind!r}; use MOVE, TELEM, or ACK")
        # isinstance checks which class an object was made from.
        if not isinstance(self.contents, expected):
            raise ValueError(f"a {self.kind} packet must contain {expected.__name__}")
