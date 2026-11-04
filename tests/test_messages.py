"""Tests for network/messages.py: the packet label and what each kind carries."""

import dataclasses

import pytest

from tar_swarm.network.messages import (
    ACK,
    MOVE,
    TELEM,
    AckContents,
    MoveContents,
    Packet,
    TelemContents,
)

# The base's 7th packet: tell the mapper to fly 100 m east, 50 m north, 10 m up.
MOVE_7 = Packet(
    packet_id=7,
    source="ground",
    destination="mapper_1",
    kind=MOVE,
    sent_at_s=12.3,
    contents=MoveContents(target_m=(100.0, 50.0, 10.0)),
)


def test_each_kind_carries_its_own_contents():
    """A correctly labeled MOVE, TELEM, and ACK can each be made and read back."""
    telem = Packet(
        packet_id=41,
        source="mapper_1",
        destination="ground",
        kind=TELEM,
        sent_at_s=12.4,
        contents=TelemContents(position_m=(60.0, 0.0, 10.0), velocity_m_s=(4.0, 0, 0)),
    )
    # The mapper's own 42nd packet is the ACK; inside, it points at the base's MOVE 7.
    ack = Packet(
        packet_id=42,
        source="mapper_1",
        destination="ground",
        kind=ACK,
        sent_at_s=12.5,
        contents=AckContents(confirms_packet_id=MOVE_7.packet_id),
    )
    assert MOVE_7.contents.target_m == (100.0, 50.0, 10.0)
    assert telem.contents.velocity_m_s == (4.0, 0, 0)
    assert ack.contents.confirms_packet_id == 7


def test_mislabeled_packet_is_refused():
    """A MOVE label with an ACK inside is a bug, so it fails the moment it's made."""
    with pytest.raises(ValueError, match="must contain MoveContents"):
        Packet(
            packet_id=8,
            source="ground",
            destination="mapper_1",
            kind=MOVE,
            sent_at_s=13.0,
            contents=AckContents(confirms_packet_id=7),
        )


def test_unknown_kind_is_refused():
    """A typo such as "MOVEE" would never match anything later, so refuse it now."""
    with pytest.raises(ValueError, match="unknown kind"):
        Packet(
            packet_id=9,
            source="ground",
            destination="mapper_1",
            kind="MOVEE",
            sent_at_s=13.0,
            contents=MoveContents(target_m=(0.0, 0.0, 10.0)),
        )


def test_packet_is_sealed():
    """Nobody can change a packet after it's made, like a sealed envelope."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        MOVE_7.destination = "relay_1"
