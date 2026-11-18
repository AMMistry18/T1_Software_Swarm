"""Tests for network/messages.py: the spec's byte formats for every message type."""

import pytest

from tar_swarm.network import messages
from tar_swarm.network.transport import keyframe_chunk_sizes


def test_sizes_match_the_spec():
    """Every whole datagram has the size the spec's table gives it."""
    sizes = {
        messages.HEALTH: 44,
        messages.COMMAND: 32,
        messages.POSE_CORRECTION: 52,
        messages.RELAY_REPOSITION: 36,
        messages.DEGENERACY: 40,
        messages.DOCKING: 36,
    }
    for msg_type, size in sizes.items():
        assert messages.datagram_bytes(msg_type) == size
    assert messages.CHUNK_HEADER_BYTES == 28


def test_pack_then_unpack_gives_back_the_values():
    """A Health packet (the spec's own example) survives the trip to bytes and
    back. Decimals come back as float32, which is close but not exact: -4.2
    becomes -4.199999809."""
    fields = (12.5, -4.2, 3.8, 12400, -68, -95, 185, 1, 42, 0)
    datagram = messages.pack(messages.HEALTH, 0x01, 7, 123_456_789, *fields)
    assert len(datagram) == 44
    assert datagram[:4] == b"DM\x01\x01"  # magic, msg_type 0x01, sender 0x01
    msg_type, sender, seq_num, timestamp_ns, back, data = messages.unpack(datagram)
    assert (msg_type, sender, seq_num, timestamp_ns) == (1, 1, 7, 123_456_789)
    assert back == pytest.approx(fields)
    assert data == b""


def test_a_keyframe_becomes_chunks_that_fit_and_rebuild():
    """A 10,000-byte keyframe becomes 9 datagrams of at most 1200 bytes, the same
    sizes the simulator uses, and their data joins back into the keyframe."""
    keyframe = bytes(range(256)) * 39 + bytes(16)  # 10,000 bytes
    datagrams = messages.pack_keyframe(0x01, 100, 0, 42, keyframe)
    assert [len(d) for d in datagrams] == keyframe_chunk_sizes(10_000)
    rebuilt = b""
    for index, datagram in enumerate(datagrams):
        msg_type, _, seq_num, _, fields, data = messages.unpack(datagram)
        assert (msg_type, seq_num) == (messages.LIDAR_CHUNK, 100 + index)
        assert fields == (42, index, 9)  # frame_id, chunk_idx, total_chunks
        rebuilt += data
    assert rebuilt == keyframe


def test_damaged_packets_are_refused():
    """Wrong magic bytes, a cut-off datagram, and an unknown type all fail
    loudly instead of being misread."""
    good = messages.pack(messages.RELAY_REPOSITION, 0x00, 1, 0, 6.2, -2.1, 5.0, 1.57)
    with pytest.raises(ValueError, match="magic"):
        messages.unpack(b"XX" + good[2:])
    with pytest.raises(ValueError, match="length"):
        messages.unpack(good[:-1])
    with pytest.raises(ValueError, match="unknown"):
        messages.unpack(good[:2] + b"\x09" + good[3:])
