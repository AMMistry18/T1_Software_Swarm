"""Packet formats from the team's spec ("Wireless Mesh Protocol & Packet
Specification", v2): how each message is written as bytes for the radio.

Every message is one UDP datagram: a 20-byte header, then a payload whose layout
depends on the message type. pack() turns numbers into those bytes and unpack()
reads them back. The simulator (transport.py) takes its packet sizes from here,
so its sizes always match the real formats.

The struct module does the conversion, following a format string: "!" means
big-endian (the byte order the spec uses), then one letter per field: B / b =
1-byte whole number (unsigned / signed), H = 2-byte unsigned, I = 4-byte unsigned,
Q = 8-byte unsigned, f = 4-byte decimal (float32), 2s = 2 raw bytes, x = 1 byte of
padding (always 0).
"""

import struct

MAGIC = b"DM"  # every datagram starts with these 2 bytes; anything else is dropped
# magic, msg_type, sender_id, seq_num, timestamp_ns, payload_len, reserved
HEADER = struct.Struct("!2sBBIQHH")

# Message types (the header's msg_type).
HEALTH = 0x01
LIDAR_CHUNK = 0x02
COMMAND = 0x03  # C2: command and control (its ACK is a COMMAND sent back)
POSE_CORRECTION = 0x04
RELAY_REPOSITION = 0x05
DEGENERACY = 0x06
DOCKING = 0x07

# Each type's payload, field by field, in the spec's order.
PAYLOADS = {
    # pos_x, pos_y, pos_z (m), battery_mv, rssi_dbm, noise_dbm, link_bitrate
    # (tenths of a Mb/s: 150 = 15.0 Mb/s), flight_mode, cpu_temp_c, dropped_pkts
    HEALTH: struct.Struct("!fffHbbHBbI"),
    # frame_id, chunk_idx, total_chunks; the chunk's data bytes follow
    LIDAR_CHUNK: struct.Struct("!IHH"),
    # target_node_id, command_id, param_1, param_2, cmd_seq_ack
    COMMAND: struct.Struct("!BBHfI"),
    # frame_id, corrected x, y, z (m), orientation quaternion w, x, y, z
    POSE_CORRECTION: struct.Struct("!Ifffffff"),
    # target x, y, z (m), target_yaw (radians)
    RELAY_REPOSITION: struct.Struct("!ffff"),
    # is_degenerate, min_eigenvalue, flow_vx, flow_vy (m/s), tof_distance (m)
    DEGENERACY: struct.Struct("!Bffff3x"),
    # dock_stage, charge_status, cell_1_mv, cell_2_mv, tof_dock_dx, tof_dock_dy (m)
    DOCKING: struct.Struct("!BBHHff2x"),
}

MAX_DATAGRAM_BYTES = 1200  # the spec's cap, so no packet gets split on the way
CHUNK_HEADER_BYTES = HEADER.size + PAYLOADS[LIDAR_CHUNK].size  # 28
MAX_CHUNK_DATA_BYTES = MAX_DATAGRAM_BYTES - CHUNK_HEADER_BYTES  # 1172


def datagram_bytes(msg_type):
    """The size, in bytes, of a whole datagram of a fixed-size type (every type
    but LiDAR chunks): the 20-byte header plus the payload."""
    return HEADER.size + PAYLOADS[msg_type].size


def pack(msg_type, sender_id, seq_num, timestamp_ns, *fields, data=b""):
    """One datagram as bytes.

    sender_id: 0x00 = base station, 0x01 = mapping drone, 0x02 and up = network
    drones. seq_num: the sender's own packet counter. timestamp_ns: when it was
    sent, in nanoseconds. fields: the payload's values in the order PAYLOADS
    lists them (*fields collects any number of values). data: bytes that follow
    the fields; only LiDAR chunks have them.
    """
    payload = PAYLOADS[msg_type].pack(*fields) + data
    header = HEADER.pack(
        MAGIC, msg_type, sender_id, seq_num, timestamp_ns, len(payload), 0
    )
    return header + payload


def unpack(datagram):
    """Read a datagram back into its parts.

    Returns (msg_type, sender_id, seq_num, timestamp_ns, fields, data), where
    fields is a tuple in the order PAYLOADS lists them. Raises ValueError for a
    datagram that isn't one of ours or got cut short.
    """
    if len(datagram) < HEADER.size:
        raise ValueError("shorter than the 20-byte header")
    magic, msg_type, sender_id, seq_num, timestamp_ns, payload_len, _ = (
        HEADER.unpack_from(datagram)
    )
    if magic != MAGIC:
        raise ValueError("not one of our packets (wrong magic bytes)")
    if msg_type not in PAYLOADS:
        raise ValueError(f"unknown message type {msg_type}")
    payload = datagram[HEADER.size :]
    layout = PAYLOADS[msg_type]
    if len(payload) != payload_len or len(payload) < layout.size:
        raise ValueError("the payload's length doesn't match the header")
    fields = layout.unpack_from(payload)
    return msg_type, sender_id, seq_num, timestamp_ns, fields, payload[layout.size :]


def pack_keyframe(sender_id, first_seq_num, timestamp_ns, frame_id, keyframe):
    """Cut one compressed LiDAR keyframe (bytes) into chunk datagrams of at most
    MAX_DATAGRAM_BYTES, numbered 0, 1, 2, ... The base can rebuild the keyframe
    only once every one of them has arrived."""
    pieces = []
    for start in range(0, len(keyframe), MAX_CHUNK_DATA_BYTES):
        pieces.append(keyframe[start : start + MAX_CHUNK_DATA_BYTES])
    return [
        pack(
            LIDAR_CHUNK,
            sender_id,
            first_seq_num + index,
            timestamp_ns,
            frame_id,
            index,
            len(pieces),
            data=piece,
        )
        for index, piece in enumerate(pieces)
    ]
