"""Pose output over UDP: a fixed 84-byte little-endian packet (see docs/udp_protocol.md).

====== ====== ======== ==================================================
offset size   type     field
====== ====== ======== ==================================================
0      4      char[4]  magic ``b"IVIO"``
4      2      u16      version (=1)
6      2      u16      flags: bit0 tracking valid, bit1 OpenXR axes,
                       bit2 initialising, bit3 degraded (few features)
8      4      u32      sequence number
12     8      u64      timestamp, ns (IMU clock, CLOCK_MONOTONIC on Linux)
20     12     f32[3]   position (m)
32     16     f32[4]   orientation, Hamilton quaternion w, x, y, z
48     12     f32[3]   linear velocity, world frame (m/s)
60     12     f32[3]   angular velocity, body frame (rad/s)
72     4      f32      position std-dev (m, sqrt of max eigenvalue)
76     4      f32      rotation std-dev (rad)
80     4      u32      CRC-32 (zlib) of bytes 0..79
====== ====== ======== ==================================================
"""
from __future__ import annotations

import socket
import struct
import zlib
from dataclasses import dataclass

import numpy as np

from .lie import quat_to_rot, rot_to_quat

MAGIC = b"IVIO"
VERSION = 1
_FMT = "<4sHHIQ3f4f3f3fff"
_BODY = struct.calcsize(_FMT)
PACKET_SIZE = _BODY + 4
assert PACKET_SIZE == 84

FLAG_VALID = 1 << 0
FLAG_OPENXR = 1 << 1
FLAG_INITIALIZING = 1 << 2
FLAG_DEGRADED = 1 << 3

# VIO world/body: x forward, y left, z up.  OpenXR: x right, y up, z backward.
M_VIO_TO_OPENXR = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])


@dataclass
class PosePacket:
    seq: int
    t_ns: int
    position: np.ndarray
    quat_wxyz: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray
    pos_std: float = 0.0
    rot_std: float = 0.0
    flags: int = FLAG_VALID


def pack(p: PosePacket) -> bytes:
    body = struct.pack(_FMT, MAGIC, VERSION, p.flags & 0xFFFF, p.seq & 0xFFFFFFFF,
                       int(p.t_ns) & 0xFFFFFFFFFFFFFFFF, *map(float, p.position),
                       *map(float, p.quat_wxyz), *map(float, p.velocity),
                       *map(float, p.angular_velocity), float(p.pos_std), float(p.rot_std))
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def unpack(buf: bytes) -> PosePacket:
    if len(buf) != PACKET_SIZE:
        raise ValueError(f"bad packet size {len(buf)}")
    body, (crc,) = buf[:_BODY], struct.unpack("<I", buf[_BODY:])
    if zlib.crc32(body) & 0xFFFFFFFF != crc:
        raise ValueError("CRC mismatch")
    v = struct.unpack(_FMT, body)
    if v[0] != MAGIC or v[1] != VERSION:
        raise ValueError("bad magic/version")
    return PosePacket(seq=v[3], t_ns=v[4], position=np.array(v[5:8]), quat_wxyz=np.array(v[8:12]),
                      velocity=np.array(v[12:15]), angular_velocity=np.array(v[15:18]),
                      pos_std=v[18], rot_std=v[19], flags=v[2])


def to_openxr(R_wb: np.ndarray, p_wb: np.ndarray, v_w=None, w_b=None):
    """Re-express a VIO pose in OpenXR axes (Y up, -Z forward, right-handed)."""
    M = M_VIO_TO_OPENXR
    R = M @ R_wb @ M.T
    out = [R, M @ p_wb]
    out.append(None if v_w is None else M @ v_w)
    out.append(None if w_b is None else M @ w_b)
    return tuple(out)


class PoseSender:
    """Fire-and-forget UDP pose publisher."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, openxr_axes: bool = False):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0
        self.openxr = openxr_axes

    def send(self, t_ns: int, R_wb, p_wb, v_w, w_b, P_pose: np.ndarray | None = None,
             flags: int = FLAG_VALID) -> bytes:
        if self.openxr:
            R_wb, p_wb, v_w, w_b = to_openxr(R_wb, p_wb, v_w, w_b)
            flags |= FLAG_OPENXR
        pos_std = rot_std = 0.0
        if P_pose is not None:
            rot_std = float(np.sqrt(max(np.linalg.eigvalsh(P_pose[0:3, 0:3]).max(), 0)))
            pos_std = float(np.sqrt(max(np.linalg.eigvalsh(P_pose[3:6, 3:6]).max(), 0)))
        pkt = pack(PosePacket(self.seq, t_ns, p_wb, rot_to_quat(R_wb), v_w, w_b,
                              pos_std, rot_std, flags))
        self.sock.sendto(pkt, self.addr)
        self.seq += 1
        return pkt

    def close(self):
        self.sock.close()


def receive_forever(port: int = 5555, host: str = "0.0.0.0", callback=print):
    """Blocking receiver loop (used by examples/udp_receiver.py and the web viewer)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((host, port))
    while True:
        buf, _ = s.recvfrom(256)
        try:
            callback(unpack(buf))
        except ValueError:
            continue


def packet_rotation(p: PosePacket) -> np.ndarray:
    return quat_to_rot(p.quat_wxyz)
