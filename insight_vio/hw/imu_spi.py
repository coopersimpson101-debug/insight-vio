"""SPI IMU readers for BMI088 and ICM-42688-P (Raspberry Pi 5 / Jetson, Linux spidev).

**Status: written from the datasheets, NOT yet validated on hardware.**  Register
addresses/values were transcribed from the Bosch BMI088 datasheet (BST-BMI088-DS001)
and the TDK ICM-42688-P datasheet (DS-000347); check them against your revision.

Design:
* ``SpiBus`` is a tiny interface (``xfer(bytes) -> bytes``) so the decoding logic is
  unit-tested on any machine with :class:`MockSpiBus`.
* Each driver returns ``(gyro rad/s (3,), accel m/s^2 (3,))`` in the *sensor* frame.
  Rotate into the IMU/body frame with ``R_body_sensor`` from the config if the chip
  is mounted rotated.
* Timestamps: take ``time.monotonic_ns()`` in the data-ready interrupt handler (GPIO
  edge via libgpiod) - the same clock picamera2 uses for ``SensorTimestamp`` on Pi OS
  (verify on your kernel) - and feed both through :mod:`insight_vio.hw.timesync`.
"""
from __future__ import annotations

import struct
import time

import numpy as np

G = 9.80665
DEG = np.pi / 180.0


class SpiBus:
    def xfer(self, data: bytes) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError


class SpidevBus(SpiBus):  # pragma: no cover - needs Linux + hardware
    def __init__(self, bus: int, device: int, speed_hz: int = 8_000_000, mode: int = 0):
        import spidev
        self.dev = spidev.SpiDev()
        self.dev.open(bus, device)
        self.dev.max_speed_hz = speed_hz
        self.dev.mode = mode

    def xfer(self, data: bytes) -> bytes:
        return bytes(self.dev.xfer2(list(data)))


class MockSpiBus(SpiBus):
    """Register-file mock: ``regs[addr] = value``. Emulates read/write framing."""

    def __init__(self, regs: dict | None = None, dummy_byte: bool = False):
        self.regs = dict(regs or {})
        self.dummy = dummy_byte
        self.writes = []

    def xfer(self, data: bytes) -> bytes:
        addr = data[0] & 0x7F
        if data[0] & 0x80:  # read: auto-increment
            n = len(data) - 1
            out = [0]
            skip = 1 if self.dummy else 0
            for i in range(n):
                out.append(0 if i < skip else self.regs.get(addr + i - skip, 0))
            return bytes(out)
        for i, v in enumerate(data[1:]):
            self.regs[addr + i] = v
            self.writes.append((addr + i, v))
        return bytes(len(data))


# ------------------------------------------------------------------------- BMI088

class BMI088:
    """Bosch BMI088 (separate accel and gyro chip-selects)."""
    ACC_CHIP_ID, ACC_CHIP_ID_VAL = 0x00, 0x1E
    ACC_DATA = 0x12
    ACC_SENSORTIME = 0x18
    ACC_CONF, ACC_RANGE = 0x40, 0x41
    ACC_PWR_CONF, ACC_PWR_CTRL, ACC_SOFTRESET = 0x7C, 0x7D, 0x7E
    GYR_CHIP_ID, GYR_CHIP_ID_VAL = 0x00, 0x0F
    GYR_DATA = 0x02
    GYR_RANGE, GYR_BANDWIDTH, GYR_LPM1 = 0x0F, 0x10, 0x11
    GYR_INT_CTRL, GYR_INT34_MAP = 0x15, 0x18
    ACC_RANGES_G = {0x00: 3, 0x01: 6, 0x02: 12, 0x03: 24}
    GYR_RANGES_DPS = {0x00: 2000, 0x01: 1000, 0x02: 500, 0x03: 250, 0x04: 125}

    def __init__(self, acc_bus: SpiBus, gyr_bus: SpiBus, acc_range: int = 0x01,
                 gyr_range: int = 0x00, acc_odr: int = 0x0B, gyr_bw: int = 0x02):
        self.acc, self.gyr = acc_bus, gyr_bus
        self.acc_range, self.gyr_range = acc_range, gyr_range
        self.acc_odr, self.gyr_bw = acc_odr, gyr_bw

    # accel reads return one dummy byte before data
    def _acc_read(self, reg: int, n: int) -> bytes:
        return self.acc.xfer(bytes([reg | 0x80] + [0] * (n + 1)))[2:]

    def _acc_write(self, reg: int, val: int):
        self.acc.xfer(bytes([reg & 0x7F, val]))

    def _gyr_read(self, reg: int, n: int) -> bytes:
        return self.gyr.xfer(bytes([reg | 0x80] + [0] * n))[1:]

    def _gyr_write(self, reg: int, val: int):
        self.gyr.xfer(bytes([reg & 0x7F, val]))

    def configure(self, sleep=time.sleep) -> None:
        self._acc_read(self.ACC_CHIP_ID, 1)            # dummy read switches accel to SPI mode
        if self._acc_read(self.ACC_CHIP_ID, 1)[0] != self.ACC_CHIP_ID_VAL:
            raise IOError("BMI088 accel chip id mismatch")
        if self._gyr_read(self.GYR_CHIP_ID, 1)[0] != self.GYR_CHIP_ID_VAL:
            raise IOError("BMI088 gyro chip id mismatch")
        self._acc_write(self.ACC_PWR_CONF, 0x00)        # active mode
        sleep(0.005)
        self._acc_write(self.ACC_PWR_CTRL, 0x04)        # accel on
        sleep(0.05)
        self._acc_write(self.ACC_CONF, 0xA0 | self.acc_odr)   # normal BW, ODR (0x0B = 800 Hz)
        self._acc_write(self.ACC_RANGE, self.acc_range)
        self._gyr_write(self.GYR_RANGE, self.gyr_range)
        self._gyr_write(self.GYR_BANDWIDTH, self.gyr_bw)    # 0x02 = 1000 Hz ODR / 116 Hz BW
        self._gyr_write(self.GYR_LPM1, 0x00)                 # normal power
        self._gyr_write(self.GYR_INT_CTRL, 0x80)             # data-ready interrupt enable
        self._gyr_write(self.GYR_INT34_MAP, 0x01)            # map to INT3

    def accel_scale(self) -> float:
        """m/s^2 per LSB: raw/32768 * 1000 mg * 2^(range+1) * 1.5 (datasheet section 5.3.4)."""
        return (2 ** (self.acc_range + 1)) * 1.5 / 32768.0 * G

    def gyro_scale(self) -> float:
        return self.GYR_RANGES_DPS[self.gyr_range] / 32768.0 * DEG

    def read(self):
        a = struct.unpack("<hhh", self._acc_read(self.ACC_DATA, 6))
        w = struct.unpack("<hhh", self._gyr_read(self.GYR_DATA, 6))
        return np.array(w) * self.gyro_scale(), np.array(a) * self.accel_scale()

    def sensor_time_s(self) -> float:
        """24-bit accel sensortime counter, 39.0625 us per LSB (wraps every ~655 s)."""
        b = self._acc_read(self.ACC_SENSORTIME, 3)
        return (b[0] | (b[1] << 8) | (b[2] << 16)) * 39.0625e-6


# ------------------------------------------------------------------- ICM-42688-P

class ICM42688:
    WHO_AM_I, WHO_AM_I_VAL = 0x75, 0x47
    DEVICE_CONFIG = 0x11
    PWR_MGMT0 = 0x4E
    GYRO_CONFIG0, ACCEL_CONFIG0 = 0x4F, 0x50
    INT_CONFIG, INT_SOURCE0 = 0x14, 0x65
    ACCEL_DATA_X1 = 0x1F         # accel X1..Z0 then gyro X1..Z0 (big-endian)
    GYRO_FS_DPS = {0: 2000, 1: 1000, 2: 500, 3: 250, 4: 125, 5: 62.5, 6: 31.25, 7: 15.625}
    ACCEL_FS_G = {0: 16, 1: 8, 2: 4, 3: 2}

    def __init__(self, bus: SpiBus, gyro_fs: int = 0, accel_fs: int = 1, odr: int = 0x06):
        self.bus, self.gyro_fs, self.accel_fs, self.odr = bus, gyro_fs, accel_fs, odr

    def _read(self, reg, n):
        return self.bus.xfer(bytes([reg | 0x80] + [0] * n))[1:]

    def _write(self, reg, val):
        self.bus.xfer(bytes([reg & 0x7F, val]))

    def configure(self, sleep=time.sleep) -> None:
        self._write(self.DEVICE_CONFIG, 0x01)            # soft reset
        sleep(0.002)
        if self._read(self.WHO_AM_I, 1)[0] != self.WHO_AM_I_VAL:
            raise IOError("ICM-42688 WHO_AM_I mismatch")
        self._write(self.GYRO_CONFIG0, (self.gyro_fs << 5) | self.odr)   # odr 0x06 = 1 kHz
        self._write(self.ACCEL_CONFIG0, (self.accel_fs << 5) | self.odr)
        self._write(self.INT_CONFIG, 0x03)               # INT1 push-pull, active high
        self._write(self.INT_SOURCE0, 0x08)              # UI data ready -> INT1
        self._write(self.PWR_MGMT0, 0x0F)                # gyro + accel low-noise mode
        sleep(0.05)

    def gyro_scale(self) -> float:
        return self.GYRO_FS_DPS[self.gyro_fs] / 32768.0 * DEG

    def accel_scale(self) -> float:
        return self.ACCEL_FS_G[self.accel_fs] / 32768.0 * G

    def read(self):
        raw = struct.unpack(">hhhhhh", self._read(self.ACCEL_DATA_X1, 12))
        return np.array(raw[3:]) * self.gyro_scale(), np.array(raw[:3]) * self.accel_scale()


class ImuReader:  # pragma: no cover - needs hardware
    """Polling/IRQ loop that timestamps samples and pushes them to a callback.

    For best timing use the data-ready GPIO (``gpiod`` line request, edge events carry a
    kernel timestamp) instead of polling; ``timestamp_fn`` lets you plug that in.
    """

    def __init__(self, imu, rate_hz: float = 400.0, R_body_sensor=None,
                 timestamp_fn=time.monotonic_ns):
        self.imu = imu
        self.period = 1.0 / rate_hz
        self.R = np.eye(3) if R_body_sensor is None else np.asarray(R_body_sensor)
        self.ts = timestamp_fn
        self.running = False

    def run(self, callback):
        self.running = True
        next_t = time.perf_counter()
        while self.running:
            t_ns = self.ts()
            w, a = self.imu.read()
            callback(t_ns, self.R @ w, self.R @ a)
            next_t += self.period
            time.sleep(max(0.0, next_t - time.perf_counter()))
