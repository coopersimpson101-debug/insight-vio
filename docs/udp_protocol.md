# UDP pose protocol v1

One datagram per filter update (camera rate, e.g. 30 Hz), 84 bytes, little-endian.
Implementation: `insight_vio/udp.py`; reference consumer: `examples/udp_receiver.py`.

| offset | size | type | field |
|---:|---:|---|---|
| 0 | 4 | `char[4]` | magic `"IVIO"` |
| 4 | 2 | `u16` | version = 1 |
| 6 | 2 | `u16` | flags (below) |
| 8 | 4 | `u32` | sequence number (wraps) |
| 12 | 8 | `u64` | timestamp, ns, IMU/host monotonic clock, of the pose |
| 20 | 12 | `f32[3]` | position, m |
| 32 | 16 | `f32[4]` | orientation, Hamilton quaternion `w, x, y, z` (body → world) |
| 48 | 12 | `f32[3]` | linear velocity, world frame, m/s |
| 60 | 12 | `f32[3]` | angular velocity, body frame, rad/s (bias-corrected gyro) |
| 72 | 4 | `f32` | position σ, m (sqrt of largest eigenvalue of the 3x3 covariance) |
| 76 | 4 | `f32` | rotation σ, rad |
| 80 | 4 | `u32` | CRC-32 (zlib/IEEE, same as `zlib.crc32`) over bytes 0..79 |

Flags: bit0 tracking valid, bit1 OpenXR axes, bit2 initialising, bit3 degraded
(processing slower than budget / few features).

## Axes

* Default (bit1 = 0): VIO world frame, **Z up** (gravity −Z), X = initial forward
  heading; body frame X forward, Y left, Z up.
* `openxr_axes: true` (bit1 = 1): both world and head frames re-expressed in OpenXR
  convention (**Y up, −Z forward, X right**) via the fixed rotation
  `M = [[0,−1,0],[0,0,1],[−1,0,0]]`: `R_xr = M R Mᵀ`, `p_xr = M p`.

## Latency / prediction

The runtime should predict the pose to display time:
`q(t + Δ) = q ⊗ Exp(ω Δ)`, `p(t + Δ) = p + v Δ`, where `Δ = t_display − t_ns`.
Rendering at 90 Hz from a 30 Hz VIO also benefits from an IMU-rate propagator in the
runtime (send raw IMU on a second port; roadmap).

## Example (Python)

```python
from insight_vio.udp import unpack
pkt = unpack(datagram)          # raises ValueError on bad size/CRC/magic
```

C struct:

```c
#pragma pack(push, 1)
typedef struct {
  char magic[4]; uint16_t version, flags; uint32_t seq; uint64_t t_ns;
  float p[3], q_wxyz[4], v[3], w[3], pos_std, rot_std; uint32_t crc32;
} ivio_pose_v1;   /* sizeof == 84 */
#pragma pack(pop)
```
