"""Lie-group utilities for SO(3) and SE(3), plus Hamilton quaternions.

Conventions (used everywhere in insight-vio):

* Rotation matrices ``R_ab`` map vectors expressed in frame *b* into frame *a*:
  ``v_a = R_ab @ v_b``.
* Quaternions are **Hamilton**, stored scalar-first ``[w, x, y, z]``, with
  ``q_a ⊗ q_b`` corresponding to ``R(q_a) @ R(q_b)``. ``quat_to_rot(q)`` returns
  the active rotation (body-to-world when ``q = q_wb``).  Eigen, ROS, Ceres,
  OpenXR and the EuRoC ground truth all use Hamilton.  The MSCKF papers
  (Mourikis & Roumeliotis 2007, Trawny & Roumeliotis 2005) use **JPL**
  quaternions; :func:`jpl_to_hamilton` documents and implements the mapping.
* Perturbations are on the **right** (body/local frame):
  ``R = R_hat @ Exp(dtheta)``.  This matches Forster et al. 2017.
"""
from __future__ import annotations

import numpy as np

_SMALL = 1e-8


def skew(v: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix ``[v]x`` so that ``[v]x @ u = v x u``."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def vee(m: np.ndarray) -> np.ndarray:
    """Inverse of :func:`skew`."""
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Exponential map so(3) -> SO(3) (Rodrigues formula)."""
    phi = np.asarray(phi, dtype=float)
    theta2 = float(phi @ phi)
    K = skew(phi)
    if theta2 < _SMALL:
        # second-order Taylor expansion, re-orthonormalised implicitly by small size
        return np.eye(3) + K + 0.5 * K @ K
    theta = np.sqrt(theta2)
    return np.eye(3) + (np.sin(theta) / theta) * K + ((1.0 - np.cos(theta)) / theta2) * K @ K


def so3_log(R: np.ndarray) -> np.ndarray:
    """Logarithm map SO(3) -> so(3) returning the rotation vector (robust near 0 and pi)."""
    R = np.asarray(R, dtype=float)
    cos_t = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-6:
        return 0.5 * vee(R - R.T) * (1.0 + theta * theta / 6.0)
    if np.pi - theta < 1e-4:
        # Near pi: use the symmetric part. R = I + 2 [u]x^2 approximately -> uu^T = (R + I)/2
        B = 0.5 * (R + np.eye(3))
        k = int(np.argmax(np.diag(B)))
        u = B[:, k] / np.sqrt(max(B[k, k], 1e-12))
        u /= np.linalg.norm(u)
        # pick the sign consistent with the antisymmetric part
        w = vee(R - R.T)
        if w @ u < 0:
            u = -u
        return theta * u
    return (theta / (2.0 * np.sin(theta))) * vee(R - R.T)


def so3_right_jacobian(phi: np.ndarray) -> np.ndarray:
    """Right Jacobian J_r(phi):  Exp(phi + d) ~= Exp(phi) Exp(J_r(phi) d)."""
    phi = np.asarray(phi, dtype=float)
    theta2 = float(phi @ phi)
    K = skew(phi)
    if theta2 < _SMALL:
        return np.eye(3) - 0.5 * K + K @ K / 6.0
    theta = np.sqrt(theta2)
    return (np.eye(3) - ((1.0 - np.cos(theta)) / theta2) * K
            + ((theta - np.sin(theta)) / (theta2 * theta)) * K @ K)


def so3_right_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    """Inverse of the right Jacobian:  Log(Exp(phi) Exp(d)) ~= phi + J_r^{-1}(phi) d."""
    phi = np.asarray(phi, dtype=float)
    theta2 = float(phi @ phi)
    K = skew(phi)
    if theta2 < _SMALL:
        return np.eye(3) + 0.5 * K + K @ K / 12.0
    theta = np.sqrt(theta2)
    return (np.eye(3) + 0.5 * K
            + (1.0 / theta2 - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))) * K @ K)


def so3_left_jacobian(phi: np.ndarray) -> np.ndarray:
    """Left Jacobian J_l(phi) = J_r(-phi)."""
    return so3_right_jacobian(-np.asarray(phi, dtype=float))


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Project a nearly-orthonormal matrix back onto SO(3) via SVD."""
    U, _, Vt = np.linalg.svd(R)
    M = U @ Vt
    if np.linalg.det(M) < 0:
        U[:, -1] *= -1
        M = U @ Vt
    return M


# --------------------------------------------------------------------------- SE(3)

def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exponential map se(3) -> SE(3).  ``xi = [rho (translation part), phi (rotation)]``."""
    xi = np.asarray(xi, dtype=float)
    rho, phi = xi[:3], xi[3:]
    T = np.eye(4)
    T[:3, :3] = so3_exp(phi)
    T[:3, 3] = so3_left_jacobian(phi) @ rho
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """Logarithm map SE(3) -> se(3), returning ``[rho, phi]``."""
    phi = so3_log(T[:3, :3])
    Jl = so3_left_jacobian(phi)
    rho = np.linalg.solve(Jl, T[:3, 3])
    return np.concatenate([rho, phi])


def se3_inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def se3_from_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def se3_adjoint(T: np.ndarray) -> np.ndarray:
    """6x6 adjoint for the ``[rho, phi]`` ordering: Exp(Ad_T xi) = T Exp(xi) T^-1."""
    R, t = T[:3, :3], T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[:3, 3:] = skew(t) @ R
    Ad[3:, 3:] = R
    return Ad


# ---------------------------------------------------------------- Hamilton quaternions

def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a ⊗ b`` for ``[w, x, y, z]`` quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Hamilton quaternion ``[w,x,y,z]`` -> rotation matrix."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> Hamilton quaternion ``[w,x,y,z]`` with w >= 0 (Shepperd's method)."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s,
                      (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s,
                      (R[1, 2] + R[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return quat_normalize(q)


def quat_exp(phi: np.ndarray) -> np.ndarray:
    """Rotation vector -> unit Hamilton quaternion."""
    phi = np.asarray(phi, dtype=float)
    theta = np.linalg.norm(phi)
    if theta < 1e-12:
        q = np.array([1.0, 0.5 * phi[0], 0.5 * phi[1], 0.5 * phi[2]])
        return q / np.linalg.norm(q)
    axis = phi / theta
    return np.concatenate([[np.cos(0.5 * theta)], np.sin(0.5 * theta) * axis])


def quat_log(q: np.ndarray) -> np.ndarray:
    """Unit Hamilton quaternion -> rotation vector (shortest path)."""
    q = quat_normalize(q)
    v = q[1:]
    n = np.linalg.norm(v)
    if n < 1e-12:
        return 2.0 * v
    return 2.0 * np.arctan2(n, q[0]) * v / n


def jpl_to_hamilton(q_jpl_xyzw: np.ndarray) -> np.ndarray:
    """Convert a JPL quaternion (Trawny & Roumeliotis, MSCKF papers) to Hamilton.

    A JPL quaternion is stored vector-first ``[x, y, z, w]`` and its rotation
    matrix is ``C(q) = R_ham(q)^T`` (it represents the *passive*/global-to-local
    rotation).  JPL multiplication satisfies ``q1 ⊗_jpl q2 = q2 ⊗_ham q1``.  So the
    JPL quaternion of ``C_{L<-G}`` written ``^L_G q`` has the same four numbers as
    the Hamilton quaternion of ``R_{G<-L}``.  This function returns the Hamilton
    quaternion ``[w, x, y, z]`` describing the *same rotation matrix* as the JPL one,
    i.e. ``quat_to_rot(result) == jpl_to_rot(q_jpl)``.
    """
    x, y, z, w = q_jpl_xyzw
    return quat_normalize(np.array([w, -x, -y, -z]))


def jpl_to_rot(q_jpl_xyzw: np.ndarray) -> np.ndarray:
    """Rotation matrix of a JPL quaternion (Trawny eq. 78): C = (2w^2-1)I - 2w[q]x + 2 q q^T."""
    x, y, z, w = np.asarray(q_jpl_xyzw, dtype=float) / np.linalg.norm(q_jpl_xyzw)
    qv = np.array([x, y, z])
    return (2 * w * w - 1) * np.eye(3) - 2 * w * skew(qv) + 2 * np.outer(qv, qv)


def quat_slerp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    q0 = quat_normalize(q0)
    q1 = np.asarray(q1, dtype=float) / np.linalg.norm(q1)
    if q0 @ q1 < 0:
        q1 = -q1
    dq = quat_mul(quat_conj(q0), q1)
    return quat_normalize(quat_mul(q0, quat_exp(s * quat_log(dq))))
