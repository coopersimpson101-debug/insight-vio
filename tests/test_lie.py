import numpy as np
import pytest

from insight_vio.lie import (jpl_to_hamilton, jpl_to_rot, quat_conj, quat_exp, quat_log, quat_mul,
                             quat_slerp, quat_to_rot, rot_to_quat, se3_adjoint, se3_exp, se3_inv,
                             se3_log, skew, so3_exp, so3_left_jacobian, so3_log,
                             so3_right_jacobian, so3_right_jacobian_inv, vee)

rng = np.random.default_rng(42)


def rand_rotvec(max_angle=3.0):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v) * rng.uniform(0, max_angle)


def test_skew_vee_and_cross():
    a, b = rng.normal(size=3), rng.normal(size=3)
    assert np.allclose(skew(a) @ b, np.cross(a, b))
    assert np.allclose(vee(skew(a)), a)
    assert np.allclose(skew(a).T, -skew(a))


@pytest.mark.parametrize("angle", [0.0, 1e-9, 1e-5, 0.3, 2.0, np.pi - 1e-6, np.pi])
def test_so3_exp_log_roundtrip(angle):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    phi = axis * angle
    R = so3_exp(phi)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)
    phi2 = so3_log(R)
    assert np.allclose(so3_exp(phi2), R, atol=1e-9)
    if angle < np.pi - 1e-3:
        assert np.allclose(phi2, phi, atol=1e-9)


def test_so3_adjoint_identity():
    for _ in range(20):
        R = so3_exp(rand_rotvec())
        phi = rand_rotvec(1.0)
        assert np.allclose(R @ so3_exp(phi) @ R.T, so3_exp(R @ phi), atol=1e-12)
        assert np.allclose(R @ skew(phi) @ R.T, skew(R @ phi), atol=1e-12)


def test_right_jacobian_first_order():
    for _ in range(20):
        phi = rand_rotvec(2.5)
        d = rng.normal(size=3) * 1e-6
        lhs = so3_exp(phi + d)
        rhs = so3_exp(phi) @ so3_exp(so3_right_jacobian(phi) @ d)
        assert np.allclose(lhs, rhs, atol=1e-11)
        # inverse
        assert np.allclose(so3_right_jacobian(phi) @ so3_right_jacobian_inv(phi), np.eye(3),
                           atol=1e-10)
        # log perturbation
        l2 = so3_log(so3_exp(phi) @ so3_exp(d))
        assert np.allclose(l2, phi + so3_right_jacobian_inv(phi) @ d, atol=1e-11)
        # left jacobian
        assert np.allclose(so3_exp(phi + d), so3_exp(so3_left_jacobian(phi) @ d) @ so3_exp(phi),
                           atol=1e-11)


def test_se3_roundtrip_and_adjoint():
    for _ in range(20):
        xi = np.concatenate([rng.normal(size=3), rand_rotvec(2.5)])
        T = se3_exp(xi)
        assert np.allclose(se3_log(T), xi, atol=1e-9)
        assert np.allclose(se3_inv(T) @ T, np.eye(4), atol=1e-12)
        eta = rng.normal(size=6) * 0.3
        assert np.allclose(T @ se3_exp(eta) @ se3_inv(T), se3_exp(se3_adjoint(T) @ eta), atol=1e-9)


def test_quaternion_hamilton_consistency():
    for _ in range(50):
        Ra, Rb = so3_exp(rand_rotvec()), so3_exp(rand_rotvec())
        qa, qb = rot_to_quat(Ra), rot_to_quat(Rb)
        assert np.allclose(quat_to_rot(qa), Ra, atol=1e-12)
        assert np.allclose(quat_to_rot(quat_mul(qa, qb)), Ra @ Rb, atol=1e-12)
        assert np.allclose(quat_to_rot(quat_conj(qa)), Ra.T, atol=1e-12)
        v = rng.normal(size=3)
        qv = np.concatenate([[0.0], v])
        rotated = quat_mul(quat_mul(qa, qv), quat_conj(qa))[1:]
        assert np.allclose(rotated, Ra @ v, atol=1e-12)
    phi = rand_rotvec(3.0)
    assert np.allclose(quat_to_rot(quat_exp(phi)), so3_exp(phi), atol=1e-12)
    assert np.allclose(quat_log(quat_exp(phi)), phi, atol=1e-9)


def test_jpl_vs_hamilton():
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        # same four numbers, read as Hamilton (w first), give the transposed matrix
        R_jpl = jpl_to_rot(q)                        # q = [x, y, z, w]
        R_ham_same_numbers = quat_to_rot(np.array([q[3], q[0], q[1], q[2]]))
        assert np.allclose(R_jpl, R_ham_same_numbers.T, atol=1e-12)
        assert np.allclose(quat_to_rot(jpl_to_hamilton(q)), R_jpl, atol=1e-12)


def test_slerp_endpoints_and_midpoint():
    q0 = rot_to_quat(np.eye(3))
    phi = np.array([0.0, 0.0, 1.0])
    q1 = quat_exp(phi)
    assert np.allclose(quat_slerp(q0, q1, 0.0), q0)
    assert np.allclose(quat_to_rot(quat_slerp(q0, q1, 1.0)), quat_to_rot(q1))
    assert np.allclose(quat_to_rot(quat_slerp(q0, q1, 0.5)), so3_exp(phi / 2), atol=1e-12)
