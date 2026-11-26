#include <insight_vio/lie.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <random>

using insight_vio::jpl_to_hamilton;
using insight_vio::jpl_to_rot;
using insight_vio::quat_conj;
using insight_vio::quat_exp;
using insight_vio::quat_log;
using insight_vio::quat_mul;
using insight_vio::quat_slerp;
using insight_vio::quat_to_rot;
using insight_vio::rot_to_quat;
using insight_vio::se3_adjoint;
using insight_vio::se3_exp;
using insight_vio::se3_inv;
using insight_vio::se3_log;
using insight_vio::skew;
using insight_vio::so3_exp;
using insight_vio::so3_left_jacobian;
using insight_vio::so3_log;
using insight_vio::so3_right_jacobian;
using insight_vio::so3_right_jacobian_inv;
using insight_vio::vee;

namespace {

std::mt19937_64& rng() {
  static std::mt19937_64 gen(42);
  return gen;
}

Eigen::Vector3d rand_rotvec(double max_angle = 3.0) {
  std::normal_distribution<double> n(0.0, 1.0);
  std::uniform_real_distribution<double> u(0.0, max_angle);
  Eigen::Vector3d v(n(rng()), n(rng()), n(rng()));
  return v.normalized() * u(rng());
}

// Match numpy.allclose(a, b, atol=..., rtol=1e-5).
template <typename DerivedA, typename DerivedB>
bool allclose(const Eigen::MatrixBase<DerivedA>& a, const Eigen::MatrixBase<DerivedB>& b,
              double atol, double rtol = 1e-5) {
  return ((a - b).cwiseAbs().array() <= (atol + rtol * b.cwiseAbs().array())).all();
}

}  // namespace

TEST(Lie, SkewVeeAndCross) {
  std::normal_distribution<double> n(0.0, 1.0);
  Eigen::Vector3d a(n(rng()), n(rng()), n(rng()));
  Eigen::Vector3d b(n(rng()), n(rng()), n(rng()));
  EXPECT_TRUE((skew(a) * b).isApprox(a.cross(b), 1e-12));
  EXPECT_TRUE(vee(skew(a)).isApprox(a, 1e-12));
  EXPECT_TRUE(skew(a).transpose().isApprox(-skew(a), 1e-12));
}

class So3ExpLogRoundtrip : public ::testing::TestWithParam<double> {};

TEST_P(So3ExpLogRoundtrip, Roundtrip) {
  std::normal_distribution<double> n(0.0, 1.0);
  Eigen::Vector3d axis(n(rng()), n(rng()), n(rng()));
  axis.normalize();
  const double angle = GetParam();
  const Eigen::Vector3d phi = axis * angle;
  const Eigen::Matrix3d R = so3_exp(phi);
  EXPECT_TRUE((R.transpose() * R).isApprox(Eigen::Matrix3d::Identity(), 1e-12));
  EXPECT_NEAR(R.determinant(), 1.0, 1e-12);
  const Eigen::Vector3d phi2 = so3_log(R);
  EXPECT_TRUE(allclose(so3_exp(phi2), R, 1e-9));
  if (angle < 3.141592653589793 - 1e-3) {
    EXPECT_TRUE(allclose(phi2, phi, 1e-9));
  }
}

INSTANTIATE_TEST_SUITE_P(Angles, So3ExpLogRoundtrip,
                         ::testing::Values(0.0, 1e-9, 1e-5, 0.3, 2.0, 3.141592653589793 - 1e-6,
                                           3.141592653589793));

TEST(Lie, So3AdjointIdentity) {
  std::normal_distribution<double> n(0.0, 1.0);
  for (int i = 0; i < 20; ++i) {
    const Eigen::Matrix3d R = so3_exp(rand_rotvec());
    const Eigen::Vector3d phi = rand_rotvec(1.0);
    EXPECT_TRUE((R * so3_exp(phi) * R.transpose()).isApprox(so3_exp(R * phi), 1e-12));
    EXPECT_TRUE((R * skew(phi) * R.transpose()).isApprox(skew(R * phi), 1e-12));
  }
}

TEST(Lie, RightJacobianFirstOrder) {
  std::normal_distribution<double> n(0.0, 1.0);
  for (int i = 0; i < 20; ++i) {
    const Eigen::Vector3d phi = rand_rotvec(2.5);
    Eigen::Vector3d d(n(rng()), n(rng()), n(rng()));
    d *= 1e-6;
    const Eigen::Matrix3d lhs = so3_exp(phi + d);
    const Eigen::Matrix3d rhs = so3_exp(phi) * so3_exp(so3_right_jacobian(phi) * d);
    EXPECT_TRUE(lhs.isApprox(rhs, 1e-11));
    EXPECT_TRUE((so3_right_jacobian(phi) * so3_right_jacobian_inv(phi))
                    .isApprox(Eigen::Matrix3d::Identity(), 1e-10));
    const Eigen::Vector3d l2 = so3_log(so3_exp(phi) * so3_exp(d));
    EXPECT_TRUE(l2.isApprox(phi + so3_right_jacobian_inv(phi) * d, 1e-11));
    EXPECT_TRUE(
        so3_exp(phi + d).isApprox(so3_exp(so3_left_jacobian(phi) * d) * so3_exp(phi), 1e-11));
  }
}

TEST(Lie, Se3RoundtripAndAdjoint) {
  std::normal_distribution<double> n(0.0, 1.0);
  for (int i = 0; i < 20; ++i) {
    Eigen::Matrix<double, 6, 1> xi;
    xi.head<3>() = Eigen::Vector3d(n(rng()), n(rng()), n(rng()));
    xi.tail<3>() = rand_rotvec(2.5);
    const Eigen::Matrix4d T = se3_exp(xi);
    EXPECT_TRUE(se3_log(T).isApprox(xi, 1e-9));
    EXPECT_TRUE((se3_inv(T) * T).isApprox(Eigen::Matrix4d::Identity(), 1e-12));
    Eigen::Matrix<double, 6, 1> eta;
    for (int k = 0; k < 6; ++k) {
      eta[k] = n(rng()) * 0.3;
    }
    EXPECT_TRUE((T * se3_exp(eta) * se3_inv(T)).isApprox(se3_exp(se3_adjoint(T) * eta), 1e-9));
  }
}

TEST(Lie, QuaternionHamiltonConsistency) {
  std::normal_distribution<double> n(0.0, 1.0);
  for (int i = 0; i < 50; ++i) {
    const Eigen::Matrix3d Ra = so3_exp(rand_rotvec());
    const Eigen::Matrix3d Rb = so3_exp(rand_rotvec());
    const Eigen::Vector4d qa = rot_to_quat(Ra);
    const Eigen::Vector4d qb = rot_to_quat(Rb);
    EXPECT_TRUE(quat_to_rot(qa).isApprox(Ra, 1e-12));
    EXPECT_TRUE(quat_to_rot(quat_mul(qa, qb)).isApprox(Ra * Rb, 1e-12));
    EXPECT_TRUE(quat_to_rot(quat_conj(qa)).isApprox(Ra.transpose(), 1e-12));
    Eigen::Vector3d v(n(rng()), n(rng()), n(rng()));
    Eigen::Vector4d qv(0.0, v[0], v[1], v[2]);
    const Eigen::Vector3d rotated = quat_mul(quat_mul(qa, qv), quat_conj(qa)).tail<3>();
    EXPECT_TRUE(rotated.isApprox(Ra * v, 1e-12));
  }
  const Eigen::Vector3d phi = rand_rotvec(3.0);
  EXPECT_TRUE(quat_to_rot(quat_exp(phi)).isApprox(so3_exp(phi), 1e-12));
  EXPECT_TRUE(quat_log(quat_exp(phi)).isApprox(phi, 1e-9));
}

TEST(Lie, JplVsHamilton) {
  std::normal_distribution<double> n(0.0, 1.0);
  for (int i = 0; i < 20; ++i) {
    Eigen::Vector4d q(n(rng()), n(rng()), n(rng()), n(rng()));
    q.normalize();
    const Eigen::Matrix3d R_jpl = jpl_to_rot(q);
    const Eigen::Matrix3d R_ham_same =
        quat_to_rot(Eigen::Vector4d(q[3], q[0], q[1], q[2]));
    EXPECT_TRUE(R_jpl.isApprox(R_ham_same.transpose(), 1e-12));
    EXPECT_TRUE(quat_to_rot(jpl_to_hamilton(q)).isApprox(R_jpl, 1e-12));
  }
}

TEST(Lie, SlerpEndpointsAndMidpoint) {
  const Eigen::Vector4d q0 = rot_to_quat(Eigen::Matrix3d::Identity());
  const Eigen::Vector3d phi(0.0, 0.0, 1.0);
  const Eigen::Vector4d q1 = quat_exp(phi);
  EXPECT_TRUE(quat_slerp(q0, q1, 0.0).isApprox(q0, 1e-12));
  EXPECT_TRUE(quat_to_rot(quat_slerp(q0, q1, 1.0)).isApprox(quat_to_rot(q1), 1e-12));
  EXPECT_TRUE(quat_to_rot(quat_slerp(q0, q1, 0.5)).isApprox(so3_exp(phi / 2.0), 1e-12));
}
