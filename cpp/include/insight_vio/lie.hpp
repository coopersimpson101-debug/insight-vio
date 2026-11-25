#pragma once
// Lie-group utilities for SO(3) and SE(3), plus Hamilton quaternions.
// Conventions match insight_vio/lie.py (right perturbation, Hamilton w-first).

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>

namespace insight_vio {

inline constexpr double kLieSmall = 1e-8;

inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d m;
  m << 0.0, -v.z(), v.y(),
       v.z(), 0.0, -v.x(),
       -v.y(), v.x(), 0.0;
  return m;
}

inline Eigen::Vector3d vee(const Eigen::Matrix3d& m) {
  return Eigen::Vector3d(m(2, 1), m(0, 2), m(1, 0));
}

inline Eigen::Matrix3d so3_exp(const Eigen::Vector3d& phi) {
  const double theta2 = phi.squaredNorm();
  const Eigen::Matrix3d K = skew(phi);
  if (theta2 < kLieSmall) {
    return Eigen::Matrix3d::Identity() + K + 0.5 * K * K;
  }
  const double theta = std::sqrt(theta2);
  return Eigen::Matrix3d::Identity() + (std::sin(theta) / theta) * K +
         ((1.0 - std::cos(theta)) / theta2) * (K * K);
}

inline Eigen::Vector3d so3_log(const Eigen::Matrix3d& R) {
  double cos_t = (R.trace() - 1.0) * 0.5;
  cos_t = std::clamp(cos_t, -1.0, 1.0);
  const double theta = std::acos(cos_t);
  if (theta < 1e-6) {
    return 0.5 * vee(R - R.transpose()) * (1.0 + theta * theta / 6.0);
  }
  constexpr double kPi = 3.14159265358979323846;
  if (kPi - theta < 1e-4) {
    // Near pi: uu^T ≈ (R + I)/2. Disambiguate axis sign when [R-R^T] is tiny.
    const Eigen::Matrix3d B = 0.5 * (R + Eigen::Matrix3d::Identity());
    Eigen::Index k = 0;
    B.diagonal().maxCoeff(&k);
    Eigen::Vector3d u = B.col(static_cast<int>(k)) / std::sqrt(std::max(B(k, k), 1e-12));
    u.normalize();
    const Eigen::Vector3d w = vee(R - R.transpose());
    if (w.dot(u) < 0.0) {
      u = -u;
    }
    Eigen::Vector3d phi = theta * u;
    if ((so3_exp(phi) - R).squaredNorm() > (so3_exp(-phi) - R).squaredNorm()) {
      phi = -phi;
    }
    return phi;
  }
  return (theta / (2.0 * std::sin(theta))) * vee(R - R.transpose());
}

inline Eigen::Matrix3d so3_right_jacobian(const Eigen::Vector3d& phi) {
  const double theta2 = phi.squaredNorm();
  const Eigen::Matrix3d K = skew(phi);
  if (theta2 < kLieSmall) {
    return Eigen::Matrix3d::Identity() - 0.5 * K + K * K / 6.0;
  }
  const double theta = std::sqrt(theta2);
  return Eigen::Matrix3d::Identity() - ((1.0 - std::cos(theta)) / theta2) * K +
         ((theta - std::sin(theta)) / (theta2 * theta)) * (K * K);
}

inline Eigen::Matrix3d so3_right_jacobian_inv(const Eigen::Vector3d& phi) {
  const double theta2 = phi.squaredNorm();
  const Eigen::Matrix3d K = skew(phi);
  if (theta2 < kLieSmall) {
    return Eigen::Matrix3d::Identity() + 0.5 * K + K * K / 12.0;
  }
  const double theta = std::sqrt(theta2);
  return Eigen::Matrix3d::Identity() + 0.5 * K +
         (1.0 / theta2 - (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta))) *
             (K * K);
}

inline Eigen::Matrix3d so3_left_jacobian(const Eigen::Vector3d& phi) {
  return so3_right_jacobian(-phi);
}

inline Eigen::Matrix3d orthonormalize(const Eigen::Matrix3d& R) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  const Eigen::Matrix3d Vt = svd.matrixV().transpose();
  Eigen::Matrix3d M = U * Vt;
  if (M.determinant() < 0.0) {
    U.col(2) *= -1.0;
    M = U * Vt;
  }
  return M;
}

inline Eigen::Matrix4d se3_exp(const Eigen::Matrix<double, 6, 1>& xi) {
  const Eigen::Vector3d rho = xi.head<3>();
  const Eigen::Vector3d phi = xi.tail<3>();
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = so3_exp(phi);
  T.block<3, 1>(0, 3) = so3_left_jacobian(phi) * rho;
  return T;
}

inline Eigen::Matrix<double, 6, 1> se3_log(const Eigen::Matrix4d& T) {
  const Eigen::Vector3d phi = so3_log(T.block<3, 3>(0, 0));
  const Eigen::Matrix3d Jl = so3_left_jacobian(phi);
  const Eigen::Vector3d rho = Jl.fullPivLu().solve(T.block<3, 1>(0, 3));
  Eigen::Matrix<double, 6, 1> xi;
  xi << rho, phi;
  return xi;
}

inline Eigen::Matrix4d se3_inv(const Eigen::Matrix4d& T) {
  const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  const Eigen::Vector3d t = T.block<3, 1>(0, 3);
  Eigen::Matrix4d Ti = Eigen::Matrix4d::Identity();
  Ti.block<3, 3>(0, 0) = R.transpose();
  Ti.block<3, 1>(0, 3) = -R.transpose() * t;
  return Ti;
}

inline Eigen::Matrix4d se3_from_rt(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = R;
  T.block<3, 1>(0, 3) = t;
  return T;
}

inline Eigen::Matrix<double, 6, 6> se3_adjoint(const Eigen::Matrix4d& T) {
  const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  const Eigen::Vector3d t = T.block<3, 1>(0, 3);
  Eigen::Matrix<double, 6, 6> Ad = Eigen::Matrix<double, 6, 6>::Zero();
  Ad.block<3, 3>(0, 0) = R;
  Ad.block<3, 3>(0, 3) = skew(t) * R;
  Ad.block<3, 3>(3, 3) = R;
  return Ad;
}

inline Eigen::Vector4d quat_normalize(const Eigen::Vector4d& q_in) {
  Eigen::Vector4d q = q_in.normalized();
  if (q[0] < 0.0) {
    q = -q;
  }
  return q;
}

inline Eigen::Vector4d quat_mul(const Eigen::Vector4d& a, const Eigen::Vector4d& b) {
  const double aw = a[0], ax = a[1], ay = a[2], az = a[3];
  const double bw = b[0], bx = b[1], by = b[2], bz = b[3];
  return Eigen::Vector4d(
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw);
}

inline Eigen::Vector4d quat_conj(const Eigen::Vector4d& q) {
  return Eigen::Vector4d(q[0], -q[1], -q[2], -q[3]);
}

inline Eigen::Matrix3d quat_to_rot(const Eigen::Vector4d& q_in) {
  const Eigen::Vector4d q = quat_normalize(q_in);
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  Eigen::Matrix3d R;
  R << 1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
       2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
       2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y);
  return R;
}

inline Eigen::Vector4d rot_to_quat(const Eigen::Matrix3d& R) {
  const double tr = R.trace();
  Eigen::Vector4d q;
  if (tr > 0.0) {
    const double s = 2.0 * std::sqrt(tr + 1.0);
    q = Eigen::Vector4d(0.25 * s, (R(2, 1) - R(1, 2)) / s, (R(0, 2) - R(2, 0)) / s,
                        (R(1, 0) - R(0, 1)) / s);
  } else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
    const double s = 2.0 * std::sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2));
    q = Eigen::Vector4d((R(2, 1) - R(1, 2)) / s, 0.25 * s, (R(0, 1) + R(1, 0)) / s,
                        (R(0, 2) + R(2, 0)) / s);
  } else if (R(1, 1) > R(2, 2)) {
    const double s = 2.0 * std::sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2));
    q = Eigen::Vector4d((R(0, 2) - R(2, 0)) / s, (R(0, 1) + R(1, 0)) / s, 0.25 * s,
                        (R(1, 2) + R(2, 1)) / s);
  } else {
    const double s = 2.0 * std::sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1));
    q = Eigen::Vector4d((R(1, 0) - R(0, 1)) / s, (R(0, 2) + R(2, 0)) / s,
                        (R(1, 2) + R(2, 1)) / s, 0.25 * s);
  }
  return quat_normalize(q);
}

inline Eigen::Vector4d quat_exp(const Eigen::Vector3d& phi) {
  const double theta = phi.norm();
  if (theta < 1e-12) {
    Eigen::Vector4d q(1.0, 0.5 * phi[0], 0.5 * phi[1], 0.5 * phi[2]);
    return q.normalized();
  }
  const Eigen::Vector3d axis = phi / theta;
  Eigen::Vector4d q;
  q << std::cos(0.5 * theta), std::sin(0.5 * theta) * axis;
  return q;
}

inline Eigen::Vector3d quat_log(const Eigen::Vector4d& q_in) {
  const Eigen::Vector4d q = quat_normalize(q_in);
  const Eigen::Vector3d v = q.tail<3>();
  const double n = v.norm();
  if (n < 1e-12) {
    return 2.0 * v;
  }
  return 2.0 * std::atan2(n, q[0]) * v / n;
}

inline Eigen::Vector4d jpl_to_hamilton(const Eigen::Vector4d& q_jpl_xyzw) {
  return quat_normalize(
      Eigen::Vector4d(q_jpl_xyzw[3], -q_jpl_xyzw[0], -q_jpl_xyzw[1], -q_jpl_xyzw[2]));
}

inline Eigen::Matrix3d jpl_to_rot(const Eigen::Vector4d& q_jpl_xyzw) {
  const Eigen::Vector4d qn = q_jpl_xyzw.normalized();
  const double x = qn[0], y = qn[1], z = qn[2], w = qn[3];
  const Eigen::Vector3d qv(x, y, z);
  return (2.0 * w * w - 1.0) * Eigen::Matrix3d::Identity() - 2.0 * w * skew(qv) +
         2.0 * qv * qv.transpose();
}

inline Eigen::Vector4d quat_slerp(const Eigen::Vector4d& q0_in, const Eigen::Vector4d& q1_in,
                                  double s) {
  Eigen::Vector4d q0 = quat_normalize(q0_in);
  Eigen::Vector4d q1 = q1_in.normalized();
  if (q0.dot(q1) < 0.0) {
    q1 = -q1;
  }
  const Eigen::Vector4d dq = quat_mul(quat_conj(q0), q1);
  return quat_normalize(quat_mul(q0, quat_exp(s * quat_log(dq))));
}

}  // namespace insight_vio
