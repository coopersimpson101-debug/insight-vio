#include <insight_vio/lie.hpp>
#include <insight_vio/preintegration.hpp>

#include "golden_io.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

using insight_vio::ImuNoise;
using insight_vio::Preintegrator;
using insight_vio::iterate_imu_segments;
using insight_vio::kGravity;
using insight_vio::so3_log;
using insight_vio::test::join_path;
using insight_vio::test::load_matrix;
using insight_vio::test::load_vector;

namespace {

std::string golden_dir() {
  if (const char* env = std::getenv("INSIGHT_VIO_GOLDENS")) {
    return env;
  }
  return INSIGHT_VIO_GOLDENS_DIR;
}

struct GoldenImu {
  Eigen::VectorXd imu_t;
  Eigen::VectorXd gt_t;
  Eigen::MatrixXd gyro;
  Eigen::MatrixXd accel;
  Eigen::MatrixXd gt_R_flat;
  Eigen::MatrixXd gt_p;
  Eigen::MatrixXd gt_v;
  Eigen::Vector3d gravity;

  Eigen::Matrix3d gt_R(int i) const {
    // Golden rows are row-major ravel(R); Eigen MatrixXd storage is col-major, so
    // do not Map through row(i).data() (non-contiguous).
    Eigen::Matrix3d R;
    R << gt_R_flat(i, 0), gt_R_flat(i, 1), gt_R_flat(i, 2), gt_R_flat(i, 3), gt_R_flat(i, 4),
        gt_R_flat(i, 5), gt_R_flat(i, 6), gt_R_flat(i, 7), gt_R_flat(i, 8);
    return R;
  }
};

GoldenImu load_imu() {
  const std::string dir = golden_dir();
  GoldenImu g;
  g.imu_t = load_vector(join_path(dir, "imu_t.txt"));
  g.gt_t = load_vector(join_path(dir, "gt_t.txt"));
  g.gyro = load_matrix(join_path(dir, "gyro.txt"));
  g.accel = load_matrix(join_path(dir, "accel.txt"));
  g.gt_R_flat = load_matrix(join_path(dir, "gt_R.txt"));
  g.gt_p = load_matrix(join_path(dir, "gt_p.txt"));
  g.gt_v = load_matrix(join_path(dir, "gt_v.txt"));
  g.gravity = load_vector(join_path(dir, "gravity.txt"));
  return g;
}

}  // namespace

TEST(Preintegration, MatchesGroundTruthAndPythonReference) {
  const GoldenImu d = load_imu();
  const int i = 40;
  const int j = 240;
  Preintegrator pre(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), ImuNoise{});
  for (int k = i; k < j; ++k) {
    pre.integrate(d.gyro.row(k).transpose(), d.accel.row(k).transpose(),
                  d.imu_t[k + 1] - d.imu_t[k]);
  }

  const Eigen::Matrix3d Ri = d.gt_R(i);
  const Eigen::Matrix3d Rj = d.gt_R(j);
  const double T = d.gt_t[j] - d.gt_t[i];
  const Eigen::Matrix3d dR_gt = Ri.transpose() * Rj;
  const Eigen::Vector3d dv_gt = Ri.transpose() * (d.gt_v.row(j).transpose() - d.gt_v.row(i).transpose() -
                                                  d.gravity * T);
  const Eigen::Vector3d dp_gt =
      Ri.transpose() * (d.gt_p.row(j).transpose() - d.gt_p.row(i).transpose() -
                        d.gt_v.row(i).transpose() * T - 0.5 * d.gravity * T * T);
  EXPECT_LT(so3_log(pre.dR.transpose() * dR_gt).norm(), 1e-9);
  EXPECT_LT((pre.dv - dv_gt).norm(), 1e-9);
  EXPECT_LT((pre.dp - dp_gt).norm(), 1e-6);

  auto [Rp, pp, vp] = pre.predict(Ri, d.gt_p.row(i).transpose(), d.gt_v.row(i).transpose(), d.gravity);
  (void)Rp;
  EXPECT_LT((pp - d.gt_p.row(j).transpose()).norm(), 1e-6);
  EXPECT_LT((vp - d.gt_v.row(j).transpose()).norm(), 1e-9);

  // Bit-stable match against the Python dump for the same window.
  const std::string dir = golden_dir();
  const Eigen::Matrix3d ref_dR = load_matrix(join_path(dir, "ref_dR.txt"));
  const Eigen::Vector3d ref_dv = load_vector(join_path(dir, "ref_dv.txt"));
  const Eigen::Vector3d ref_dp = load_vector(join_path(dir, "ref_dp.txt"));
  EXPECT_TRUE(pre.dR.isApprox(ref_dR, 1e-12));
  EXPECT_TRUE(pre.dv.isApprox(ref_dv, 1e-12));
  EXPECT_TRUE(pre.dp.isApprox(ref_dp, 1e-12));
  EXPECT_NEAR(pre.dt, load_vector(join_path(dir, "ref_meta.txt"))[2], 1e-15);
}

TEST(Preintegration, BiasJacobiansFirstOrderCorrection) {
  const GoldenImu d = load_imu();
  const int i = 0;
  const int j = 100;
  const Eigen::Vector3d bg0 = Eigen::Vector3d::Zero();
  const Eigen::Vector3d ba0 = Eigen::Vector3d::Zero();
  const Eigen::Vector3d dbg(0.004, -0.003, 0.002);
  const Eigen::Vector3d dba(0.03, 0.02, -0.04);
  Preintegrator base(bg0, ba0, ImuNoise{});
  Preintegrator ref(bg0 + dbg, ba0 + dba, ImuNoise{});
  for (int k = i; k < j; ++k) {
    const double dt = d.imu_t[k + 1] - d.imu_t[k];
    base.integrate(d.gyro.row(k).transpose(), d.accel.row(k).transpose(), dt);
    ref.integrate(d.gyro.row(k).transpose(), d.accel.row(k).transpose(), dt);
  }
  auto [dR, dv, dp] = base.corrected(bg0 + dbg, ba0 + dba);
  const double eff_R = so3_log(base.dR.transpose() * ref.dR).norm();
  const double eff_v = (base.dv - ref.dv).norm();
  const double eff_p = (base.dp - ref.dp).norm();
  EXPECT_LT(so3_log(dR.transpose() * ref.dR).norm(), 0.01 * eff_R);
  EXPECT_LT((dv - ref.dv).norm(), 0.02 * eff_v);
  EXPECT_LT((dp - ref.dp).norm(), 0.02 * eff_p);
}

TEST(Preintegration, BiasJacobiansVsNumeric) {
  const GoldenImu d = load_imu();
  auto run = [&](const Eigen::Vector3d& bg, const Eigen::Vector3d& ba) {
    Preintegrator p(bg, ba, ImuNoise{});
    for (int k = 0; k < 150; ++k) {
      p.integrate(d.gyro.row(k).transpose(), d.accel.row(k).transpose(), 1.0 / 200.0);
    }
    return p;
  };
  const Preintegrator base = run(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  const double h = 1e-6;
  for (int ax = 0; ax < 3; ++ax) {
    Eigen::Vector3d e = Eigen::Vector3d::Zero();
    e[ax] = h;
    const Preintegrator pg = run(e, Eigen::Vector3d::Zero());
    const Preintegrator pa = run(Eigen::Vector3d::Zero(), e);
    EXPECT_TRUE((so3_log(base.dR.transpose() * pg.dR) / h).isApprox(base.J_Rg.col(ax), 1e-4));
    EXPECT_TRUE(((pg.dv - base.dv) / h).isApprox(base.J_vg.col(ax), 1e-4));
    EXPECT_TRUE(((pg.dp - base.dp) / h).isApprox(base.J_pg.col(ax), 1e-4));
    EXPECT_TRUE(((pa.dv - base.dv) / h).isApprox(base.J_va.col(ax), 1e-4));
    EXPECT_TRUE(((pa.dp - base.dp) / h).isApprox(base.J_pa.col(ax), 1e-4));
  }
}

TEST(Preintegration, IterateSegmentsCoversInterval) {
  Eigen::VectorXd times = Eigen::VectorXd::LinSpaced(200, 0.0, 0.995);
  // Match numpy arange(0, 1.0, 0.005)
  times.resize(200);
  for (int i = 0; i < 200; ++i) {
    times[i] = i * 0.005;
  }
  const auto segs = iterate_imu_segments(times, 0.1012, 0.3507);
  double sum = 0.0;
  for (const auto& s : segs) {
    sum += s.second;
  }
  EXPECT_NEAR(sum, 0.3507 - 0.1012, 1e-12);
  ASSERT_FALSE(segs.empty());
  EXPECT_EQ(segs.front().first, 20);
}

TEST(Preintegration, GravityConstantMatchesPython) {
  EXPECT_TRUE(kGravity.isApprox(Eigen::Vector3d(0.0, 0.0, -9.81), 0.0));
}
