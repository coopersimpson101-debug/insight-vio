#include <insight_vio/lie.hpp>
#include <insight_vio/triangulation.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

using insight_vio::so3_exp;
using insight_vio::triangulate_gn;
using insight_vio::triangulate_linear;

namespace {

std::mt19937_64& rng() {
  static std::mt19937_64 gen(7);
  return gen;
}

void make_views(const Eigen::Vector3d& p_w, int n, double noise, double baseline,
                std::vector<Eigen::Matrix3d>& Rs, std::vector<Eigen::Vector3d>& ps,
                Eigen::MatrixX2d& zs) {
  std::normal_distribution<double> nrm(0.0, 1.0);
  Rs.clear();
  ps.clear();
  zs.resize(n, 2);
  for (int i = 0; i < n; ++i) {
    Eigen::Vector3d phi(nrm(rng()), nrm(rng()), nrm(rng()));
    phi *= 0.1;
    const Eigen::Matrix3d R = so3_exp(phi);
    const Eigen::Vector3d p(baseline * i / std::max(n - 1, 1), nrm(rng()) * 0.05,
                            nrm(rng()) * 0.05);
    const Eigen::Vector3d pc = R.transpose() * (p_w - p);
    zs(i, 0) = pc[0] / pc[2] + nrm(rng()) * noise;
    zs(i, 1) = pc[1] / pc[2] + nrm(rng()) * noise;
    Rs.push_back(R);
    ps.push_back(p);
  }
}

}  // namespace

TEST(Triangulation, Exact) {
  const Eigen::Vector3d p_w(0.4, -0.2, 4.0);
  std::vector<Eigen::Matrix3d> Rs;
  std::vector<Eigen::Vector3d> ps;
  Eigen::MatrixX2d zs;
  make_views(p_w, 5, 0.0, 0.3, Rs, ps, zs);
  EXPECT_TRUE(triangulate_linear(Rs, ps, zs).isApprox(p_w, 1e-9));
  const auto res = triangulate_gn(Rs, ps, zs);
  EXPECT_TRUE(res.ok);
  EXPECT_TRUE(res.p_w.isApprox(p_w, 1e-9));
}

TEST(Triangulation, NoisyAccuracy) {
  std::uniform_real_distribution<double> ux(-1.0, 1.0);
  std::uniform_real_distribution<double> uz(2.0, 5.0);
  std::vector<double> errs;
  errs.reserve(200);
  for (int trial = 0; trial < 200; ++trial) {
    const Eigen::Vector3d p_w(ux(rng()), ux(rng()), uz(rng()));
    std::vector<Eigen::Matrix3d> Rs;
    std::vector<Eigen::Vector3d> ps;
    Eigen::MatrixX2d zs;
    make_views(p_w, 6, 1.0 / 400.0, 0.4, Rs, ps, zs);
    const auto res = triangulate_gn(Rs, ps, zs);
    ASSERT_TRUE(res.ok);
    errs.push_back((res.p_w - p_w).norm() / p_w[2]);
  }
  std::sort(errs.begin(), errs.end());
  EXPECT_LT(errs[errs.size() / 2], 0.02);
  EXPECT_LT(errs[static_cast<size_t>(0.95 * (errs.size() - 1))], 0.06);
}

TEST(Triangulation, RejectsPointBehind) {
  std::vector<Eigen::Matrix3d> Rs;
  std::vector<Eigen::Vector3d> ps;
  Eigen::MatrixX2d zs_unused;
  make_views(Eigen::Vector3d(0.0, 0.0, 4.0), 5, 0.0, 0.3, Rs, ps, zs_unused);
  const Eigen::Vector3d behind(0.3, -0.2, -4.0);
  Eigen::MatrixX2d zs(static_cast<int>(Rs.size()), 2);
  for (size_t i = 0; i < Rs.size(); ++i) {
    const Eigen::Vector3d pc = Rs[i].transpose() * (behind - ps[i]);
    zs(static_cast<int>(i), 0) = pc[0] / pc[2];
    zs(static_cast<int>(i), 1) = pc[1] / pc[2];
  }
  EXPECT_FALSE(triangulate_gn(Rs, ps, zs).ok);
}
