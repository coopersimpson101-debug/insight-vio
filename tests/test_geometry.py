import numpy as np
import pytest

from insight_vio.camera import Camera
from insight_vio.lie import so3_exp
from insight_vio.metrics import ate, associate, rpe, umeyama
from insight_vio.stats import chi2_quantile, normal_quantile
from insight_vio.triangulation import triangulate_gn, triangulate_linear

rng = np.random.default_rng(7)


@pytest.mark.parametrize("model,dist", [
    ("radtan", [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]),   # EuRoC cam0
    ("equidistant", [0.0034, 0.0122, -0.0101, 0.0026]),
])
def test_distortion_roundtrip(model, dist):
    cam = Camera(752, 480, 458.654, 457.296, 367.215, 248.375, model, dist)
    uv = np.column_stack([rng.uniform(20, 730, 200), rng.uniform(20, 460, 200)])
    xn = cam.pixel_to_normalized(uv)
    uv2 = cam.normalized_to_pixel(xn)
    assert np.max(np.abs(uv2 - uv)) < 1e-6


def test_project_validity():
    cam = Camera(640, 480, 400, 400, 319.5, 239.5)
    uv, ok = cam.project(np.array([[0, 0, 2.0], [0, 0, -1.0], [10, 0, 1.0]]))
    assert ok.tolist() == [True, False, False]
    assert np.allclose(uv[0], [319.5, 239.5])


def make_views(p_w, n=5, noise=0.0, baseline=0.3):
    Rs, ps, zs = [], [], []
    for i in range(n):
        R = so3_exp(rng.normal(size=3) * 0.1)
        p = np.array([baseline * i / max(n - 1, 1), rng.normal() * 0.05, rng.normal() * 0.05])
        pc = R.T @ (p_w - p)
        z = pc[:2] / pc[2] + rng.normal(0, noise, 2)
        Rs.append(R)
        ps.append(p)
        zs.append(z)
    return Rs, ps, np.array(zs)


def test_triangulation_exact():
    p_w = np.array([0.4, -0.2, 4.0])
    Rs, ps, zs = make_views(p_w)
    assert np.allclose(triangulate_linear(Rs, ps, zs), p_w, atol=1e-9)
    res = triangulate_gn(Rs, ps, zs)
    assert res.ok and np.allclose(res.p_w, p_w, atol=1e-9)


def test_triangulation_noisy_accuracy():
    errs = []
    for _ in range(200):
        p_w = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(2, 5)])
        Rs, ps, zs = make_views(p_w, n=6, noise=1.0 / 400.0, baseline=0.4)
        res = triangulate_gn(Rs, ps, zs)
        assert res.ok
        errs.append(np.linalg.norm(res.p_w - p_w) / p_w[2])
    # relative depth error with 1 px noise, 0.4 m baseline, 2-5 m depth
    assert np.median(errs) < 0.02
    assert np.percentile(errs, 95) < 0.06


def test_triangulation_rejects_point_behind():
    Rs, ps, _ = make_views(np.array([0.0, 0.0, 4.0]))
    behind = np.array([0.3, -0.2, -4.0])             # negative depth in every view
    zs = []
    for R, p in zip(Rs, ps):
        pc = R.T @ (behind - p)
        zs.append(pc[:2] / pc[2])
    assert not triangulate_gn(Rs, ps, np.array(zs)).ok


def test_chi2_quantiles():
    assert abs(normal_quantile(0.975) - 1.959964) < 1e-5
    assert abs(chi2_quantile(0.95, 1) - 3.841459) < 1e-4
    assert abs(chi2_quantile(0.95, 10) - 18.307038) < 1e-4
    assert abs(chi2_quantile(0.99, 50) - 76.153891) < 1e-3


def test_umeyama_recovers_transform():
    X = rng.normal(size=(100, 3))
    R = so3_exp(np.array([0.3, -0.5, 1.2]))
    t = np.array([1.0, -2.0, 0.5])
    Y = (2.0 * (R @ X.T)).T + t
    s, R2, t2 = umeyama(X, Y, with_scale=True)
    assert np.isclose(s, 2.0) and np.allclose(R2, R) and np.allclose(t2, t)
    res = ate((R.T @ (Y - t).T).T, Y)   # rotated/translated copy -> zero ATE
    assert res.rmse < 1e-9


def test_rpe_zero_for_rigidly_transformed_trajectory():
    t = np.arange(0, 5, 0.1)
    gt_R = np.array([so3_exp(np.array([0, 0, 0.3 * ti])) for ti in t])
    gt_p = np.column_stack([np.cos(t), np.sin(t), 0.1 * t])
    G = so3_exp(np.array([0.1, 0.2, 0.3]))
    est_R = np.array([G @ R for R in gt_R])
    est_p = (G @ gt_p.T).T + 1.0
    r = rpe(t, est_R, est_p, gt_R, gt_p, delta=1.0)
    assert r.n_pairs > 0 and r.trans_rmse < 1e-9 and r.rot_rmse_deg < 1e-6


def test_associate():
    ie, ig = associate(np.array([0.0, 0.5, 1.001, 3.0]), np.arange(0, 2.0, 0.1), 0.01)
    assert ie.tolist() == [0, 1, 2] and ig.tolist() == [0, 5, 10]
