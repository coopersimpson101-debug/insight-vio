import numpy as np

from insight_vio.imu import GRAVITY, ImuNoise, Preintegrator, iterate_imu_segments
from insight_vio.lie import so3_exp, so3_log
from insight_vio.sim import SimConfig, simulate


def noiseless_sim(duration=3.0, rate=200.0):
    n = ImuNoise(0.0, 0.0, 0.0, 0.0)
    cfg = SimConfig(duration=duration, imu_rate=rate, noise=n, gyro_bias0_std=0.0,
                    accel_bias0_std=0.0, n_wall_landmarks=10, n_pillar_landmarks=0)
    return simulate(cfg)


def test_preintegration_matches_ground_truth():
    d = noiseless_sim()
    i, j = 40, 240                      # 1 s window at 200 Hz
    pre = Preintegrator(np.zeros(3), np.zeros(3), ImuNoise())
    for k in range(i, j):
        pre.integrate(d.gyro[k], d.accel[k], d.imu_t[k + 1] - d.imu_t[k])
    Ri, Rj = d.gt_R[i], d.gt_R[j]
    T = d.gt_t[j] - d.gt_t[i]
    dR_gt = Ri.T @ Rj
    dv_gt = Ri.T @ (d.gt_v[j] - d.gt_v[i] - GRAVITY * T)
    dp_gt = Ri.T @ (d.gt_p[j] - d.gt_p[i] - d.gt_v[i] * T - 0.5 * GRAVITY * T * T)
    assert np.linalg.norm(so3_log(pre.dR.T @ dR_gt)) < 1e-9
    assert np.linalg.norm(pre.dv - dv_gt) < 1e-9
    assert np.linalg.norm(pre.dp - dp_gt) < 1e-6
    Rp, pp, vp = pre.predict(Ri, d.gt_p[i], d.gt_v[i])
    assert np.linalg.norm(pp - d.gt_p[j]) < 1e-6
    assert np.linalg.norm(vp - d.gt_v[j]) < 1e-9


def test_bias_jacobians_first_order_correction():
    d = noiseless_sim()
    i, j = 0, 100
    bg0, ba0 = np.zeros(3), np.zeros(3)
    dbg = np.array([0.004, -0.003, 0.002])
    dba = np.array([0.03, 0.02, -0.04])
    base = Preintegrator(bg0, ba0, ImuNoise())
    ref = Preintegrator(bg0 + dbg, ba0 + dba, ImuNoise())
    for k in range(i, j):
        dt = d.imu_t[k + 1] - d.imu_t[k]
        base.integrate(d.gyro[k], d.accel[k], dt)
        ref.integrate(d.gyro[k], d.accel[k], dt)
    dR, dv, dp = base.corrected(bg0 + dbg, ba0 + dba)
    # first-order correction error must be much smaller than the effect of the bias change
    eff_R = np.linalg.norm(so3_log(base.dR.T @ ref.dR))
    eff_v = np.linalg.norm(base.dv - ref.dv)
    eff_p = np.linalg.norm(base.dp - ref.dp)
    assert np.linalg.norm(so3_log(dR.T @ ref.dR)) < 0.01 * eff_R
    assert np.linalg.norm(dv - ref.dv) < 0.02 * eff_v
    assert np.linalg.norm(dp - ref.dp) < 0.02 * eff_p


def test_bias_jacobians_vs_numeric():
    d = noiseless_sim(duration=1.0)
    def run(bg, ba):
        p = Preintegrator(bg, ba, ImuNoise())
        for k in range(150):
            p.integrate(d.gyro[k], d.accel[k], 1 / 200)
        return p
    base = run(np.zeros(3), np.zeros(3))
    h = 1e-6
    for ax in range(3):
        e = np.zeros(3)
        e[ax] = h
        pg = run(e, np.zeros(3))
        pa = run(np.zeros(3), e)
        assert np.allclose(so3_log(base.dR.T @ pg.dR) / h, base.J_Rg[:, ax], atol=1e-4)
        assert np.allclose((pg.dv - base.dv) / h, base.J_vg[:, ax], atol=1e-4)
        assert np.allclose((pg.dp - base.dp) / h, base.J_pg[:, ax], atol=1e-4)
        assert np.allclose((pa.dv - base.dv) / h, base.J_va[:, ax], atol=1e-4)
        assert np.allclose((pa.dp - base.dp) / h, base.J_pa[:, ax], atol=1e-4)


def test_preintegration_covariance_matches_monte_carlo():
    """Propagated 9x9 covariance agrees with sample covariance over noisy realisations."""
    rng = np.random.default_rng(3)
    noise = ImuNoise(gyro_noise=2e-3, accel_noise=2e-2, gyro_walk=0, accel_walk=0)
    d = noiseless_sim(duration=1.0)
    N, K, dt = 400, 60, 1 / 200
    errs = []
    clean = Preintegrator(np.zeros(3), np.zeros(3), noise)
    for k in range(K):
        clean.integrate(d.gyro[k], d.accel[k], dt)
    for _ in range(N):
        p = Preintegrator(np.zeros(3), np.zeros(3), noise)
        for k in range(K):
            p.integrate(d.gyro[k] + rng.normal(0, noise.gyro_noise / np.sqrt(dt), 3),
                        d.accel[k] + rng.normal(0, noise.accel_noise / np.sqrt(dt), 3), dt)
        errs.append(np.concatenate([so3_log(clean.dR.T @ p.dR), p.dv - clean.dv, p.dp - clean.dp]))
    S = np.cov(np.array(errs).T)
    for blk in (slice(0, 3), slice(3, 6), slice(6, 9)):
        ratio = np.trace(S[blk, blk]) / np.trace(clean.cov[blk, blk])
        assert 0.8 < ratio < 1.25, ratio


def test_iterate_segments_covers_interval():
    times = np.arange(0, 1.0, 0.005)
    segs = list(iterate_imu_segments(times, 0.1012, 0.3507))
    assert np.isclose(sum(dt for _, dt in segs), 0.3507 - 0.1012)
    assert segs[0][0] == 20
