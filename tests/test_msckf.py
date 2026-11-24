import numpy as np
import pytest

from insight_vio.imu import ImuNoise, Preintegrator
from insight_vio.lie import so3_exp, so3_log
from insight_vio.metrics import ate
from insight_vio.msckf import (CameraExtrinsics, FeatureTrack, Msckf, MsckfConfig,
                               propagation_jacobians)
from insight_vio.pipeline import dead_reckon, filter_config_for_sim, run_msckf_on_sim
from insight_vio.sim import R_BC_FORWARD, SimConfig, simulate

rng = np.random.default_rng(11)


@pytest.fixture(scope="module")
def sim20():
    return simulate(SimConfig(duration=20.0, seed=0))


def test_propagation_jacobian_matches_numeric():
    noise = ImuNoise()
    gyro = rng.normal(size=(20, 3)) * 0.5
    acc = rng.normal(size=(20, 3)) + np.array([0, 0, 9.81])
    dt = 0.005
    R0, p0, v0 = so3_exp(rng.normal(size=3)), rng.normal(size=3), rng.normal(size=3)
    bg0, ba0 = rng.normal(size=3) * 0.01, rng.normal(size=3) * 0.05

    def prop(R, p, v, bg, ba):
        pre = Preintegrator(bg, ba, noise)
        for w, a in zip(gyro, acc):
            pre.integrate(w, a, dt)
        return pre, pre.predict(R, p, v)

    pre, (R1, p1, v1) = prop(R0, p0, v0, bg0, ba0)
    F, Q = propagation_jacobians(pre, R0, noise)
    h = 1e-6
    for col in range(15):
        dx = np.zeros(15)
        dx[col] = h
        _, (R2, p2, v2) = prop(R0 @ so3_exp(dx[0:3]), p0 + dx[3:6], v0 + dx[6:9],
                               bg0 + dx[9:12], ba0 + dx[12:15])
        num = np.concatenate([so3_log(R1.T @ R2), p2 - p1, v2 - v1, dx[9:12], dx[12:15]]) / h
        assert np.allclose(num, F[:, col], atol=2e-4), (col, num, F[:, col])
    assert np.allclose(Q, Q.T) and np.all(np.linalg.eigvalsh(Q) > -1e-15)


def _filter_with_clones():
    cams = [CameraExtrinsics(R_BC_FORWARD, np.array([0.05, 0.03, 0.0])),
            CameraExtrinsics(R_BC_FORWARD, np.array([0.05, -0.03, 0.0]))]
    f = Msckf(MsckfConfig(cameras=cams))
    f.initialize(0.0, np.eye(3), np.zeros(3), np.zeros(3))
    p_f = np.array([3.0, 0.4, 0.2])
    for k in range(4):
        f.R = so3_exp(rng.normal(size=3) * 0.05)
        f.p = np.array([0.0, 0.1 * k, 0.02 * k])
        f.t = 0.05 * k
        cid = f._augment()
        obs = []
        for ci, ex in enumerate(cams):
            pc = ex.R_bc.T @ (f.R.T @ (p_f - f.p) - ex.p_bc)
            obs.append((ci, pc[:2] / pc[2]))
        f.features.setdefault(1, FeatureTrack())
        f.features[1].obs[cid] = obs
    return f, p_f


def test_measurement_jacobian_numeric_and_nullspace():
    f, p_f = _filter_with_clones()
    r, H_x, H_f, p_est = f.measurement_jacobian(1)
    assert np.allclose(p_est, p_f, atol=1e-8) and np.allclose(r, 0, atol=1e-9)
    # numeric Jacobian w.r.t. clone 2 rotation/position with the feature held fixed
    cid = list(f.clones)[2]
    idx = f._clone_index(cid)

    def predict(clones):
        out = []
        for c, obs in f.features[1].obs.items():
            _, R, p = clones[c]
            for cam, _ in obs:
                ex = f.cfg.cameras[cam]
                pc = ex.R_bc.T @ (R.T @ (p_f - p) - ex.p_bc)
                out.append(pc[:2] / pc[2])
        return np.concatenate(out)

    base = predict(f.clones)
    h = 1e-7
    for j in range(6):
        cl = dict(f.clones)
        t, R, p = cl[cid]
        d = np.zeros(6)
        d[j] = h
        cl[cid] = (t, R @ so3_exp(d[:3]), p + d[3:])
        num = (predict(cl) - base) / h
        assert np.allclose(num, H_x[:, idx + j], atol=1e-5)
    # null-space projection removes the feature
    r_o, H_o = f._feature_jacobian(1)
    Q, _ = np.linalg.qr(H_f, mode="complete")
    assert np.allclose(Q[:, 3:].T @ H_f, 0, atol=1e-10)
    assert H_o.shape[0] == H_x.shape[0] - 3


def test_stereo_msckf_ate_on_synthetic(sim20):
    res = run_msckf_on_sim(sim20, filter_config_for_sim(sim20, stereo=True), stereo=True)
    dr = ate(dead_reckon(sim20), sim20.gt_p[sim20.cam_idx])
    print(f"\n[stereo MSCKF 20 s] ATE rmse = {res.ate.rmse * 100:.2f} cm, max = "
          f"{res.ate.max * 100:.2f} cm, IMU-only ATE = {dr.rmse:.2f} m, "
          f"{res.seconds_per_frame * 1e3:.1f} ms/frame, stats={res.stats}")
    assert res.ate.rmse < 0.03          # 3 cm (measured ~0.3 cm on the dev machine)
    assert res.ate.rmse < 0.05 * dr.rmse
    # final orientation error
    dR = so3_log(res.gt_R[-1].T @ res.R[-1])
    assert np.degrees(np.linalg.norm(dR)) < 1.0


def test_mono_msckf_ate_on_synthetic(sim20):
    res = run_msckf_on_sim(sim20, filter_config_for_sim(sim20, stereo=False), stereo=False)
    print(f"\n[mono MSCKF 20 s] ATE rmse = {res.ate.rmse * 100:.2f} cm")
    assert res.ate.rmse < 0.05


def test_chi2_gating_rejects_injected_outliers(sim20):
    # corrupt 30 tracks per frame for 10 frames with large constant offsets (a "bad match")
    outliers = {}
    frames = sim20.frames
    for fi in range(50, 60):
        bad = {}
        for fid, obs in list(frames[fi].items())[:30]:
            bad[fid] = [(c, z + np.array([0.05, -0.04])) for c, z in obs]
        outliers[fi] = bad
    cfg = filter_config_for_sim(sim20, stereo=True)
    res = run_msckf_on_sim(sim20, cfg, stereo=True, outlier_frames=outliers)
    assert res.stats["features_rejected_chi2"] + res.stats["features_failed_triang"] > 30
    assert res.ate.rmse < 0.05


def test_time_offset_parameter_is_applied():
    data = simulate(SimConfig(duration=6.0, seed=2))
    td = 0.012
    shifted = type(data)(**{**data.__dict__, "cam_t": data.cam_t - td})
    cfg = filter_config_for_sim(data, stereo=True, time_offset=td)
    res = run_msckf_on_sim(shifted, cfg, stereo=True)
    assert np.allclose(res.t, data.cam_t)   # filter times are on the IMU clock
    assert res.ate.rmse < 0.03


def test_static_initialization_gravity_alignment():
    R_true = so3_exp(np.array([0.1, -0.2, 0.0]))
    acc = np.tile(R_true.T @ np.array([0, 0, 9.81]), (200, 1)) + rng.normal(0, 0.01, (200, 3))
    gyr = np.tile([0.002, -0.001, 0.003], (200, 1))
    f = Msckf(MsckfConfig())
    f.initialize_static(0.0, acc, gyr)
    # gravity direction matches (yaw is arbitrary)
    assert np.allclose(f.R @ (R_true.T @ np.array([0, 0, 1.0])), [0, 0, 1], atol=2e-3)
    assert np.allclose(f.bg, [0.002, -0.001, 0.003])
