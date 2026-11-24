"""Trajectory evaluation: Umeyama alignment, ATE and RPE (Sturm et al. / Zhang & Scaramuzza)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lie import quat_slerp, quat_to_rot, rot_to_quat, so3_log


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = False):
    """Find (s, R, t) minimising sum ||dst_i - (s R src_i + t)||^2  (Umeyama 1991)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    n = len(src)
    cov = xd.T @ xs / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    if with_scale:
        var_s = (xs ** 2).sum() / n
        s = float(np.trace(np.diag(D) @ S) / var_s)
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


@dataclass
class AteResult:
    rmse: float
    mean: float
    median: float
    max: float
    errors: np.ndarray
    scale: float
    R: np.ndarray
    t: np.ndarray


def ate(est_p: np.ndarray, gt_p: np.ndarray, with_scale: bool = False) -> AteResult:
    """Absolute trajectory error (translation) after Umeyama alignment of est onto gt."""
    s, R, t = umeyama(est_p, gt_p, with_scale)
    aligned = (s * (R @ np.asarray(est_p).T)).T + t
    e = np.linalg.norm(aligned - gt_p, axis=1)
    return AteResult(float(np.sqrt(np.mean(e ** 2))), float(e.mean()), float(np.median(e)),
                     float(e.max()), e, s, R, t)


@dataclass
class RpeResult:
    trans_rmse: float      # m
    rot_rmse_deg: float    # deg
    n_pairs: int


def rpe(t: np.ndarray, est_R, est_p, gt_R, gt_p, delta: float = 1.0) -> RpeResult:
    """Relative pose error over a fixed time delta (seconds)."""
    t = np.asarray(t, float)
    et, er = [], []
    j = 0
    for i in range(len(t)):
        while j < len(t) and t[j] < t[i] + delta:
            j += 1
        if j >= len(t):
            break
        dE_R = est_R[i].T @ est_R[j]
        dE_p = est_R[i].T @ (est_p[j] - est_p[i])
        dG_R = gt_R[i].T @ gt_R[j]
        dG_p = gt_R[i].T @ (gt_p[j] - gt_p[i])
        E_R = dG_R.T @ dE_R
        E_p = dG_R.T @ (dE_p - dG_p)
        et.append(np.linalg.norm(E_p))
        er.append(np.degrees(np.linalg.norm(so3_log(E_R))))
    if not et:
        return RpeResult(float("nan"), float("nan"), 0)
    et, er = np.array(et), np.array(er)
    return RpeResult(float(np.sqrt(np.mean(et ** 2))), float(np.sqrt(np.mean(er ** 2))), len(et))


def associate(t_est: np.ndarray, t_gt: np.ndarray, max_diff: float = 0.02):
    """Nearest-timestamp association. Returns index arrays (i_est, i_gt)."""
    t_gt = np.asarray(t_gt)
    idx = np.searchsorted(t_gt, t_est)
    ie, ig = [], []
    for i, k in enumerate(idx):
        best = None
        for c in (k - 1, k):
            if 0 <= c < len(t_gt):
                d = abs(t_gt[c] - t_est[i])
                if d <= max_diff and (best is None or d < best[1]):
                    best = (c, d)
        if best is not None:
            ie.append(i)
            ig.append(best[0])
    return np.array(ie, int), np.array(ig, int)


def interpolate_poses(t_query, t, p, q):
    """Linear/slerp interpolation of a trajectory (q: Hamilton wxyz). Returns (p, R)."""
    t = np.asarray(t)
    out_p, out_R = [], []
    for tq in t_query:
        k = int(np.clip(np.searchsorted(t, tq) - 1, 0, len(t) - 2))
        s = float(np.clip((tq - t[k]) / (t[k + 1] - t[k]), 0.0, 1.0))
        out_p.append((1 - s) * p[k] + s * p[k + 1])
        out_R.append(quat_to_rot(quat_slerp(q[k], q[k + 1], s)))
    return np.array(out_p), np.array(out_R)


# -------------------------------------------------------------- TUM trajectory I/O

def save_tum(path, t, R_list, p_list) -> None:
    """Write ``timestamp tx ty tz qx qy qz qw`` (TUM format, Hamilton quaternion)."""
    with open(path, "w", encoding="utf-8") as f:
        for ti, R, p in zip(t, R_list, p_list):
            w, x, y, z = rot_to_quat(R)
            f.write(f"{ti:.9f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {x:.9f} {y:.9f} {z:.9f} {w:.9f}\n")


def load_tum(path):
    """Read a TUM trajectory. Returns (t, p (N,3), q_wxyz (N,4))."""
    data = np.loadtxt(path, comments="#", ndmin=2)
    t = data[:, 0]
    p = data[:, 1:4]
    q = np.column_stack([data[:, 7], data[:, 4], data[:, 5], data[:, 6]])
    return t, p, q
