"""Stereo feature front-end: FAST/Shi-Tomasi detection + pyramidal KLT tracking.

Per frame:

1. Track the previous left-image features into the new left image with KLT,
   optionally seeded by the gyro-predicted rotation (``R_c1_c0``), with a
   forward-backward consistency check.
2. Detect new FAST corners in image regions without tracks (grid bucketed).
3. Match every left feature into the right image with KLT (seeded at zero
   disparity) and reject matches violating the epipolar constraint derived from
   the calibrated stereo extrinsics.
4. Emit ``{feature_id: [(cam, xy_normalized), ...]}`` for the MSCKF.

Images are numpy arrays (uint8 or float).  OpenCV is *not* used here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import Camera
from .features import detect_features, suppression_mask
from .klt import build_pyramid, track, track_bidirectional
from .lie import skew


@dataclass
class FrontendConfig:
    max_features: int = 150
    fast_threshold: float = 15.0
    min_distance: int = 15
    pyramid_levels: int = 4
    win: int = 15
    fb_threshold: float = 0.7
    epipolar_threshold_px: float = 2.0


class StereoFrontend:
    def __init__(self, cameras: list[Camera], config: FrontendConfig | None = None):
        self.cams = cameras
        self.cfg = config or FrontendConfig()
        self.prev_pyr = None
        self.pts = np.zeros((0, 2))
        self.ids = np.zeros(0, dtype=np.int64)
        self.next_id = 0
        if len(cameras) > 1:
            cl, cr = cameras[0], cameras[1]
            R_rl = cr.R_bc.T @ cl.R_bc
            t_rl = cr.R_bc.T @ (cl.p_bc - cr.p_bc)
            self.E = skew(t_rl) @ R_rl
        else:
            self.E = None

    def _predict(self, pts: np.ndarray, R_c1_c0: np.ndarray | None) -> np.ndarray | None:
        if R_c1_c0 is None or len(pts) == 0:
            return None
        cam = self.cams[0]
        xn = cam.pixel_to_normalized(pts)
        h = np.column_stack([xn, np.ones(len(xn))]) @ R_c1_c0.T
        ok = h[:, 2] > 1e-3
        h[~ok, 2] = 1.0
        pred = cam.normalized_to_pixel(h[:, :2] / h[:, 2:3])
        pred[~ok] = pts[~ok]
        return pred

    def process(self, left: np.ndarray, right: np.ndarray | None = None,
                R_c1_c0: np.ndarray | None = None) -> dict:
        cfg = self.cfg
        pyr = build_pyramid(left, cfg.pyramid_levels)
        # 1. temporal tracking
        if self.prev_pyr is not None and len(self.pts):
            guess = self._predict(self.pts, R_c1_c0)
            new, ok, _ = track_bidirectional(self.prev_pyr, pyr, self.pts, guess,
                                             fb_threshold=cfg.fb_threshold, win=cfg.win)
            self.pts, self.ids = new[ok], self.ids[ok]
        # 2. detection in empty regions
        need = cfg.max_features - len(self.pts)
        if need > 0:
            allowed = suppression_mask(left.shape, self.pts, cfg.min_distance)
            new_pts, _ = detect_features(left, need, cfg.fast_threshold, allowed=allowed)
            if len(new_pts):
                self.pts = np.vstack([self.pts, new_pts])
                self.ids = np.concatenate([self.ids, np.arange(self.next_id,
                                                               self.next_id + len(new_pts))])
                self.next_id += len(new_pts)
        self.prev_pyr = pyr

        xn_l = self.cams[0].pixel_to_normalized(self.pts) if len(self.pts) else np.zeros((0, 2))
        out = {int(i): [(0, xn_l[k])] for k, i in enumerate(self.ids)}
        # 3. stereo matching
        if right is not None and self.E is not None and len(self.pts):
            pyr_r = build_pyramid(right, cfg.pyramid_levels)
            pr, ok, _ = track(pyr, pyr_r, self.pts, win=cfg.win)
            xn_r = self.cams[1].pixel_to_normalized(pr)
            line = np.column_stack([xn_l, np.ones(len(xn_l))]) @ self.E.T   # E x_l
            num = np.abs(np.sum(np.column_stack([xn_r, np.ones(len(xn_r))]) * line, axis=1))
            dist = num / np.maximum(np.linalg.norm(line[:, :2], axis=1), 1e-12)
            ok &= dist * self.cams[1].mean_focal < cfg.epipolar_threshold_px
            for k in np.nonzero(ok)[0]:
                out[int(self.ids[k])].append((1, xn_r[k]))
        return out
