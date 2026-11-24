import numpy as np
import pytest

from insight_vio.camera import Camera
from insight_vio.features import detect_features, fast_mask, nms3x3, shi_tomasi_score
from insight_vio.frontend import FrontendConfig, StereoFrontend
from insight_vio.klt import bilinear, build_pyramid, track, track_bidirectional
from insight_vio.sim import R_BC_FORWARD, fourier_shift, render_pattern_image


def test_fast_detects_square_corners_only():
    img = np.zeros((60, 60), np.float32)
    img[20:40, 20:40] = 200.0
    m = fast_mask(img, threshold=50)
    ys, xs = np.nonzero(m)
    assert len(xs) > 0
    corners = np.array([[20, 20], [39, 20], [20, 39], [39, 39]])
    for x, y in zip(xs, ys):
        # every detection must lie close to one of the four square corners
        assert np.min(np.abs(corners - [x, y]).max(axis=1)) <= 2
    # flat region and straight edge midpoints are not corners
    assert not m[30, 30] and not m[20, 30] and not m[5, 5]


def test_shi_tomasi_prefers_corners_over_edges():
    img = np.zeros((60, 60), np.float32)
    img[20:40, 20:40] = 200.0
    s = shi_tomasi_score(img)
    assert s[20, 20] > 10 * s[20, 30]          # corner vs edge midpoint
    assert s[30, 30] < 1e-6                     # flat


def test_nms_keeps_single_max():
    score = np.zeros((7, 7))
    score[3, 3], score[3, 4] = 5.0, 4.0
    keep = nms3x3(score, score > 0)
    assert keep[3, 3] and not keep[3, 4]


def test_detect_features_spread_and_count():
    img = render_pattern_image(320, 240, seed=2, smooth=2.0)
    pts, sc = detect_features(img, max_features=60, threshold=8)
    assert 30 <= len(pts) <= 60
    assert len(sc) == len(pts) and np.all(sc > 0)
    # grid bucketing: no two features closer than 2 px
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2) + np.eye(len(pts)) * 1e9
    assert d.min() >= 2.0


def test_bilinear_exact_on_linear_image():
    yy, xx = np.mgrid[0:20, 0:30].astype(np.float32)
    img = 2 * xx + 3 * yy
    x, y = np.array([3.25, 10.5]), np.array([7.75, 2.1])
    assert np.allclose(bilinear(img, x, y), 2 * x + 3 * y, atol=1e-4)


def _interior(pts, w, h, m=40):
    return (pts[:, 0] > m) & (pts[:, 0] < w - m) & (pts[:, 1] > m) & (pts[:, 1] < h - m)


@pytest.mark.parametrize("shift,levels", [((0.37, -0.81), 1), ((3.3, -2.7), 4),
                                          ((12.4, 7.9), 4), ((25.2, -18.6), 5)])
def test_klt_recovers_subpixel_shift(shift, levels):
    img = render_pattern_image(640, 480, seed=1, smooth=4.0)
    img2 = fourier_shift(img, *shift)
    pts, _ = detect_features(img, 120, threshold=5)
    pts = pts[_interior(pts, 640, 480, 60)]
    p0, p1 = build_pyramid(img, levels), build_pyramid(img2, levels)
    new, ok, fb = track_bidirectional(p0, p1, pts, fb_threshold=0.2)
    err = np.linalg.norm(new - pts - np.array(shift), axis=1)
    assert ok.mean() > 0.8
    assert np.median(err[ok]) < 0.03
    assert np.max(err[ok]) < 0.15


def test_klt_with_guess_handles_large_motion():
    img = render_pattern_image(400, 300, seed=4, smooth=3.0)
    shift = np.array([60.0, -35.0])
    img2 = fourier_shift(img, *shift)
    pts, _ = detect_features(img, 100, threshold=5)
    pts = pts[(pts[:, 0] > 70) & (pts[:, 0] < 330) & (pts[:, 1] > 50) & (pts[:, 1] < 250)]
    p0, p1 = build_pyramid(img, 3), build_pyramid(img2, 3)
    guess = pts + shift + np.array([2.0, -1.5])     # e.g. gyro prediction with some error
    new, ok, _ = track(p0, p1, pts, guess)
    assert ok.mean() > 0.9
    assert np.median(np.linalg.norm(new[ok] - pts[ok] - shift, axis=1)) < 0.03


def test_stereo_frontend_ids_persist_and_stereo_matches():
    w, h, f = 400, 300, 300.0
    R = R_BC_FORWARD
    cams = [Camera(w, h, f, f, w / 2, h / 2, R_bc=R, p_bc=np.array([0, 0.03, 0])),
            Camera(w, h, f, f, w / 2, h / 2, R_bc=R, p_bc=np.array([0, -0.03, 0]))]
    # fronto-parallel textured plane at depth Z: right image = left shifted by -f*b/Z
    Z, b = 2.0, 0.06
    disp = f * b / Z
    left = render_pattern_image(w, h, seed=5, smooth=3.0)
    right = fourier_shift(left, -disp, 0.0)
    fe = StereoFrontend(cams, FrontendConfig(max_features=80, fast_threshold=5))
    m1 = fe.process(left, right)
    stereo = [v for v in m1.values() if len(v) == 2]
    assert len(m1) > 40 and len(stereo) > 0.7 * len(m1)
    for obs in stereo:
        # normalized disparity equals b / Z, epipolar line is horizontal
        assert abs((obs[0][1][0] - obs[1][1][0]) - b / Z) < 0.5 / f
        assert abs(obs[0][1][1] - obs[1][1][1]) < 0.5 / f
    # next frame: whole scene shifted by 2.5 px -> same ids continue
    left2 = fourier_shift(left, 2.5, 1.0)
    right2 = fourier_shift(right, 2.5, 1.0)
    m2 = fe.process(left2, right2)
    common = set(m1) & set(m2)
    assert len(common) > 0.8 * len(m1)
    for fid in list(common)[:20]:
        dx = (m2[fid][0][1] - m1[fid][0][1]) * f
        assert np.allclose(dx, [2.5, 1.0], atol=0.1)
