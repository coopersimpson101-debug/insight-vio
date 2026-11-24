"""FAST-9 corner detection with Shi-Tomasi (min-eigenvalue) scoring, hand-written in numpy.

FAST (Rosten & Drummond 2006): pixel ``p`` is a corner if at least ``n`` contiguous
pixels on the 16-pixel Bresenham circle of radius 3 are all brighter than
``I(p) + t`` or all darker than ``I(p) - t``.

Shi-Tomasi (1994) score: the smaller eigenvalue of the windowed structure tensor
``M = sum_w [gx^2, gx gy; gx gy, gy^2]``, i.e.
``lambda_min = (a + c)/2 - sqrt(((a - c)/2)^2 + b^2)``.  We use it to rank FAST
corners, run 3x3 non-maximum suppression and bucket the survivors on a grid so
features are spread across the image (important for VIO conditioning).
"""
from __future__ import annotations

import numpy as np

# (dx, dy) offsets of the radius-3 Bresenham circle, in circular order
CIRCLE = np.array([(0, -3), (1, -3), (2, -2), (3, -1), (3, 0), (3, 1), (2, 2), (1, 3),
                   (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3)])


def _contiguous_arc(b: np.ndarray, n: int) -> np.ndarray:
    """b: (16, h, w) bool ring tests -> (h, w) True where >= n contiguous (circularly)."""
    ext = np.concatenate([b, b[: n - 1]], axis=0)       # 16 + n - 1 entries
    if n == 9:
        w2 = ext[:-1] & ext[1:]
        w4 = w2[:-2] & w2[2:]
        w8 = w4[:-4] & w4[4:]
        w9 = w8[:16] & ext[8:24]
        return w9.any(axis=0)
    acc = ext[0:16].copy()
    for j in range(1, n):
        acc &= ext[j:j + 16]
    return acc.any(axis=0)


def fast_mask(img: np.ndarray, threshold: float = 20.0, arc: int = 9) -> np.ndarray:
    """Boolean corner mask (same shape as ``img``; 3-pixel border is always False)."""
    img = np.asarray(img, dtype=np.float32)
    H, W = img.shape
    c = img[3:H - 3, 3:W - 3]
    ring = np.stack([img[3 + dy:H - 3 + dy, 3 + dx:W - 3 + dx] for dx, dy in CIRCLE])
    # (a C++ port should add the classic 4-compass-pixel early rejection here;
    #  in numpy the fully vectorised test is faster than gathering candidates)
    bright = ring > (c + threshold)
    dark = ring < (c - threshold)
    out = np.zeros((H, W), dtype=bool)
    out[3:H - 3, 3:W - 3] = _contiguous_arc(bright, arc) | _contiguous_arc(dark, arc)
    return out


def sobel(img: np.ndarray):
    """3x3 Sobel gradients (gx, gy), edge-replicated, scaled by 1/8."""
    p = np.pad(np.asarray(img, dtype=np.float32), 1, mode="edge")
    gx = ((p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])) / 8.0
    gy = ((p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])) / 8.0
    return gx, gy


def box_sum(img: np.ndarray, r: int) -> np.ndarray:
    """Sum over a (2r+1)^2 window via an integral image (edge-replicated)."""
    p = np.pad(img, r + 1, mode="edge").astype(np.float64)
    S = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    H, W = img.shape
    return (S[k:k + H, k:k + W] - S[0:H, k:k + W] - S[k:k + H, 0:W] + S[0:H, 0:W])


def shi_tomasi_score(img: np.ndarray, radius: int = 2) -> np.ndarray:
    """Per-pixel minimum eigenvalue of the structure tensor (window (2r+1)^2)."""
    gx, gy = sobel(img)
    a = box_sum(gx * gx, radius)
    b = box_sum(gx * gy, radius)
    c = box_sum(gy * gy, radius)
    return 0.5 * (a + c) - np.sqrt(0.25 * (a - c) ** 2 + b * b)


def nms3x3(score: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep mask pixels whose score is the strict-ish maximum of their 3x3 neighbourhood."""
    s = np.where(mask, score, -np.inf)
    p = np.pad(s, 1, mode="constant", constant_values=-np.inf)
    H, W = s.shape
    keep = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nb = p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            # on ties the pixel earlier in raster order wins
            keep &= (s > nb) | ((s == nb) & ((dy > 0) | ((dy == 0) & (dx > 0))))
    return keep


def detect_features(img: np.ndarray, max_features: int = 150, threshold: float = 20.0,
                    cell_size: int | None = None, allowed: np.ndarray | None = None,
                    border: int = 8, min_score: float = 1e-3):
    """FAST-9 + Shi-Tomasi + NMS + grid bucketing.

    Returns ``(pts (N,2) float [x, y], scores (N,))`` sorted by descending score.
    ``allowed`` is an optional boolean mask (False where detection is suppressed,
    e.g. around already-tracked features).
    """
    img = np.asarray(img, dtype=np.float32)
    H, W = img.shape
    corners = fast_mask(img, threshold)
    if allowed is not None:
        corners &= allowed
    corners[:border, :] = False
    corners[-border:, :] = False
    corners[:, :border] = False
    corners[:, -border:] = False
    score = shi_tomasi_score(img)
    corners &= score > min_score
    corners = nms3x3(score, corners)
    ys, xs = np.nonzero(corners)
    if len(xs) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    sc = score[ys, xs]
    order = np.argsort(-sc, kind="stable")
    xs, ys, sc = xs[order], ys[order], sc[order]
    if cell_size is None:
        cell_size = max(8, int(np.sqrt(H * W / max(max_features, 1))))
    cell = (ys // cell_size) * (W // cell_size + 1) + (xs // cell_size)
    _, first = np.unique(cell, return_index=True)   # best (first in sorted order) per cell
    first = np.sort(first)[:max_features]
    pts = np.stack([xs[first], ys[first]], axis=1).astype(float)
    return pts, sc[first]


def suppression_mask(shape, pts: np.ndarray, radius: int) -> np.ndarray:
    """Boolean mask that is False within ``radius`` px (square) of any point."""
    H, W = shape
    m = np.ones((H, W), dtype=bool)
    for x, y in np.round(np.asarray(pts)).astype(int):
        m[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1] = False
    return m
