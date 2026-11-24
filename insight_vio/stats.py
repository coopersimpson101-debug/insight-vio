"""Small statistics helpers (no SciPy dependency)."""
from __future__ import annotations

import math
from functools import lru_cache


def normal_quantile(p: float) -> float:
    """Inverse standard-normal CDF by bisection on ``math.erf`` (accurate to ~1e-12)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _chi2_cdf(x: float, k: int) -> float:
    """Regularised lower incomplete gamma P(k/2, x/2) via series / continued fraction."""
    if x <= 0:
        return 0.0
    a, z = 0.5 * k, 0.5 * x
    gln = math.lgamma(a)
    if z < a + 1.0:
        s, term, n = 1.0 / a, 1.0 / a, a
        for _ in range(1000):
            n += 1.0
            term *= z / n
            s += term
            if abs(term) < abs(s) * 1e-15:
                break
        return s * math.exp(-z + a * math.log(z) - gln)
    # Lentz continued fraction for Q
    b = z + 1.0 - a
    c, d = 1.0 / 1e-300, 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return 1.0 - math.exp(-z + a * math.log(z) - gln) * h


@lru_cache(maxsize=4096)
def chi2_quantile(p: float, dof: int) -> float:
    """Chi-squared quantile (e.g. ``chi2_quantile(0.95, 1) = 3.841``) by bisection on the CDF."""
    if dof <= 0:
        raise ValueError("dof must be positive")
    # Wilson-Hilferty initial bracket
    z = normal_quantile(p)
    wh = dof * (1.0 - 2.0 / (9.0 * dof) + z * math.sqrt(2.0 / (9.0 * dof))) ** 3
    lo, hi = 0.0, max(4.0 * wh, 10.0 + 10.0 * dof)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _chi2_cdf(mid, dof) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
