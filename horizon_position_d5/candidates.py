"""E7/E7-prime hypotheses, not derivatives certified for every terrain maximum."""

import numpy as np


def position_tangents(
    alpha_rad,
    distance_m,
    gradient_en,
    bearings_deg,
    method,
    earth_radius_m=6371e3,
):
    """Return hypothesized d alpha/d(E,N,U), shape (..., n_bearing, 3).

    E7 uses a periodic angular secant. E7-prime uses the selected bilinear
    cell's gradient, which is one-sided at a crease. The vertical identity
    holds at a unique maximum even at fixed near/far boundaries. The radial
    identity fails for a fixed inner cutoff or a discrete march; a stable
    world-fixed outer boundary does not itself invalidate it. The caller
    retains boundary flags rather than excluding those bearings from metrics.
    """
    a, r = np.asarray(alpha_rad), np.asarray(distance_m)
    g = np.asarray(gradient_en)
    beta = np.radians(np.asarray(bearings_deg))
    if (
        beta.ndim != 1
        or len(beta) < 3
        or a.shape != r.shape
        or a.shape[-1] != len(beta)
        or g.shape != (*a.shape, 2)
        or not np.all(np.isfinite(a))
        or not np.all(np.isfinite(r))
        or not np.all(np.isfinite(g))
        or np.any(r <= 0)
    ):
        raise ValueError("invalid horizon, distance, gradient or bearing arrays")
    step = 2 * np.pi / len(beta)
    if not np.allclose(np.diff(beta), step, rtol=0, atol=1e-12):
        raise ValueError("bearings must be a full periodic uniform ascending grid")
    cos2 = np.cos(a) ** 2
    radial = np.sin(a) * np.cos(a) / r + cos2 / earth_radius_m
    if method == "e7":
        slope = -(np.roll(a, -1, axis=-1) - np.roll(a, 1, axis=-1)) / (2 * step)
    elif method == "e7prime":
        slope = cos2 * (g[..., 0] * np.sin(beta) - g[..., 1] * np.cos(beta))
    else:
        raise ValueError("method must be e7 or e7prime")
    cross = slope / r
    return np.stack(
        [
            np.cos(beta) * radial + np.sin(beta) * cross,
            np.sin(beta) * radial - np.cos(beta) * cross,
            -cos2 / r,
        ],
        axis=-1,
    )
