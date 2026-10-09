"""Declared azimuth-step injections with externally supplied contamination."""

import numpy as np

from .drive import uniform_window_means


def ramp_window_means(edges, start, duration):
    """Analytic means of a unit clipped ramp, including partial windows."""
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("positive finite ramp duration required")
    return uniform_window_means(edges, [(start, start + duration, 1 / duration)], y0=0)


def simulate_azimuth_window(
    edges, *, count_start, duration, tau_s, step, contamination, noise_sd=0, rng=None
):
    """Add a lagged step to caller-supplied independent elevation structure.

    Contamination may be a measured neighboring-leg profile. This generator
    never learns it from the fitted nuisance coefficients. Noise is per-row
    independent Gaussian here; this is an explicitly declared idealization.
    """
    contamination = np.asarray(contamination, float)
    n = len(edges) - 1
    if contamination.shape not in ((n,), (2, n)):
        raise ValueError("one contamination value per integration required")
    if not np.isfinite([count_start, tau_s, step, noise_sd]).all() or noise_sd < 0:
        raise ValueError("finite step parameters and nonnegative noise required")
    rng = np.random.default_rng() if rng is None else rng
    ramp = ramp_window_means(edges, count_start - tau_s, duration)
    if contamination.ndim == 2:
        # Two measured neighboring-azimuth profiles, supplied by the
        # caller and normalized independently at the turnaround. Their
        # variation across the injected slew is an explicit blend model.
        contamination = (1 - ramp) * contamination[0] + ramp * contamination[1]
    return step * ramp + contamination + noise_sd * rng.normal(size=len(contamination))
