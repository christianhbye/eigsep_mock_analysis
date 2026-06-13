"""Anti-aliased open-sky weight on the beam's MWSS grid.

The horizon comes from eigsep_terrain as an elevation curve alpha_h(az),
azimuth = atan2(E, N) (North->East). The beam grid is MWSS with polar
angle theta (0 = zenith, pi = nadir) and azimuth phi from ENU East
(croissant beam_rot=0). Open sky <=> elevation > alpha_h <=> theta <
theta_h, with theta_h = pi/2 - alpha_h. The frame map between the two
azimuths is phi = pi/2 - az.

The returned weight W(theta, phi) in [0, 1] is the fraction of each
theta-cell that lies above the horizon, so a sub-pixel horizon shift
changes the boundary cell continuously (no boolean-grid floor).
"""

import numpy as np
import s2fft.sampling.s2_samples as s2


def mwss_grid(lmax):
    """Return ``(thetas, phis)`` [rad] for the MWSS grid at ``lmax``."""
    L = lmax + 1
    thetas = np.asarray(s2.thetas(L, sampling="mwss"))
    phis = np.asarray(s2.phis_equiang(L, sampling="mwss"))
    return thetas, phis


def _theta_edges(thetas):
    """Cell edges: midpoints between thetas, with poles at 0 and pi."""
    mid = 0.5 * (thetas[1:] + thetas[:-1])
    return np.concatenate([[0.0], mid, [np.pi]])


def _alpha_on_phi(alpha_h, az_grid, phis):
    """Interpolate alpha_h(az) onto MWSS azimuths via phi = pi/2 - az."""
    az_of_phi = np.mod(np.pi / 2 - phis, 2 * np.pi)
    # periodic linear interpolation in azimuth
    xp = np.concatenate([az_grid, az_grid[:1] + 2 * np.pi])
    fp = np.concatenate([alpha_h, alpha_h[:1]])
    order = np.argsort(xp)
    return np.interp(az_of_phi, xp[order], fp[order])


def open_sky_weight(alpha_h, az_grid, thetas, phis):
    """Fractional open-sky weight ``W(theta, phi)`` in ``[0, 1]``.

    Parameters
    ----------
    alpha_h : (n_az,) horizon elevation [rad] vs azimuth.
    az_grid : (n_az,) azimuth [rad] of each ``alpha_h`` sample
        (= ``atan2(E, N)``).
    thetas, phis : MWSS grid axes [rad] from :func:`mwss_grid`.

    Returns
    -------
    W : (n_theta, n_phi) float, 1 = open sky, 0 = blocked.
    """
    alpha_h = np.asarray(alpha_h, dtype=np.float64)
    theta_h = np.pi / 2 - _alpha_on_phi(alpha_h, np.asarray(az_grid), phis)
    edges = _theta_edges(thetas)
    lo = edges[:-1][:, None]
    hi = edges[1:][:, None]
    # fraction of [lo, hi] with theta < theta_h (open)
    frac = (theta_h[None, :] - lo) / (hi - lo)
    return np.clip(frac, 0.0, 1.0)


def boolean_weight(alpha_h, az_grid, thetas, phis):
    """Boolean (0/1) open-sky mask via a cell-center test (cross-checks)."""
    W = open_sky_weight(alpha_h, az_grid, thetas, phis)
    return (W >= 0.5).astype(np.float64)
