"""Fractional open-sky weight on the beam's MWSS grid, in JAX.

The horizon arrives as an elevation curve ``alpha_h(az)`` with
``az = atan2(E, N)`` (North->East).  The beam grid is MWSS with polar
angle theta (0 = zenith) and azimuth phi measured from ENU East
(croissant ``beam_rot=0``), so the frame map is ``phi = pi/2 - az``.
Open sky <=> elevation > alpha_h <=> theta < theta_h with
``theta_h = pi/2 - alpha_h``.

``W(theta, phi)`` in [0, 1] is the fraction of each grid **cell** above
the horizon, integrated over both theta and phi.  Integrating over the
phi cell is what band-limits the horizon: a curve sampled far finer than
the ~258 MWSS azimuths is averaged into the cell rather than
point-sampled at its centre, so no separate azimuth reduction step is
needed or wanted (eigsep_mock_analysis issue #10).

Everything is built in ``jnp`` so that ``dW/d alpha_h`` comes from
autodiff: a boolean mask has zero gradient almost everywhere and cannot
carry a horizon derivative at all.
"""

import jax.numpy as jnp
import numpy as np
import s2fft.sampling.s2_samples as s2

#: phi sub-samples per cell.  Fixed by the convergence test, not chosen:
#: 258 x 180 = 46440 fine azimuths against the native curve's 46080, so
#: the integral samples the curve at essentially its own resolution.
DEFAULT_SUB = 180


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


def open_sky_weight(alpha_h, az_grid, lmax, sub=DEFAULT_SUB):
    """Fractional open-sky weight ``W(theta, phi)`` in ``[0, 1]``.

    Parameters
    ----------
    alpha_h : (n_az,) array
        Horizon elevation [rad] vs azimuth.  Differentiable input.
    az_grid : (n_az,) array
        Azimuth [rad] of each ``alpha_h`` sample, ``= atan2(E, N)``,
        ascending on ``[0, 2 pi)``.
    lmax : int
        Band limit of the beam grid.
    sub : int
        Sub-samples per phi cell for the azimuth integral.

    Returns
    -------
    W : (n_theta, n_phi) jax.Array
        1 = open sky, 0 = blocked.

    """
    # az_grid is a fixed grid, validated concretely; alpha_h is the
    # differentiable input and is never inspected for its values.
    az_np = np.asarray(az_grid, dtype=np.float64)
    alpha_h = jnp.asarray(alpha_h)
    if az_np.ndim != 1 or alpha_h.shape != az_np.shape:
        raise ValueError(
            f"alpha_h {alpha_h.shape} and az_grid {az_np.shape} must be "
            "the same 1-D shape"
        )
    if np.any(np.diff(az_np) <= 0):
        raise ValueError(
            "az_grid must be strictly ascending; alpha_h is defined on "
            "az = atan2(E, N) over [0, 2*pi)"
        )
    if az_np[0] < 0.0 or az_np[-1] >= 2 * np.pi:
        raise ValueError(
            f"az_grid must lie in [0, 2*pi), got [{az_np[0]}, {az_np[-1]}]"
        )
    az_grid = jnp.asarray(az_np)

    thetas, phis = mwss_grid(lmax)
    n_theta, n_phi = thetas.size, phis.size
    dphi = 2.0 * np.pi / n_phi

    # phi cell centres -> sub sample points spanning each cell
    off = (jnp.arange(sub) + 0.5) / sub - 0.5
    fine_phi = (jnp.asarray(phis)[:, None] + off[None, :] * dphi).ravel()

    az_of_phi = jnp.mod(jnp.pi / 2 - fine_phi, 2 * jnp.pi)
    alpha_fine = jnp.interp(az_of_phi, az_grid, alpha_h, period=2 * jnp.pi)
    theta_h = jnp.pi / 2 - alpha_fine

    edges = _theta_edges(thetas)
    lo = jnp.asarray(edges[:-1])[:, None]
    hi = jnp.asarray(edges[1:])[:, None]
    # Fraction of [lo, hi] with theta < theta_h.  Linear in theta, not
    # sin-theta weighted: cells are ~1.4 deg so sin is nearly constant
    # across one, making this first-order accurate in the sub-cell
    # horizon position.
    frac = jnp.clip((theta_h[None, :] - lo) / (hi - lo), 0.0, 1.0)
    return frac.reshape(n_theta, n_phi, sub).mean(axis=2)
