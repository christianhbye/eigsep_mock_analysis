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

The cell integral is ``croissant.horizon_weights`` with
``horizon_subsamples``; this module adds the ``alpha_h(az)`` curve, its
validation and the MWSS grid.
"""

import croissant as cro
import jax.numpy as jnp
import numpy as np
import s2fft.sampling.s2_samples as s2

#: phi sub-samples per cell.  Fixed by the convergence test, not chosen:
#: 258 x 180 = 46440 fine azimuths against the native curve's 46080, so
#: the integral samples the curve at essentially its own resolution.
DEFAULT_SUB = 180

#: UTM 12N grid convergence at the D5 site [deg]: true bearings (CCW from
#: East) are grid bearings plus this, so compass ``A_true = A_grid -
#: 1.5208``.  pyproj and the spherical formula agree (eigsep_analysis
#: issue #8); the team's memo 008 uses the rounded 1.52.
GRID_CONVERGENCE_DEG = 1.5208


def grid_bearings_to_az(bearings_deg, convergence_deg=GRID_CONVERGENCE_DEG):
    """Compass azimuth [rad] in ``[0, 2 pi)`` for UTM grid bearings.

    Parameters
    ----------
    bearings_deg : array
        Bearings [deg] counter-clockwise from grid East, as the team's
        ``horizon_profiles`` products store them (``atan2(dN, dE)``).
    convergence_deg : float
        Grid convergence [deg]; the true bearing is the grid bearing
        plus this.

    Returns
    -------
    az : array
        ``atan2(E, N)`` in the true frame, North->East, in ``[0, 2 pi)``.

    """
    bearings = np.asarray(bearings_deg, dtype=np.float64)
    az_deg = 90.0 - (bearings + convergence_deg)
    return np.radians(np.mod(az_deg, 360.0))


def curve_from_grid_bearings(
    bearings_deg, elev_rad, convergence_deg=GRID_CONVERGENCE_DEG
):
    """Turn a grid-bearing horizon profile into an ``open_sky_weight`` curve.

    Returns ``(az_grid, alpha_h)``, both sorted by ``az`` ascending on
    ``[0, 2 pi)``, as :func:`open_sky_weight` requires.
    """
    az = grid_bearings_to_az(bearings_deg, convergence_deg)
    order = np.argsort(az)
    return az[order], np.asarray(elev_rad, dtype=np.float64)[order]


def mwss_grid(lmax):
    """Return ``(thetas, phis)`` [rad] for the MWSS grid at ``lmax``."""
    L = lmax + 1
    thetas = np.asarray(s2.thetas(L, sampling="mwss"))
    phis = np.asarray(s2.phis_equiang(L, sampling="mwss"))
    return thetas, phis


def open_sky_weight(alpha_h, az_grid, lmax, sub=DEFAULT_SUB):
    """Fractional open-sky weight ``W(theta, phi)`` in ``[0, 1]``.

    Cell edges in theta lie halfway between rows, with the poles at 0
    and pi.  Within a cell the open fraction is linear in theta, not
    sin-theta weighted: cells are ~1.4 deg, so sin is nearly constant
    across one, making this first-order accurate in the sub-cell horizon
    position.  It is averaged over *sub* azimuths spread evenly across
    each phi cell.

    Parameters
    ----------
    alpha_h : (n_az,) array
        Horizon elevation [rad] vs azimuth, within ``[-pi/2, pi/2]``.
        Differentiable input; the range is checked only when it is
        concrete, not under a JAX trace.
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

    def theta_h(az):
        # croissant passes compass azimuth, which is our az = atan2(E, N)
        return jnp.pi / 2 - jnp.interp(az, az_grid, alpha_h, period=2 * jnp.pi)

    thetas, phis = mwss_grid(lmax)
    return cro.horizon_weights(thetas, phis, theta_h, horizon_subsamples=sub)
