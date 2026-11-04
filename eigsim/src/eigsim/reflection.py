"""The sky reflected by a flat ground: a flat-mirror column (notebook 036 G0).

A beam that sees the ground below the local horizontal also sees the sky
reflected by it. For a flat, specular ground at the antenna (a mirror
about the local horizontal plane, P = diag(1, 1, -1) in ENU) the power
the beam receives from a ground direction n is the sky's in the mirrored
direction P n, times the ground's power reflectivity R:

    T_refl(t) = R * int B(R_d^-1 n) M(n) T_sky(P n, t) dOmega / int B,

where R_d is the drive rotation (body to ENU) and M selects ground
directions below the horizontal whose mirrored ray reaches open sky.
Substituting m = P n turns this into an ordinary sky integral over the
upper hemisphere with a *mirrored* beam:

    B(R_d^-1 P m) = B_flip(R'^-1 m),   R' = P R_d P,

with B_flip(x) = B(P x) the beam flipped about its own body x-y plane
(theta -> pi - theta) and R' a proper rotation. For the drive
R_d = mis Rx(el) Rz(az), P Rx(el) P = Rx(-el) and P Rz(az) P = Rz(az),
so R' = (P mis P) Rx(-el) Rz(az): the mirrored column is
:func:`~eigsim.simulate.simulate_path` with the flipped beam, the
elevation negated and the misalignment conjugated by P.

The mirrored direction m must be open sky (elevation > alpha_h(az)) and
its partner P m ground (-elevation <= alpha_h): together elevation >
|alpha_h|, i.e. the open-sky weight of the horizon |alpha_h|
(:func:`mirror_open_sky_weight`).

R is applied by the caller as a frequency shape (an incoherent, power
sum); :func:`fresnel_normal_incidence` gives the normal-incidence value
for a lossy dielectric half-space.
"""

import jax.numpy as jnp
import numpy as np
import s2fft

from .horizon import open_sky_weight
from .simulate import compute_fgnd, simulate_path

#: The mirror about the local horizontal plane (ENU).
P_MIRROR = np.diag([1.0, 1.0, -1.0])

EPS0 = 8.8541878128e-12  # F/m


def mirror_beam(beam_data, sampling="mwss"):
    """The beam flipped about its body x-y plane, B_flip(x) = B(P x).

    On an equiangular grid whose colatitudes are symmetric about pi/2
    (MWSS: theta_t = pi t / L, t = 0..L), the flip reverses the theta
    axis; phi is unchanged.

    Parameters
    ----------
    beam_data : array_like, (n_freq, n_theta, n_phi)
    sampling : str
        Only ``"mwss"`` (symmetric colatitudes) is supported.
    """
    if sampling != "mwss":
        raise ValueError("mirror_beam needs MWSS sampling (symmetric theta)")
    beam = np.asarray(beam_data)
    L = beam.shape[1] - 1
    th = np.asarray(s2fft.sampling.s2_samples.thetas(L=L, sampling="mwss"))
    if th.size != beam.shape[1] or not np.allclose(th + th[::-1], np.pi):
        raise ValueError("beam theta axis is not MWSS-symmetric")
    return beam[:, ::-1, :].copy()


def mirror_misalignment(misalignment=None):
    """P mis P: the misalignment of the mirrored geometry (identity: None)."""
    if misalignment is None:
        return None
    return P_MIRROR @ np.asarray(misalignment, dtype=float) @ P_MIRROR


def mirror_open_sky_weight(alpha_h, az_grid, lmax, **kw):
    """Open-sky weight of mirrored directions: elevation > |alpha_h|.

    See the module docstring; for a horizon above the horizontal
    everywhere this equals :func:`~eigsim.horizon.open_sky_weight`.
    """
    return open_sky_weight(jnp.abs(jnp.asarray(alpha_h)), az_grid, lmax, **kw)


def fresnel_normal_incidence(freqs_mhz, eps_r, sigma_s_m):
    """Power reflectivity at normal incidence of a lossy half-space.

    R = |(1 - sqrt(eps_c)) / (1 + sqrt(eps_c))|^2 with
    eps_c = eps_r - i sigma / (omega eps0). Polarisation-independent at
    normal incidence.

    Parameters
    ----------
    freqs_mhz : array_like
    eps_r : float
        Relative permittivity.
    sigma_s_m : float
        Conductivity, S/m.
    """
    w = 2 * np.pi * np.asarray(freqs_mhz, dtype=float) * 1e6
    n = np.sqrt(eps_r - 1j * sigma_s_m / (w * EPS0))
    return np.abs((1 - n) / (1 + n)) ** 2


def reflected_sky_path(
    beam_data,
    freqs_mhz,
    sky,
    times_jd,
    elevations_deg,
    azimuths_deg,
    horizon_mirror,
    misalignment=None,
    sampling="mwss",
    sky_alm=None,
    **sim_kw,
):
    """The flat-mirror reflected sky per time sample, before R.

    Parameters
    ----------
    beam_data : array_like, (n_freq, n_theta, n_phi)
        Unrotated (body-frame) beam power, MWSS.
    freqs_mhz, sky, times_jd, elevations_deg, azimuths_deg
        As :func:`~eigsim.simulate.simulate_path` (the *real* drive
        angles; the elevation is negated here).
    horizon_mirror : array_like, (n_theta, n_phi)
        :func:`mirror_open_sky_weight` on the beam's grid.
    misalignment : (3, 3) array or None
        The real mount misalignment (conjugated by P here).
    **sim_kw
        Passed on (``lat``, ``lon``, ``alt``, ...). ``Tgnd`` is fixed at
        0: the column carries the reflected sky only.

    Returns
    -------
    s_mirror : np.ndarray, (n_time, n_freq)
        int B_flip M T_sky / int B, K (the sky map as given, so without a
        CMB the map does not carry).
    f_mirror : np.ndarray, (n_time, n_freq)
        int B_flip M / int B: the beam fraction that carries reflected
        sky (a CMB adds T_CMB f_mirror).
    """
    sim_kw = dict(sim_kw)
    sim_kw["Tgnd"] = 0.0
    flip = mirror_beam(beam_data, sampling)
    mis = mirror_misalignment(misalignment)
    el = -np.asarray(elevations_deg, dtype=float)
    az = np.asarray(azimuths_deg, dtype=float)
    beam_kw = {"horizon": horizon_mirror}
    s = simulate_path(
        flip,
        freqs_mhz,
        sky,
        times_jd,
        el,
        az,
        sampling=sampling,
        beam_kw=beam_kw,
        sky_alm=sky_alm,
        misalignment=mis,
        **sim_kw,
    )
    ori, inv = np.unique(np.column_stack([el, az]), axis=0, return_inverse=True)
    fg = compute_fgnd(
        flip,
        freqs_mhz,
        ori[:, 0],
        ori[:, 1],
        sampling=sampling,
        beam_kw=beam_kw,
        misalignment=mis,
    )
    f_mirror = 1.0 - np.asarray(fg)[inv.reshape(-1)]
    return np.asarray(s), f_mirror
