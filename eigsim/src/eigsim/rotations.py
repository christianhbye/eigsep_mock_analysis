"""Beam rotation utilities for the EIGSEP drive system.

The EIGSEP antenna sits on a turntable mounted on a box that hangs from
a horizontal suspension cable in a lunar crater.  Two mechanical degrees
of freedom let the antenna scan the sky:

* **Elevation drive** — swings the entire box around the suspension
  (X-axis in ENU).  ``elevation_deg = 0`` means the antenna points at
  zenith; positive angles tilt toward local South (right-hand rule).
* **Azimuth drive** — spins the antenna on a turntable on top of the
  box (Z'-axis of the box frame).

The combined rotation in the fixed ENU frame is

    R = Rx(elevation) @ Rz(azimuth)

i.e. the turntable acts first (in the box frame), then the box tilts.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import s2fft
from croissant.rotations import rotmat_to_eulerZYZ

# ── elementary rotation matrices ──────────────────────────────────────


def rotation_matrix_x(angle_rad):
    """Rotation matrix around the X-axis (East in ENU)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_matrix_z(angle_rad):
    """Rotation matrix around the Z-axis (Up in ENU)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation_matrix_y(angle_rad):
    """Rotation matrix around the Y-axis (North in ENU).

    The drive cannot produce this rotation — it has only an elevation
    axis (X) and a turntable (Z) — which is exactly why a Y tilt is an
    identifiable misalignment rather than an encoder offset.

    **Direction (load-bearing for the sign of dT/d eps_y).** Right-hand
    rule about +Y (North), so a *positive* angle tilts the zenith toward
    **East**: ``R_y(a) @ zhat == (sin a, 0, cos a)``.  Mirror of
    elevation, which tilts toward South for a positive angle about +X.
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def misalignment_matrix(tilt_y_deg=0.0, tilt_z_deg=0.0):
    """Outer (mount-to-ground) misalignment of the whole instrument.

    ``tilt_y_deg`` is the levelling error about the North axis and
    ``tilt_z_deg`` the error of the azimuth reference against true North.
    A tilt about X is omitted on purpose: it is exactly an elevation
    encoder offset (``Rx(eps) @ Rx(el) == Rx(eps + el)``) and carries no
    independent information.

    **Directions**, both right-handed about their ENU axis, and both
    load-bearing for the signs of ``dT/d eps_y`` and ``dT/d eps_z``:

    * positive ``tilt_y_deg`` tilts the boresight toward **East** (the
      East component of the boresight becomes ``+sin(eps_y) cos(el)``);
    * positive ``tilt_z_deg`` rotates the azimuth reference **East toward
      North**, i.e. the same sense as the turntable's own azimuth, and
      simply adds to it: at zenith ``Rz(eps) @ Rx(0) @ Rz(az) ==
      Rx(0) @ Rz(az + eps)``.
    """
    return rotation_matrix_z(np.radians(tilt_z_deg)) @ rotation_matrix_y(
        np.radians(tilt_y_deg)
    )


# ── EIGSEP drive rotation ────────────────────────────────────────────


def drive_rotation_matrix(elevation_deg, azimuth_deg, misalignment=None):
    """Combined rotation matrix for the EIGSEP drive system.

    Parameters
    ----------
    elevation_deg : float
        Elevation drive angle in degrees.  0 = zenith pointing.
        Follows the right-hand rule around the X-axis (East):
        positive angles tilt the antenna toward South.
    azimuth_deg : float
        Turntable angle in degrees.  Positive = counterclockwise
        when viewed from above (East toward North).
    misalignment : (3, 3) array or None
        Outer mount-to-ground misalignment, applied *outside* the drive
        as ``R_mis @ Rx(el) @ Rz(az)`` so that it is static in the
        topocentric frame and does not rotate with the drive.  ``None``
        reproduces the bare drive exactly.

    Returns
    -------
    R : np.ndarray
        3x3 rotation matrix.

    """
    R = rotation_matrix_x(np.radians(elevation_deg)) @ rotation_matrix_z(
        np.radians(azimuth_deg)
    )
    if misalignment is None:
        return R
    return np.asarray(misalignment) @ R


# ── beam-data rotation ───────────────────────────────────────────────


def beam_to_alm(data, lmax, sampling, nside=None, niter=0):
    """Forward SHT of beam data (vectorised over frequencies).

    Parameters
    ----------
    data : array_like
        Beam power pattern.  Shape ``(N_freqs, N_theta, N_phi)`` for
        non-healpix samplings, or ``(N_freqs, N_pix)`` for healpix.
    lmax : int
        Maximum spherical harmonic degree of the data.
    sampling : str
        Sampling scheme (``"mwss"``, ``"healpix"``, etc.).
    nside : int or None
        HEALPix nside.  Required when ``sampling="healpix"``.
    niter : int
        Number of SHT iterations (passed to ``s2fft``).

    Returns
    -------
    alm : jax.Array
        Spherical harmonic coefficients, shape
        ``(N_freqs, lmax+1, 2*lmax+1)``.

    """
    L = lmax + 1
    data = jnp.asarray(data)
    fwd = partial(
        s2fft.forward,
        L=L,
        spin=0,
        nside=nside,
        sampling=sampling,
        method="jax",
        reality=True,
        iter=niter,
    )
    return jax.vmap(fwd)(data)


def rotate_alm_to_beam(
    alm,
    lmax,
    sampling,
    elevation_deg,
    azimuth_deg,
    nside=None,
):
    """Wigner-D rotation of alm followed by inverse SHT.

    Models the **commanded drive only** — ``Rx(elevation) @
    Rz(azimuth)``.  It takes no misalignment: a mount-to-ground
    misalignment is applied by :func:`eigsim.simulate`,
    :func:`eigsim.simulate_path` and :func:`eigsim.compute_fgnd` through
    their ``misalignment=`` argument, which they pass to
    :func:`drive_rotation_matrix` themselves.

    Parameters
    ----------
    alm : jax.Array
        Spherical harmonic coefficients from :func:`beam_to_alm`.
    lmax : int
        Maximum spherical harmonic degree.
    sampling : str
        Sampling scheme for the inverse SHT.
    elevation_deg : float
        Elevation drive angle in degrees.
    azimuth_deg : float
        Turntable angle in degrees.
    nside : int or None
        HEALPix nside.  Required when ``sampling="healpix"``.

    Returns
    -------
    rotated : jax.Array
        Rotated beam data in pixel space.

    """
    L = lmax + 1

    # Wigner-D rotation
    R = drive_rotation_matrix(elevation_deg, azimuth_deg)
    euler = rotmat_to_eulerZYZ(R)
    dl_array = s2fft.generate_rotate_dls(L, euler[1])

    rot = partial(
        s2fft.utils.rotation.rotate_flms,
        L=L,
        rotation=euler,
        dl_array=dl_array,
    )
    alm_rot = jax.vmap(rot)(alm)

    # inverse SHT back to pixel space
    inv = partial(
        s2fft.inverse,
        L=L,
        spin=0,
        nside=nside,
        sampling=sampling,
        method="jax",
        reality=True,
    )
    return jax.vmap(inv)(alm_rot)


def rotate_beam_data(
    data,
    lmax,
    sampling,
    elevation_deg,
    azimuth_deg,
    nside=None,
    niter=0,
):
    """Rotate beam data for a given drive configuration.

    Convenience wrapper that calls :func:`beam_to_alm` followed by
    :func:`rotate_alm_to_beam`.  When calling this for many
    orientations of the same beam, prefer computing the alm once with
    :func:`beam_to_alm` and then calling :func:`rotate_alm_to_beam`
    per orientation.

    Like :func:`rotate_alm_to_beam`, this models the **commanded drive
    only**.  A mount-to-ground misalignment is applied by
    :func:`eigsim.simulate`, :func:`eigsim.simulate_path` and
    :func:`eigsim.compute_fgnd` through their ``misalignment=``
    argument; it is not reachable from here.

    Parameters
    ----------
    data : array_like
        Beam power pattern.  Shape ``(N_freqs, N_theta, N_phi)`` for
        non-healpix samplings, or ``(N_freqs, N_pix)`` for healpix.
    lmax : int
        Maximum spherical harmonic degree of the data.
    sampling : str
        Sampling scheme (``"mwss"``, ``"healpix"``, etc.).
    elevation_deg : float
        Elevation drive angle in degrees.
    azimuth_deg : float
        Turntable angle in degrees.
    nside : int or None
        HEALPix nside.  Required when ``sampling="healpix"``.
    niter : int
        Number of SHT iterations (passed to ``s2fft``).

    Returns
    -------
    rotated : jax.Array
        Rotated beam data, same shape and sampling as *data*.

    """
    if np.isclose(elevation_deg, 0.0) and np.isclose(azimuth_deg, 0.0):
        return jnp.asarray(data)

    alm = beam_to_alm(data, lmax, sampling, nside=nside, niter=niter)
    return rotate_alm_to_beam(
        alm, lmax, sampling, elevation_deg, azimuth_deg, nside=nside
    )
