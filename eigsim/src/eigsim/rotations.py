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


# ── EIGSEP drive rotation ────────────────────────────────────────────


def drive_rotation_matrix(elevation_deg, azimuth_deg):
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

    Returns
    -------
    R : np.ndarray
        3x3 rotation matrix.

    """
    return rotation_matrix_x(np.radians(elevation_deg)) @ rotation_matrix_z(
        np.radians(azimuth_deg)
    )


# ── beam-data rotation ───────────────────────────────────────────────


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

    The rotation is carried out in spherical-harmonic space:
    forward SHT -> Wigner-D rotation -> inverse SHT.  The returned
    array has the same shape and sampling as the input and can be
    passed directly to ``croissant.Beam``.

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

    L = lmax + 1
    data = jnp.asarray(data)

    # forward SHT (vectorised over frequency axis)
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
    alm = jax.vmap(fwd)(data)

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
    rotated = jax.vmap(inv)(alm_rot)

    return rotated
