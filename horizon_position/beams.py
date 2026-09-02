"""Beam patterns for the antenna comparison: sampling conversion and idealisations.

`eigsim` ships the EIGSEP bowtie already on the MWSS grid, but the Vivaldi
feed exists only as a HEALPix map, so it has to be brought onto the same grid
before the two can be pushed through an identical pipeline. Everything here is
pure: no I/O, no config.

JAX_ENABLE_X64 is set on import because `s2fft` picks up the flag when jax is
first imported, and a float32 transform is not accurate enough to reproduce the
stored MWSS bowtie. `setdefault` leaves an explicit caller setting alone.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import s2fft  # noqa: E402

# HEALPix quadrature is not exact, so the forward transform needs iterative
# refinement. This is the value eigsim's stored MWSS bowtie was made with:
# test_beams.py checks the round trip reproduces it bit-for-bit, and at
# niter=0 the same round trip is off by 12 per cent.
NITER = 3


def healpix_to_mwss(bm_hp, nside, lmax, niter=NITER):
    """Resample a HEALPix beam onto the MWSS grid via its spherical harmonics.

    Parameters
    ----------
    bm_hp : array_like
        Beam power pattern, shape ``(n_freqs, n_pix)``.
    nside : int
        HEALPix nside of ``bm_hp``.
    lmax : int
        Band limit to transform at. The MWSS grid that comes back is the one
        ``eigsim`` uses at this ``lmax``, i.e. ``(lmax + 2, 2 * lmax + 2)``.
    niter : int
        Forward-transform refinement iterations. Do not lower it; see NITER.

    Returns
    -------
    np.ndarray
        Beam on the MWSS grid, shape ``(n_freqs, lmax + 2, 2 * lmax + 2)``.

    """
    bm_hp = np.asarray(bm_hp, dtype=np.float64)
    if bm_hp.ndim != 2:
        raise ValueError(f"expected (n_freqs, n_pix), got shape {bm_hp.shape}")
    L = lmax + 1
    out = []
    for row in bm_hp:  # one frequency at a time; vmap over 200+ maps OOMs
        alm = s2fft.forward(
            row,
            L=L,
            spin=0,
            nside=nside,
            sampling="healpix",
            method="jax",
            reality=True,
            iter=niter,
        )
        out.append(
            np.asarray(
                s2fft.inverse(
                    alm, L=L, spin=0, sampling="mwss", method="jax", reality=True
                )
            )
        )
    return np.stack(out)


def isotropic_beam(n_freqs, grid_shape):
    """A uniform, frequency-independent beam on the MWSS grid.

    The chromaticity-free reference: it still sees the real horizon, so its
    open-sky fraction is pure geometry and its residual curve isolates what the
    sky alone contributes. Any constant works -- `eigsim.simulate` normalises
    by the full-sphere beam integral -- so this is ones.
    """
    return np.ones((n_freqs, *grid_shape), dtype=np.float64)


def band_limited_power_fraction(bm_hp, nside, lmax, niter=NITER):
    """Fraction of each channel's beam power that survives truncation at ``lmax``.

    A directive beam is not necessarily resolved by the grid its HEALPix map
    was sampled on. Values indistinguishable from 1 mean the comparison is not
    limited by the band limit; anything materially below 1 means the beam is
    being smoothed and the chromaticity it shows is not its own.
    """
    bm_hp = np.asarray(bm_hp, dtype=np.float64)
    L = lmax + 1
    out = []
    for row in bm_hp:
        alm = s2fft.forward(
            row,
            L=L,
            spin=0,
            nside=nside,
            sampling="healpix",
            method="jax",
            reality=True,
            iter=niter,
        )
        back = np.asarray(
            s2fft.inverse(
                alm,
                L=L,
                spin=0,
                nside=nside,
                sampling="healpix",
                method="jax",
                reality=True,
            )
        )
        out.append(np.sum(back**2) / np.sum(row**2))
    return np.array(out)
