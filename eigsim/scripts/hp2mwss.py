#!/usr/bin/env python
"""Convert HEALPix .npz data files to MWSS sampling.

The forward SHT on HEALPix is iterative (controlled by --niter) and slow,
but the resulting MWSS representation is exact at the chosen bandwidth so
all subsequent SHTs are fast and lossless.

Usage
-----
    uv run python scripts/hp2mwss.py                  # defaults (niter=3)
    uv run python scripts/hp2mwss.py --niter 5         # more iterations
"""

import argparse
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import s2fft
import s2fft.sampling.s2_samples as s2

jax.config.update("jax_enable_x64", True)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _hp2mwss(data, nside, niter):
    """Forward SHT on HEALPix, inverse SHT on MWSS."""
    lmax = 2 * nside
    L = lmax + 1
    alm = s2fft.forward(
        jnp.asarray(data, dtype=jnp.float64),
        L=L,
        spin=0,
        nside=nside,
        sampling="healpix",
        method="jax",
        reality=True,
        iter=niter,
    )
    return s2fft.inverse(
        alm,
        L=L,
        spin=0,
        sampling="mwss",
        method="jax",
        reality=True,
    )


def convert_beam(niter):
    """Convert the EIGSEP bowtie beam from HEALPix to MWSS."""
    src = DATA_DIR / "eigsep_bowtie_v000.npz"
    dst = DATA_DIR / "eigsep_bowtie_v000_mwss.npz"
    print(f"Converting beam: {src.name} -> {dst.name} (niter={niter})")

    d = np.load(src)
    freqs, bm, nside = d["freqs"], d["bm"], int(d["nside"])
    lmax = 2 * nside
    L = lmax + 1

    fwd = partial(
        s2fft.forward,
        L=L,
        spin=0,
        nside=nside,
        sampling="healpix",
        method="jax",
        reality=True,
        iter=niter,
    )
    inv = partial(
        s2fft.inverse,
        L=L,
        spin=0,
        sampling="mwss",
        method="jax",
        reality=True,
    )

    alm = jax.vmap(fwd)(jnp.asarray(bm, dtype=jnp.float64))
    bm_mwss = np.asarray(jax.vmap(inv)(alm))

    np.savez(dst, freqs=freqs, bm=bm_mwss, lmax=lmax)
    print(f"  Saved {dst.name}: bm shape {bm_mwss.shape}, lmax={lmax}")


def convert_horizon():
    """Convert the horizon model from HEALPix to MWSS.

    Uses nearest-neighbor interpolation instead of SHT because the
    horizon has a sharp NaN/finite boundary that causes Gibbs ringing.
    """
    src = DATA_DIR / "horizon.npz"
    dst = DATA_DIR / "horizon_mwss.npz"
    print(f"Converting horizon: {src.name} -> {dst.name} (nearest-neighbor)")

    d = np.load(src)
    horizon, nside = d["horizon"], int(d["nside"])
    center, height = d["center"], d["height"]

    lmax = 2 * nside
    L = lmax + 1

    thetas = s2.thetas(L, sampling="mwss")
    phis = s2.phis_equiang(L, sampling="mwss")

    # Map each MWSS pixel to its nearest HEALPix pixel
    horizon_mwss = np.empty((len(thetas), len(phis)), dtype=np.float64)
    for i, theta in enumerate(thetas):
        for j, phi in enumerate(phis):
            ipix = s2.hp_ang2pix(nside, float(theta), float(phi))
            horizon_mwss[i, j] = horizon[ipix]

    np.savez(dst, horizon=horizon_mwss, center=center, height=height, lmax=lmax)
    print(f"  Saved {dst.name}: horizon shape {horizon_mwss.shape}, lmax={lmax}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HEALPix data to MWSS sampling"
    )
    parser.add_argument(
        "--niter",
        type=int,
        default=3,
        help="Number of SHT iterations for the healpix forward transform (default: 3)",
    )
    args = parser.parse_args()

    t0 = time.time()
    convert_beam(args.niter)
    convert_horizon()
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
