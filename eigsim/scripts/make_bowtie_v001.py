#!/usr/bin/env python
"""Build the v001 EIGSEP bowtie beam from the HFSS complex far field.

SUPERSEDED by make_bowtie_v002.py. The source's frequency labels are one
3.90625 MHz step low (its 46.875 MHz slice is the 50.78 MHz beam; confirmed
by Dominic 2026-09-16), so every v001 beam sits at the wrong frequency.
Kept to reproduce v001 exactly; configs eigsep_v001*.

The source is Dominic's HFSS export as committed to data-analysis
(``hfss_beam_maps/bowtie_beam.npz``): a complex Cartesian E-field
``beam_cart`` (Ex, Ey, Ez) in mV at 1 W incident power, HEALPix RING at
nside 32, on the 52 correlator channels the transmitter uses
(46.875-246.09 MHz, every 16th channel). Its pattern matches BK's
2025-10-31 bowtie-on-box simulation.

The power pattern is |E|^2, so this beam and the complex beam a coherent
ground model needs are the same beam. Each channel is normalised to unit
mean over the sphere (directivity). croissant normalises by the beam
integral, so the scale never reaches a simulation. The realized
efficiency, the |E|^2 gain integrated over the sphere and divided by
4 pi, is stored alongside.

Outputs, in eigsim/data:

- ``eigsep_bowtie_v001.npz``: HEALPix, native channels.
- ``eigsep_bowtie_v001_mwss.npz``: MWSS, native channels (config ``eigsep``).
- ``eigsep_bowtie_v001_1mhz_mwss.npz``: MWSS, cubic spline in frequency to
  50-246 MHz in 1 MHz steps (config ``eigsep_1mhz``). Structure faster than
  the 3.9 MHz channel spacing is not in the source, so the spline invents
  what lies between channels.

The HEALPix maps are transformed at lmax = 2 * nside, the band limit
``hp2mwss.py`` uses, and zero-padded to the horizon's lmax, so the beam and
``horizon_mwss.npz`` share one MWSS grid.

Usage
-----
    uv run python eigsim/scripts/make_bowtie_v001.py [--src PATH]
"""

import argparse
import datetime
import hashlib
import os
import subprocess
import time
from pathlib import Path

# s2fft reads the flag when jax is first imported, so it must be set before;
# jax.config.update afterwards leaves s2fft in float32.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import s2fft  # noqa: E402
from scipy.constants import epsilon_0, mu_0  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_SRC = (
    "/home/christian/Documents/research/eigsep/data-analysis/"
    "hfss_beam_maps/bowtie_beam.npz"
)
# HEALPix forward-transform refinement, as for v000 (horizon_position/beams.py).
NITER = 3
FREQS_1MHZ = np.arange(50.0, 247.0)  # MHz; the source stops at 246.09 MHz


def _git(repo, *args):
    """Output of a git command in *repo*, or "unknown"."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def realized_efficiency(beam_cart_mv):
    """The |E|^2 realized gain integrated over the sphere, divided by 4 pi."""
    eta0 = np.sqrt(mu_0 / epsilon_0)
    e2 = (np.abs(beam_cart_mv * 1e-3) ** 2).sum(axis=1)  # V^2, 1 W incident
    return (e2 * 4 * np.pi / (2 * eta0)).mean(axis=1)


def healpix_to_padded_mwss(bm_hp, nside, lmax_out, niter=NITER):
    """Transform HEALPix maps at lmax = 2 * nside and resample onto MWSS.

    The alm are zero-padded to *lmax_out* before the inverse transform, so
    the result lives on the MWSS grid of *lmax_out* while carrying no
    structure beyond the source band limit.

    Returns
    -------
    bm_mwss : np.ndarray
        Shape ``(n_freqs, lmax_out + 2, 2 * lmax_out + 2)``.
    residual : np.ndarray
        Per channel, the largest HEALPix round-trip error at the source band
        limit, relative to the channel's peak.

    """
    L_in, L_out = 2 * nside + 1, lmax_out + 1
    if L_out < L_in:
        raise ValueError(f"lmax_out={lmax_out} is below the source lmax {L_in - 1}")
    maps, residual = [], []
    for row in np.asarray(bm_hp, dtype=np.float64):
        alm = s2fft.forward(
            jnp.asarray(row),
            L=L_in,
            spin=0,
            nside=nside,
            sampling="healpix",
            method="jax",
            reality=True,
            iter=niter,
        )
        back = s2fft.inverse(
            alm,
            L=L_in,
            spin=0,
            nside=nside,
            sampling="healpix",
            method="jax",
            reality=True,
        )
        residual.append(float(np.max(np.abs(np.asarray(back).real - row)) / row.max()))
        padded = jnp.zeros((L_out, 2 * L_out - 1), dtype=alm.dtype)
        padded = padded.at[:L_in, L_out - L_in : L_out + L_in - 1].set(alm)
        mwss = s2fft.inverse(
            padded, L=L_out, spin=0, sampling="mwss", method="jax", reality=True
        )
        maps.append(np.asarray(mwss).real)
    return np.stack(maps), np.array(residual)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", default=DEFAULT_SRC, help="HFSS bowtie_beam.npz")
    parser.add_argument("--niter", type=int, default=NITER)
    args = parser.parse_args()
    t0 = time.time()

    src = Path(args.src)
    d = np.load(src)
    beam_cart, freqs_mhz, nside = d["beam_cart"], d["freqs"], int(d["nside"])
    lmax = int(np.load(DATA_DIR / "horizon_mwss.npz")["lmax"])
    print(
        f"Source {src.name}: beam_cart {beam_cart.shape}, nside {nside}, "
        f"{freqs_mhz.size} channels {freqs_mhz[0]}-{freqs_mhz[-1]} MHz"
    )

    here = Path(__file__).resolve().parent
    status = _git(here, "status", "--porcelain")
    provenance = (
        f"source {src} (sha256 {hashlib.sha256(src.read_bytes()).hexdigest()}, "
        f"data-analysis commit "
        f"{_git(src.parent, 'log', '-1', '--format=%h', '--', src.name)}); "
        f"HFSS export by Dominic, pattern matching BK's 2025-10-31 bowtie-on-box "
        f"simulation; built {datetime.date.today().isoformat()} by "
        f"eigsim/scripts/make_bowtie_v001.py at mock_analysis "
        f"{_git(here, 'rev-parse', '--short', 'HEAD')}"
        + (" with uncommitted changes" if status not in ("", "unknown") else "")
    )
    recipe = (
        "bm = |Ex|^2 + |Ey|^2 + |Ez|^2 from beam_cart, each channel divided by "
        "its mean over the sphere (directivity: integral 4 pi). "
    )
    mwss_recipe = (
        f"HEALPix -> alm at lmax {2 * nside} (s2fft, niter {args.niter}), "
        f"zero-padded to lmax {lmax}, inverse onto MWSS. "
    )
    freqs_note = "freqs in Hz. realized_efficiency = |E|^2 realized gain over 4 pi."

    power = (np.abs(beam_cart) ** 2).sum(axis=1)
    bm_hp = power / power.mean(axis=1, keepdims=True)
    eff = realized_efficiency(beam_cart)
    np.savez(
        DATA_DIR / "eigsep_bowtie_v001.npz",
        freqs=freqs_mhz * 1e6,
        bm=bm_hp,
        nside=nside,
        realized_efficiency=eff,
        description=recipe + "HEALPix RING, native channels. " + freqs_note,
        provenance=provenance,
    )

    print(f"HEALPix -> MWSS (lmax {2 * nside} padded to {lmax})...", flush=True)
    bm_mwss, residual = healpix_to_padded_mwss(bm_hp, nside, lmax, args.niter)
    np.savez(
        DATA_DIR / "eigsep_bowtie_v001_mwss.npz",
        freqs=freqs_mhz * 1e6,
        bm=bm_mwss,
        lmax=lmax,
        realized_efficiency=eff,
        description=recipe + mwss_recipe + "Native channels. " + freqs_note,
        provenance=provenance,
    )

    bm_1mhz = CubicSpline(freqs_mhz, bm_mwss, axis=0)(FREQS_1MHZ)
    np.savez(
        DATA_DIR / "eigsep_bowtie_v001_1mhz_mwss.npz",
        freqs=FREQS_1MHZ * 1e6,
        bm=bm_1mhz,
        lmax=lmax,
        description=recipe
        + mwss_recipe
        + "Cubic spline (not-a-knot) in frequency from the native channels to "
        "50-246 MHz in 1 MHz steps; structure faster than the 3.906 MHz channel "
        "spacing is invented by the spline. freqs in Hz.",
        provenance=provenance,
    )

    w = np.asarray(s2fft.utils.quadrature_jax.quad_weights(L=lmax + 1, sampling="mwss"))
    integral = np.einsum("ftp,t->f", bm_mwss, w) / (4 * np.pi)
    print("Checks:")
    print(
        f"  HEALPix round trip at lmax {2 * nside}: max residual "
        f"{residual.max():.2e} of peak (channel {freqs_mhz[residual.argmax()]} MHz)"
    )
    print(f"  MWSS integral / 4 pi: {integral.min():.6f} to {integral.max():.6f}")
    print(
        f"  minimum bm: HEALPix {bm_hp.min():.3e}, MWSS {bm_mwss.min():.3e}, "
        f"1 MHz {bm_1mhz.min():.3e}"
    )
    for f in (50.78125, 101.5625, 148.4375, 199.21875, 246.09375):
        i = int(np.flatnonzero(freqs_mhz == f)[0])
        print(f"  realized efficiency at {f:9.5f} MHz: {eff[i]:.3f}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
