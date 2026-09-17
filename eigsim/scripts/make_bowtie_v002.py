#!/usr/bin/env python
"""Build the v002 EIGSEP bowtie beam from the native HFSS far field.

The source is Dominic's native HFSS export of the complex Cartesian far
field, stored as an exact MWSS grid by
``beam_models/hfss_native_sep2026/build_mwss.py``:
``bowtie_native_mwss_L180.npz`` holds ``E_mV`` (52, 3, 181, 360), rE in mV
at 1 W incident power, on theta = 0..180 deg and phi = 0..359 deg in 1 deg
steps -- exactly MWSS at L = 180 -- and ``freqs_mhz``, 50.78125-250.0 MHz
(k * 3.90625 MHz, k = 13..64).

Why v002 replaces v001. v001 was built from data-analysis
``hfss_beam_maps/bowtie_beam.npz``, the same export resampled onto HEALPix
nside 32, whose frequency labels are one 3.90625 MHz step low: its slice
labelled 46.875 MHz is the 50.78 MHz beam (verified at all 52 frequencies,
confirmed by Dominic 2026-09-16), and its 250 MHz beam was dropped. v001
therefore attached every beam to the wrong frequency, and carried HEALPix
resampling error. v002 uses the native grid, where the transform is exact
(eigsep_analysis MOD-62).

Recipe, per frequency:
- bm = |Ex|^2 + |Ey|^2 + |Ez|^2 on the native grid, divided by its
  quadrature integral over 4 pi (directivity: integral 4 pi);
- exact forward MWSS transform at L = 180, truncated to the horizon's
  lmax, inverse onto that MWSS grid, so the beam and ``horizon_mwss.npz``
  share one grid. The power pattern has no content near that lmax, and
  the script reports the power it drops.

Outputs, in eigsim/data (no HEALPix file, MOD-62):
- ``eigsep_bowtie_v002_mwss.npz``: native channels (config ``eigsep``).
- ``eigsep_bowtie_v002_1mhz_mwss.npz``: cubic spline in frequency to
  51-250 MHz in 1 MHz steps (config ``eigsep_1mhz``). Structure faster than
  the 3.906 MHz channel spacing is not in the source; the spline invents it.

Usage
-----
    uv run python eigsim/scripts/make_bowtie_v002.py [--src PATH]
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
DEFAULT_SRC = Path.home() / (
    "Documents/research/eigsep/beam_models/hfss_native_sep2026/"
    "bowtie_native_mwss_L180.npz"
)
FREQS_1MHZ = np.arange(51.0, 251.0)  # MHz; the source spans 50.78-250.0 MHz


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


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def native_to_mwss(power, L_in, lmax_out):
    """Exact MWSS transform at *L_in*, truncated to *lmax_out*, resynthesised.

    Returns
    -------
    bm : np.ndarray
        Shape ``(n_freqs, lmax_out + 2, 2 * lmax_out + 2)``.
    dropped : np.ndarray
        Per channel, the fraction of harmonic power above *lmax_out*.

    """
    L_out = lmax_out + 1
    if L_out > L_in:
        raise ValueError(f"lmax_out={lmax_out} exceeds the source lmax {L_in - 1}")
    maps, dropped = [], []
    for row in power:
        alm = s2fft.forward(
            jnp.asarray(row),
            L=L_in,
            spin=0,
            sampling="mwss",
            method="jax",
            reality=True,
        )
        alm = np.asarray(alm)
        c = L_in - 1
        kept = alm[:L_out, c - (L_out - 1) : c + L_out]
        total = np.sum(np.abs(alm) ** 2)
        dropped.append(float(1 - np.sum(np.abs(kept) ** 2) / total))
        out = s2fft.inverse(
            jnp.asarray(kept),
            L=L_out,
            spin=0,
            sampling="mwss",
            method="jax",
            reality=True,
        )
        maps.append(np.asarray(out).real)
    return np.stack(maps), np.array(dropped)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", default=str(DEFAULT_SRC), help="native MWSS npz")
    args = parser.parse_args()
    t0 = time.time()

    src = Path(args.src)
    d = np.load(src)
    E, freqs_mhz, L_in = d["E_mV"], d["freqs_mhz"], int(d["L"])
    assert str(d["sampling"]) == "mwss" and E.shape[2:] == (L_in + 1, 2 * L_in)
    lmax = int(np.load(DATA_DIR / "horizon_mwss.npz")["lmax"])
    print(
        f"Source {src.name}: E_mV {E.shape}, MWSS L {L_in}, "
        f"{freqs_mhz.size} channels {freqs_mhz[0]}-{freqs_mhz[-1]} MHz"
    )

    w_in = np.asarray(s2fft.utils.quadrature_jax.quad_weights(L=L_in, sampling="mwss"))
    e2 = (np.abs(E) ** 2).sum(axis=1)  # mV^2, (n_freqs, n_theta, n_phi)
    integral = np.einsum("ftp,t->f", e2, w_in)  # over 4 pi
    power = e2 * (4 * np.pi / integral)[:, None, None]
    eta0 = np.sqrt(mu_0 / epsilon_0)
    eff = integral * 1e-6 / (2 * eta0)  # radiated W per 1 W incident

    here = Path(__file__).resolve().parent
    status = _git(here, "status", "--porcelain")
    provenance = (
        f"source {src} (sha256 {_sha256(src)}); native HFSS export by Dominic, "
        f"shared 2026-09-16 (beam_models/hfss_native_sep2026, see its "
        f"PROVENANCE.json); built {datetime.date.today().isoformat()} by "
        f"eigsim/scripts/make_bowtie_v002.py at mock_analysis "
        f"{_git(here, 'rev-parse', '--short', 'HEAD')}"
        + (" with uncommitted changes" if status not in ("", "unknown") else "")
    )
    recipe = (
        "bm = |Ex|^2 + |Ey|^2 + |Ez|^2 from the native 1 deg MWSS export, each "
        "channel divided by its quadrature integral over the sphere "
        "(directivity: integral 4 pi). "
    )
    mwss_recipe = (
        f"Exact MWSS forward transform at L {L_in}, truncated to lmax {lmax}, "
        f"inverse onto the lmax {lmax} MWSS grid; no HEALPix. "
    )
    freqs_note = (
        "freqs in Hz: 50.78125-250.0 MHz, k * 3.90625 for k = 13..64. "
        "realized_efficiency = |E|^2 realized gain over 4 pi."
    )

    print(f"Native MWSS L {L_in} -> lmax {lmax}...", flush=True)
    bm, dropped = native_to_mwss(power, L_in, lmax)
    np.savez(
        DATA_DIR / "eigsep_bowtie_v002_mwss.npz",
        freqs=freqs_mhz * 1e6,
        bm=bm,
        lmax=lmax,
        realized_efficiency=eff,
        description=recipe + mwss_recipe + "Native channels. " + freqs_note,
        provenance=provenance,
    )

    bm_1mhz = CubicSpline(freqs_mhz, bm, axis=0)(FREQS_1MHZ)
    np.savez(
        DATA_DIR / "eigsep_bowtie_v002_1mhz_mwss.npz",
        freqs=FREQS_1MHZ * 1e6,
        bm=bm_1mhz,
        lmax=lmax,
        description=recipe
        + mwss_recipe
        + "Cubic spline (not-a-knot) in frequency from the native channels to "
        "51-250 MHz in 1 MHz steps; structure faster than the 3.906 MHz channel "
        "spacing is invented by the spline. freqs in Hz.",
        provenance=provenance,
    )

    w = np.asarray(s2fft.utils.quadrature_jax.quad_weights(L=lmax + 1, sampling="mwss"))
    check = np.einsum("ftp,t->f", bm, w) / (4 * np.pi)
    print("Checks:")
    print(f"  harmonic power above lmax {lmax}: max {dropped.max():.2e}")
    print(f"  MWSS integral / 4 pi: {check.min():.6f} to {check.max():.6f}")
    print(f"  minimum bm: native grid {bm.min():.3e}, 1 MHz {bm_1mhz.min():.3e}")
    for f in (50.78125, 101.5625, 148.4375, 199.21875, 250.0):
        i = int(np.flatnonzero(freqs_mhz == f)[0])
        print(f"  realized efficiency at {f:9.5f} MHz: {eff[i]:.3f}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
