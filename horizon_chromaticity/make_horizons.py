"""Build the three horizon masks for the chromaticity comparison.

Cases (boolean masks on the MWSS grid, True = open sky):

- ``nohorizon``: theta = pi cut, all pixels open.
- ``quarry``: constant-theta cut at the MWSS ring boundary whose
  blocked solid angle best matches the EIGSEP horizon.
- ``eigsep``: realistic horizon from ``horizon_mwss.npz`` (the file
  stores distance-to-terrain; finite = blocked, NaN = open sky).

Saves ``output/horizons.npz`` and prints a summary.

Usage
-----
uv run python horizon_chromaticity/make_horizons.py
"""

import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import s2fft

import eigsim

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main():
    horizon_raw, lmax = eigsim.load_horizon()
    ntheta, nphi = horizon_raw.shape
    L = lmax + 1
    assert (ntheta, nphi) == (L + 1, 2 * L), "not MWSS sampling"

    # Per-ring quadrature weights; the per-pixel weight is w[ring].
    # Same weighting eigsim.simulate() uses internally.
    w = np.asarray(s2fft.utils.quadrature_jax.quad_weights(L=L, sampling="mwss"))
    theta = np.linspace(0.0, np.pi, ntheta)

    mask_eigsep = np.isnan(horizon_raw)  # True = open sky
    omega_target = (w[:, None] * ~mask_eigsep).sum()

    # Quarry: block whole rings i >= i_cut; pick the cut minimizing
    # |blocked solid angle - target|.
    ring_omega = nphi * w
    blocked = np.array([ring_omega[i:].sum() for i in range(ntheta + 1)])
    i_cut = int(np.argmin(np.abs(blocked - omega_target)))
    mask_quarry = np.zeros((ntheta, nphi), dtype=bool)
    mask_quarry[:i_cut] = True
    theta_c = theta[i_cut - 1] if i_cut > 0 else np.nan  # last open ring

    mask_nohorizon = np.ones((ntheta, nphi), dtype=bool)

    omega_blocked = {
        case: (w[:, None] * ~mask).sum()
        for case, mask in [
            ("nohorizon", mask_nohorizon),
            ("quarry", mask_quarry),
            ("eigsep", mask_eigsep),
        ]
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / "horizons.npz"
    np.savez(
        out,
        nohorizon=mask_nohorizon,
        quarry=mask_quarry,
        eigsep=mask_eigsep,
        lmax=lmax,
        i_cut=i_cut,
        theta_c_rad=theta_c,
        theta_c_deg=np.degrees(theta_c),
        omega_blocked_target=omega_target,
        omega_blocked_nohorizon=omega_blocked["nohorizon"],
        omega_blocked_quarry=omega_blocked["quarry"],
        omega_blocked_eigsep=omega_blocked["eigsep"],
    )

    print(f"Saved {out}")
    print(f"  quarry cut: ring {i_cut}, theta_c = {np.degrees(theta_c):.2f} deg")
    for case in ("nohorizon", "quarry", "eigsep"):
        om = omega_blocked[case]
        print(
            f"  {case:10s} blocked solid angle = {om:7.4f} sr "
            f"({om / (4 * np.pi):5.1%} of sphere)"
        )


if __name__ == "__main__":
    main()
