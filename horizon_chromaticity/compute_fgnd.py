"""Compute per-orientation ground fractions for each horizon case.

Loads the horizon masks built by ``make_horizons.py``, runs
``eigsim.compute_fgnd()`` over the same orientation grid as
``run_sims.py``, and saves ``output/fgnd_<case>.npz``.

Combined with ``eigsim.correct_ground_loss()``, this turns the
``t_sys`` arrays in ``chromaticity_<case>.npz`` into ground-loss
corrected sky temperatures:

    t_sky = (t_sys - t_rcvr - fgnd * Tgnd) / (1 - fgnd)

No sky model or time axis is involved — only the beam, the horizon
mask, and the drive rotations — so this is fast compared to the full
simulations.

Usage
-----
uv run python horizon_chromaticity/make_horizons.py   # once, first
uv run python horizon_chromaticity/compute_fgnd.py [--case eigsep]
"""

import argparse
import hashlib
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

import eigsim

CASES = ("nohorizon", "quarry", "eigsep")
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--case",
        choices=CASES,
        default=None,
        help="single horizon case (default: all cases)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cases = CASES if args.case is None else (args.case,)
    cfg = eigsim.load_config()

    horizons_file = OUTPUT_DIR / "horizons.npz"
    if not horizons_file.exists():
        raise SystemExit(f"{horizons_file} not found — run make_horizons.py first")
    hz = np.load(horizons_file)

    print("Loading beam...")
    beam_freqs_hz, beam_data, lmax = eigsim.load_beam()
    freqs_mhz = np.array(cfg["frequencies"], dtype=float)
    freq_idx = np.isin(beam_freqs_hz / 1e6, freqs_mhz)
    beam_data = beam_data[freq_idx]
    n_freqs = len(freqs_mhz)
    assert beam_data.shape[0] == n_freqs

    ori = cfg["orientations"]
    elev_vals = np.array(ori["elevations"], dtype=float)
    az_vals = np.array(ori["azimuths"], dtype=float)
    elev_grid, az_grid = np.meshgrid(elev_vals, az_vals, indexing="ij")
    elevations = elev_grid.ravel()
    azimuths = az_grid.ravel()
    n_ori = len(elevations)

    OUTPUT_DIR.mkdir(exist_ok=True)
    for case in cases:
        horizon = hz[case]
        mask_sha = hashlib.sha256(np.ascontiguousarray(horizon).tobytes()).hexdigest()

        print(f"Case '{case}': {n_ori} orientations x {n_freqs} freqs")
        t0 = time.time()
        fgnd = eigsim.compute_fgnd(
            beam_data,
            freqs_mhz,
            elevations,
            azimuths,
            beam_kw={"horizon": horizon},
            verbose=True,
        )
        print(f"  done in {time.time() - t0:.0f}s")

        outfile = OUTPUT_DIR / f"fgnd_{case}.npz"
        np.savez_compressed(
            outfile,
            fgnd=np.asarray(fgnd),  # (N_orientations, N_freqs)
            # Axes (same conventions as chromaticity_<case>.npz)
            freqs_mhz=freqs_mhz,
            elevations=elevations,  # flat, one per orientation
            azimuths=azimuths,  # flat, one per orientation
            elev_vals=elev_vals,  # grid axis values
            az_vals=az_vals,  # grid axis values
            # Metadata
            case=case,
            mask_sha=mask_sha,
            t_ground=cfg["ground"]["temperature"],
            t_receiver=cfg["receiver"]["temperature"],
            beam_file=cfg["beam"]["file"],
            beam_sampling=cfg["beam"]["sampling"],
            beam_lmax=lmax,
            horizon_file=cfg["horizon"]["file"],
            eigsim_version=eigsim.__version__,
        )
        print(f"  saved {outfile}")


if __name__ == "__main__":
    main()
