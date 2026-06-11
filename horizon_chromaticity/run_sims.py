"""Run a noiseless EIGSEP chromaticity simulation for one horizon case.

Loads the horizon mask built by ``make_horizons.py``, runs
``eigsim.simulate()`` over the canonical orientation grid for one
sidereal day, and saves raw noiseless ``t_sys`` to
``output/chromaticity_<case>.npz``.

Orientations are processed in batches; completed batches on disk
(``output/<case>_batch_*.npz``) are reused on the next run, so the
script is safe to interrupt and rerun.

Usage
-----
uv run python horizon_chromaticity/make_horizons.py   # once, first
uv run python horizon_chromaticity/run_sims.py --case eigsep
uv run python horizon_chromaticity/run_sims.py --case flat --zenith-only
"""

import argparse
import hashlib
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import croissant as cro
import numpy as np
from astropy import units as u
from astropy.time import Time
from pygdsm import GlobalSkyModel16

import eigsim

CASES = ("nohorizon", "quarry", "eigsep", "flat")
T_START = "2026-07-01 06:00:00"  # UTC (July 1 2026 00:00 Mountain Time)
SIDEREAL_DAY_S = cro.constants.sidereal_day["earth"]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--case", required=True, choices=CASES)
    p.add_argument(
        "--zenith-only",
        action="store_true",
        help="simulate only the zenith pointing (elevation 0, azimuth 0)",
    )
    p.add_argument(
        "--n-times",
        type=int,
        default=1436,
        help="time samples over one sidereal day (default ~1 min cadence)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="orientations per checkpoint batch",
    )
    p.add_argument(
        "--max-orientations",
        type=int,
        default=None,
        help="truncate the orientation grid (smoke tests only)",
    )
    p.add_argument(
        "--freq-stride",
        type=int,
        default=1,
        help="use every Nth config frequency (smoke tests only)",
    )
    p.add_argument(
        "--output-tag",
        default="",
        help="suffix for output/batch filenames (smoke tests only)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = eigsim.load_config()

    horizons_file = OUTPUT_DIR / "horizons.npz"
    if not horizons_file.exists():
        raise SystemExit(f"{horizons_file} not found — run make_horizons.py first")
    hz = np.load(horizons_file)
    horizon = hz[args.case]
    mask_sha = hashlib.sha256(np.ascontiguousarray(horizon).tobytes()).hexdigest()
    if args.case == "quarry":
        theta_c_deg = float(hz["theta_c_deg"])
    elif args.case == "flat":
        theta_c_deg = float(hz["theta_c_flat_deg"])
    else:
        theta_c_deg = np.nan
    omega_blocked = float(hz[f"omega_blocked_{args.case}"])

    print("Loading beam...")
    beam_freqs_hz, beam_data, lmax = eigsim.load_beam()
    freqs_mhz = np.array(cfg["frequencies"], dtype=float)[:: args.freq_stride]
    freq_idx = np.isin(beam_freqs_hz / 1e6, freqs_mhz)
    beam_data = beam_data[freq_idx]
    n_freqs = len(freqs_mhz)
    assert beam_data.shape[0] == n_freqs
    print(f"  Selected {n_freqs}/{len(beam_freqs_hz)} beam channels")

    print("Generating sky model (GSM16)...")
    sky_cfg = cfg["sky"]
    gsm = GlobalSkyModel16(
        freq_unit="MHz",
        data_unit="TRJ",
        resolution=sky_cfg["resolution"],
        include_cmb=sky_cfg["include_cmb"],
    )
    sky_map = gsm.generate(freqs_mhz)
    sky = cro.Sky(sky_map, freqs_mhz, sampling="healpix", coord="galactic")

    print("Building time array...")
    t_start = Time(T_START, scale="utc")
    t_end = t_start + SIDEREAL_DAY_S * u.s
    times = cro.utils.time_array(t_start=t_start, t_end=t_end, N_times=args.n_times)
    times_jd = times.jd

    ori = cfg["orientations"]
    if args.zenith_only:
        elev_vals = np.array([0.0])
        az_vals = np.array([0.0])
    else:
        elev_vals = np.array(ori["elevations"], dtype=float)
        az_vals = np.array(ori["azimuths"], dtype=float)
    elev_grid, az_grid = np.meshgrid(elev_vals, az_vals, indexing="ij")
    elevations = elev_grid.ravel()
    azimuths = az_grid.ravel()
    if args.max_orientations is not None:
        elevations = elevations[: args.max_orientations]
        azimuths = azimuths[: args.max_orientations]
    n_ori = len(elevations)
    print(f"Case '{args.case}': {n_ori} orientations")

    print("Pre-computing sky ALM...")
    sky_alm = eigsim.precompute_sky_alm(sky)

    OUTPUT_DIR.mkdir(exist_ok=True)
    n_batches = int(np.ceil(n_ori / args.batch_size))
    print(
        f"Running simulation ({n_ori} orientations x {args.n_times} times "
        f"x {n_freqs} freqs) in {n_batches} batches of {args.batch_size}..."
    )

    wall_start = time.time()
    batch_files = []
    for b in range(n_batches):
        i0 = b * args.batch_size
        i1 = min(i0 + args.batch_size, n_ori)
        batch_file = OUTPUT_DIR / f"{args.case}{args.output_tag}_batch_{b:04d}.npz"
        batch_files.append(batch_file)

        if batch_file.exists():
            npz = np.load(batch_file)
            if "mask_sha" not in npz or str(npz["mask_sha"]) != mask_sha:
                raise SystemExit(
                    f"{batch_file} was produced with a different horizon mask "
                    "(horizons.npz changed since). Delete the stale "
                    f"{args.case}{args.output_tag}_batch_*.npz files and rerun."
                )
            print(f"  Batch {b + 1}/{n_batches} [{i0}:{i1}] — found on disk, skipping")
            continue

        print(f"  Batch {b + 1}/{n_batches} [{i0}:{i1}]")
        t0 = time.time()
        t_sys = eigsim.simulate(
            beam_data,
            freqs_mhz,
            sky,
            times_jd,
            elevations[i0:i1],
            azimuths[i0:i1],
            beam_kw={"horizon": horizon},
            sky_alm=sky_alm,
            verbose=True,
        )
        np.savez(batch_file, t_sys=np.asarray(t_sys), mask_sha=mask_sha)
        print(f"  Batch {b + 1}/{n_batches} done in {time.time() - t0:.0f}s")

    print(f"All batches complete in {(time.time() - wall_start) / 3600:.1f} h")

    print("Merging batches...")
    t_sys = np.concatenate([np.load(f)["t_sys"] for f in batch_files], axis=0)
    assert t_sys.shape == (n_ori, args.n_times, n_freqs)

    outfile = OUTPUT_DIR / f"chromaticity_{args.case}{args.output_tag}.npz"
    print(f"Saving to {outfile}...")
    np.savez_compressed(
        outfile,
        # Simulation output (noiseless system temperature)
        t_sys=t_sys,
        # Axes
        freqs_mhz=freqs_mhz,
        times_jd=times_jd,
        elevations=elevations,  # flat, one per orientation
        azimuths=azimuths,  # flat, one per orientation
        elev_vals=elev_vals,  # grid axis values
        az_vals=az_vals,  # grid axis values
        # Horizon metadata
        case=args.case,
        theta_c_deg=theta_c_deg,
        omega_blocked=omega_blocked,
        # Config / metadata
        t_start=T_START,
        n_times=args.n_times,
        lon=cfg["location"]["lon"],
        lat=cfg["location"]["lat"],
        alt=cfg["location"]["alt"],
        world=cfg["world"],
        t_ground=cfg["ground"]["temperature"],
        t_receiver=cfg["receiver"]["temperature"],
        sky_model=sky_cfg["model"],
        sky_resolution=sky_cfg["resolution"],
        sky_include_cmb=sky_cfg["include_cmb"],
        beam_file=cfg["beam"]["file"],
        beam_sampling=cfg["beam"]["sampling"],
        beam_lmax=lmax,
        horizon_file=cfg["horizon"]["file"],
        mask_sha=mask_sha,
        eigsim_version=eigsim.__version__,
    )
    print(f"Done. Output size: {outfile.stat().st_size / 1e6:.0f} MB")

    for f in batch_files:
        f.unlink(missing_ok=True)
    print("Batch files cleaned up.")


if __name__ == "__main__":
    main()
