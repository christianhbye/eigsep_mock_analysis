"""Run zenith-only t_sys and fgnd for each of the 19 antenna positions.

Loads the horizon curves from make_horizons.py, builds an anti-aliased
open-sky mask per position, and runs eigsim.simulate (zenith pointing,
N_ori=1) plus eigsim.compute_fgnd. Per-position batches are checkpointed
to output/pos<tag>_batch_*.npz and merged into output/position_sims.npz.

Usage (from the monorepo root):
    uv run python horizon_position/run_sims.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import croissant as cro
import numpy as np
from astropy import units as u
from astropy.time import Time
from pygdsm import GlobalSkyModel16

import eigsim

sys.path.insert(0, str(Path(__file__).resolve().parent))
from masks import mwss_grid, open_sky_weight  # noqa: E402

T_START = "2026-07-01 06:00:00"  # UTC, matches horizon_chromaticity
SIDEREAL_DAY_S = cro.constants.sidereal_day["earth"]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n-times", type=int, default=1436, help="time samples over one sidereal day"
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
        help="suffix for batch/output filenames (smoke tests only)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = eigsim.load_config()

    hz_file = OUTPUT_DIR / "horizons_position.npz"
    if not hz_file.exists():
        raise SystemExit(f"{hz_file} not found - run make_horizons.py first")
    hz = np.load(hz_file, allow_pickle=True)
    names = [str(n) for n in hz["names"]]
    alpha_h = hz["alpha_h"]
    az_grid = hz["az_grid"]
    enu = hz["enu"]
    pos_sha = str(hz["pos_sha"])
    n_pos = len(names)

    print("Loading beam...")
    beam_freqs_hz, beam_data, lmax = eigsim.load_beam()
    freqs_mhz = np.array(cfg["frequencies"], dtype=float)[:: args.freq_stride]
    freq_idx = np.isin(beam_freqs_hz / 1e6, freqs_mhz)
    beam_data = beam_data[freq_idx]
    n_freqs = len(freqs_mhz)
    assert beam_data.shape[0] == n_freqs
    thetas, phis = mwss_grid(lmax)

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

    print("Pre-computing sky ALM...")
    sky_alm = eigsim.precompute_sky_alm(sky)

    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Running {n_pos} positions x {args.n_times} times x {n_freqs} freqs...")
    wall0 = time.time()
    batch_files = []
    for i, name in enumerate(names):
        bf = OUTPUT_DIR / f"pos{args.output_tag}_batch_{i:02d}.npz"
        batch_files.append(bf)
        if bf.exists():
            npz = np.load(bf)
            if str(npz["pos_sha"]) != pos_sha:
                raise SystemExit(
                    f"{bf} was produced with different positions "
                    "(horizons_position.npz changed). Delete the stale "
                    f"pos{args.output_tag}_batch_*.npz files and rerun."
                )
            print(f"  [{i:2d}] {name:10s} found on disk, skipping")
            continue
        print(f"  [{i:2d}] {name:10s} simulating...")
        t0 = time.time()
        W = open_sky_weight(alpha_h[i], az_grid, thetas, phis)
        t_sys = eigsim.simulate(
            beam_data,
            freqs_mhz,
            sky,
            times_jd,
            [0.0],
            [0.0],
            beam_kw={"horizon": W},
            sky_alm=sky_alm,
        )  # (1, n_times, n_freqs)
        fgnd = eigsim.compute_fgnd(
            beam_data,
            freqs_mhz,
            [0.0],
            [0.0],
            beam_kw={"horizon": W},
        )  # (1, n_freqs)
        np.savez(
            bf,
            t_sys=np.asarray(t_sys)[0],
            fgnd=np.asarray(fgnd)[0],
            pos_sha=pos_sha,
        )
        print(f"       done in {time.time() - t0:.0f}s")

    print(f"All positions complete in {(time.time() - wall0) / 60:.1f} min")

    t_sys = np.stack([np.load(f)["t_sys"] for f in batch_files], axis=0)
    fgnd = np.stack([np.load(f)["fgnd"] for f in batch_files], axis=0)
    assert t_sys.shape == (n_pos, args.n_times, n_freqs)
    assert fgnd.shape == (n_pos, n_freqs)

    out = OUTPUT_DIR / f"position_sims{args.output_tag}.npz"
    np.savez_compressed(
        out,
        t_sys=t_sys,  # (n_pos, n_times, n_freqs)
        fgnd=fgnd,  # (n_pos, n_freqs)
        names=np.array(names),
        enu=enu,
        freqs_mhz=freqs_mhz,
        times_jd=times_jd,
        t_start=T_START,
        n_times=args.n_times,
        t_ground=cfg["ground"]["temperature"],
        t_receiver=cfg["receiver"]["temperature"],
        lon=cfg["location"]["lon"],
        lat=cfg["location"]["lat"],
        alt=cfg["location"]["alt"],
        sky_model=sky_cfg["model"],
        beam_lmax=lmax,
        pos_sha=pos_sha,
        eigsim_version=eigsim.__version__,
    )
    print(f"Saved {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    for f in batch_files:
        f.unlink(missing_ok=True)
    print("Batch files cleaned up.")


if __name__ == "__main__":
    main()
