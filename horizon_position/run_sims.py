"""Run zenith-only t_sys and fgnd for each of the 19 antenna positions.

Loads the horizon curves from make_horizons.py, builds an anti-aliased
open-sky mask per position with eigsim.open_sky_weight, and runs
eigsim.simulate (zenith pointing, N_ori=1) plus eigsim.compute_fgnd.
Results accumulate in memory and output/position_sims<tag>.npz is written
once at the end; an interrupted run has nothing to resume from and
restarts from scratch (~50 min: simulate and compute_fgnd recompile their
orientation graph on every call, ~2.5 min per position at lmax 128).

Usage (from the monorepo root):
    uv run python horizon_position/run_sims.py
"""

import argparse
import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_ENABLE_X64", "1")

import croissant as cro
import numpy as np
from astropy import units as u
from astropy.time import Time
from pygdsm import GlobalSkyModel16

import eigsim

T_START = "2026-07-01 06:00:00"  # UTC, matches horizon_chromaticity
SIDEREAL_DAY_S = cro.constants.sidereal_day["earth"]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
# The instrument paper's figures were made with the v000 beam on its 1 MHz
# grid; pinning the config keeps a change of eigsim's default out of them.
# Pass it to EVERY eigsim entry point, not just load_beam/load_config: the
# ground and receiver temperatures and the site are read from the config
# inside simulate/compute_fgnd, and this script writes the v000 values into
# the npz metadata. They agree with the current default today, so leaving
# the call sites on None was inert -- until eigsep.yaml moves, at which
# point the metadata would be a silent lie. make_sensitivity.py imports
# this constant and does the same.
EIGSIM_CONFIG = "eigsep_v000"


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
        help="suffix for the output filename (smoke tests only)",
    )
    return p.parse_args()


def load_inputs(n_times, freq_stride=1):
    """Build every input the simulation loop and the savez block need.

    Loads the horizon curves, the beam, the GSM16 sky and the time array,
    and pre-computes the sky ALM. ``alpha_h`` and ``az_grid`` are the
    native (46080-sample) horizon curves, unreduced -- see the "No azimuth
    reduction" comment in ``main``.

    Returns a ``SimpleNamespace`` with ``cfg``, ``names``, ``enu``,
    ``pos_sha``, ``alpha_h``, ``az_grid``, ``beam_data``, ``freqs_mhz``,
    ``lmax``, ``sky``, ``times_jd``, ``sky_alm``.
    """
    cfg = eigsim.load_config(EIGSIM_CONFIG)

    hz_file = OUTPUT_DIR / "horizons_position.npz"
    if not hz_file.exists():
        raise SystemExit(f"{hz_file} not found - run make_horizons.py first")
    hz = np.load(hz_file, allow_pickle=True)
    names = [str(n) for n in hz["names"]]
    alpha_h = hz["alpha_h"]
    az_grid = hz["az_grid"]
    enu = hz["enu"]
    pos_sha = str(hz["pos_sha"])

    print("Loading beam...")
    beam_freqs_hz, beam_data, lmax = eigsim.load_beam(config=EIGSIM_CONFIG)
    freqs_mhz = np.array(cfg["frequencies"], dtype=float)[::freq_stride]
    freq_idx = np.isin(beam_freqs_hz / 1e6, freqs_mhz)
    beam_data = beam_data[freq_idx]
    assert beam_data.shape[0] == len(freqs_mhz)

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
    times = cro.utils.time_array(t_start=t_start, t_end=t_end, N_times=n_times)
    times_jd = times.jd

    print("Pre-computing sky ALM...")
    sky_alm = eigsim.precompute_sky_alm(sky, times_jd, config=EIGSIM_CONFIG)

    return SimpleNamespace(
        cfg=cfg,
        names=names,
        enu=enu,
        pos_sha=pos_sha,
        alpha_h=alpha_h,
        az_grid=az_grid,
        beam_data=beam_data,
        freqs_mhz=freqs_mhz,
        lmax=lmax,
        sky=sky,
        times_jd=times_jd,
        sky_alm=sky_alm,
    )


def main():
    args = parse_args()
    inp = load_inputs(args.n_times, freq_stride=args.freq_stride)
    n_pos = len(inp.names)
    n_freqs = len(inp.freqs_mhz)

    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Running {n_pos} positions x {args.n_times} times x {n_freqs} freqs...")
    wall0 = time.time()
    t_sys_all, fgnd_all = [], []
    for i, name in enumerate(inp.names):
        print(f"  [{i:2d}] {name:10s} simulating...")
        t0 = time.time()
        # No azimuth reduction: eigsim.open_sky_weight integrates over the phi
        # cell, which *is* the band-limiting.  Reducing first would apply it
        # twice.  See eigsep_mock_analysis issue #10.
        W = eigsim.open_sky_weight(inp.alpha_h[i], inp.az_grid, inp.lmax)
        t_sys_all.append(
            np.asarray(
                eigsim.simulate(
                    inp.beam_data,
                    inp.freqs_mhz,
                    inp.sky,
                    inp.times_jd,
                    [0.0],
                    [0.0],
                    beam_kw={"horizon": W},
                    sky_alm=inp.sky_alm,
                    config=EIGSIM_CONFIG,
                )
            )[0]
        )
        fgnd_all.append(
            np.asarray(
                eigsim.compute_fgnd(
                    inp.beam_data,
                    inp.freqs_mhz,
                    [0.0],
                    [0.0],
                    beam_kw={"horizon": W},
                )
            )[0]
        )
        print(f"       done in {time.time() - t0:.0f}s")

    print(f"All positions complete in {(time.time() - wall0) / 60:.1f} min")

    t_sys = np.stack(t_sys_all, axis=0)
    fgnd = np.stack(fgnd_all, axis=0)
    assert t_sys.shape == (n_pos, args.n_times, n_freqs)
    assert fgnd.shape == (n_pos, n_freqs)

    out = OUTPUT_DIR / f"position_sims{args.output_tag}.npz"
    np.savez_compressed(
        out,
        t_sys=t_sys,  # (n_pos, n_times, n_freqs)
        fgnd=fgnd,  # (n_pos, n_freqs)
        names=np.array(inp.names),
        enu=inp.enu,
        freqs_mhz=inp.freqs_mhz,
        times_jd=inp.times_jd,
        t_start=T_START,
        n_times=args.n_times,
        t_ground=inp.cfg["ground"]["temperature"],
        t_receiver=inp.cfg["receiver"]["temperature"],
        lon=inp.cfg["location"]["lon"],
        lat=inp.cfg["location"]["lat"],
        alt=inp.cfg["location"]["alt"],
        sky_model=inp.cfg["sky"]["model"],
        beam_lmax=inp.lmax,
        pos_sha=inp.pos_sha,
        eigsim_version=eigsim.__version__,
    )
    print(f"Saved {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
