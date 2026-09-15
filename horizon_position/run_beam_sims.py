"""Run the nominal-horizon sidereal day once per antenna, for the beam comparison.

Same site, same sky, same horizon, same times as `run_sims.py`'s nominal
position -- only the beam changes. Three of them:

  bowtie     the EIGSEP antenna, from eigsim's packaged MWSS beam
  vivaldi    the HERA Phase II feed used in isolation (no dish), which is what
             the October 2024 suspension flew; HEALPix, resampled by beams.py
  isotropic  a uniform beam: the chromaticity-free reference, still behind the
             real horizon

Every input comes from `run_sims.load_inputs` and the open-sky weight is built
exactly as run_sims.py builds its nominal row, so the bowtie row reproduces
position_sims.npz's nominal row (test_smoke.py pins it). The npz is written
once at the end; an interrupted run restarts from scratch.

`--vivaldi` points at the HEALPix beam file. It is not in this repo and not in
eigsim; pass the path or set EIGSEP_VIVALDI_BEAM.

Usage (from the monorepo root):
    uv run python horizon_position/run_beam_sims.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402

import eigsim  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beams import (  # noqa: E402
    band_limited_power_fraction,
    healpix_to_mwss,
    isotropic_beam,
)
from run_sims import EIGSIM_CONFIG, OUTPUT_DIR, T_START, load_inputs  # noqa: E402

DEFAULT_VIVALDI = "/home/christian/Documents/research/eigsep/eigsep_vivaldi.npz"
TAGS = ("bowtie", "vivaldi", "isotropic")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-times", type=int, default=1436)
    p.add_argument(
        "--freq-stride",
        type=int,
        default=1,
        help="use every Nth config frequency (smoke tests only)",
    )
    p.add_argument(
        "--beams",
        nargs="+",
        choices=TAGS,
        default=list(TAGS),
        help="antennas to simulate, in output order",
    )
    p.add_argument(
        "--output-tag",
        default="",
        help="suffix for output/beam_sims<tag>.npz (smoke tests only)",
    )
    p.add_argument(
        "--vivaldi",
        default=os.environ.get("EIGSEP_VIVALDI_BEAM", DEFAULT_VIVALDI),
        help="HEALPix Vivaldi beam npz (keys: freqs, bm, nside)",
    )
    return p.parse_args()


def load_beams(wanted, vivaldi_path, inp):
    """The requested beams on the bowtie's MWSS grid and ``inp.freqs_mhz``.

    Only builds what is asked for: the Vivaldi resample costs minutes.
    Returns ``(beams, note)``; ``note`` records the band-limit check, which is
    what licenses comparing a directive feed against a broad one on a grid
    sized for the latter.
    """
    beams, note = {}, ""
    if "bowtie" in wanted:
        beams["bowtie"] = inp.beam_data
    if "isotropic" in wanted:
        beams["isotropic"] = isotropic_beam(inp.freqs_mhz.size, inp.beam_data.shape[1:])
    if "vivaldi" in wanted:
        vp = Path(vivaldi_path)
        if not vp.exists():
            raise SystemExit(
                f"{vp} not found -- pass --vivaldi or set EIGSEP_VIVALDI_BEAM"
            )
        viv = np.load(vp)
        viv_bm = viv["bm"][np.isin(viv["freqs"] / 1e6, inp.freqs_mhz)]
        if viv_bm.shape[0] != inp.freqs_mhz.size:
            raise SystemExit(
                f"vivaldi: {viv_bm.shape[0]} of {inp.freqs_mhz.size} config "
                "frequencies present in the beam file"
            )
        nside = int(viv["nside"])
        step = max(1, len(viv_bm) // 5)
        frac = band_limited_power_fraction(viv_bm[::step], nside, inp.lmax)
        note = (
            f"vivaldi band-limited power fraction at lmax={inp.lmax}: {frac.min():.9f}"
        )
        print(f"  {note}")
        if frac.min() < 1 - 1e-4:
            raise SystemExit(
                "the Vivaldi is not resolved at the bowtie's band limit; the "
                "comparison would measure the grid, not the antenna"
            )
        print(f"  resampling vivaldi HEALPix (nside={nside}) -> MWSS...", flush=True)
        beams["vivaldi"] = healpix_to_mwss(viv_bm, nside, inp.lmax)
    return beams, note


def main():
    args = parse_args()
    inp = load_inputs(args.n_times, freq_stride=args.freq_stride)
    i_nom = inp.names.index("nominal")

    print(f"Loading beams for {', '.join(args.beams)}...")
    beams, note = load_beams(args.beams, args.vivaldi, inp)
    # Exactly run_sims.py's nominal mask: the native, unreduced horizon curve.
    # open_sky_weight's phi-cell integral is the band-limiting; reducing the
    # curve first would apply it twice (issue #10).
    W = eigsim.open_sky_weight(inp.alpha_h[i_nom], inp.az_grid, inp.lmax)

    OUTPUT_DIR.mkdir(exist_ok=True)
    t_sys, fgnd = [], []
    for tag in args.beams:
        print(f"  {tag:10s} simulating...", flush=True)
        t0 = time.time()
        ts = eigsim.simulate(
            beams[tag],
            inp.freqs_mhz,
            inp.sky,
            inp.times_jd,
            [0.0],
            [0.0],
            beam_kw={"horizon": W},
            sky_alm=inp.sky_alm,
            config=EIGSIM_CONFIG,
        )
        fg = eigsim.compute_fgnd(
            beams[tag],
            inp.freqs_mhz,
            [0.0],
            [0.0],
            beam_kw={"horizon": W},
        )
        t_sys.append(np.asarray(ts)[0])
        fgnd.append(np.asarray(fg)[0])
        print(f"       done in {time.time() - t0:.0f}s")

    t_sys = np.stack(t_sys)
    fgnd = np.stack(fgnd)
    assert t_sys.shape == (len(args.beams), args.n_times, inp.freqs_mhz.size)

    out = OUTPUT_DIR / f"beam_sims{args.output_tag}.npz"
    np.savez_compressed(
        out,
        t_sys=t_sys,
        fgnd=fgnd,
        beams=np.array(args.beams),
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
        vivaldi_source=str(Path(args.vivaldi).name),
        band_limit_note=note,
        pos_sha=inp.pos_sha,
        eigsim_version=eigsim.__version__,
    )
    print(f"\nwrote {out}")
    for tag, f in zip(args.beams, fgnd):
        print(f"  {tag:10s} ground fraction {f.mean():.4f}  (eta {1 - f.mean():.4f})")


if __name__ == "__main__":
    main()
