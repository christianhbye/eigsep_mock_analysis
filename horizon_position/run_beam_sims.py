"""Run the nominal-horizon sidereal day once per antenna, for the beam comparison.

Same site, same sky, same horizon, same times as `run_sims.py`'s nominal
position -- only the beam changes. Three of them:

  bowtie     the EIGSEP antenna, from eigsim's packaged MWSS beam
  vivaldi    the HERA Phase II feed used in isolation (no dish), which is what
             the October 2024 suspension flew; HEALPix, resampled by beams.py
  isotropic  a uniform beam: the chromaticity-free reference, still behind the
             real horizon

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

import croissant as cro  # noqa: E402
import numpy as np  # noqa: E402
from astropy import units as u  # noqa: E402
from astropy.time import Time  # noqa: E402
from pygdsm import GlobalSkyModel16  # noqa: E402

import eigsim  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beams import (  # noqa: E402
    band_limited_power_fraction,
    healpix_to_mwss,
    isotropic_beam,
)
from masks import mwss_grid, open_sky_weight  # noqa: E402

T_START = "2026-07-01 06:00:00"  # UTC, identical to run_sims.py
SIDEREAL_DAY_S = cro.constants.sidereal_day["earth"]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_VIVALDI = "/home/christian/Documents/research/eigsep/eigsep_vivaldi.npz"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-times", type=int, default=1436)
    p.add_argument(
        "--vivaldi",
        default=os.environ.get("EIGSEP_VIVALDI_BEAM", DEFAULT_VIVALDI),
        help="HEALPix Vivaldi beam npz (keys: freqs, bm, nside)",
    )
    return p.parse_args()


TAGS = ("bowtie", "vivaldi", "isotropic")


def load_beams(wanted, vivaldi_path, freqs_mhz):
    """The requested beams on one MWSS grid and one frequency grid.

    Only builds what is asked for: the Vivaldi resample costs minutes, and a
    rerun with every checkpoint on disk should not pay it. Returns
    ``(beams, lmax, note)``; ``note`` records the band-limit check, which is
    what licenses comparing a directive feed against a broad one on a grid
    sized for the latter.
    """
    bow_freqs_hz, bow_bm, lmax = eigsim.load_beam()

    def on_grid(freqs_hz, bm, tag):
        idx = np.isin(freqs_hz / 1e6, freqs_mhz)
        out = bm[idx]
        if out.shape[0] != freqs_mhz.size:
            raise SystemExit(
                f"{tag}: {out.shape[0]} of {freqs_mhz.size} config frequencies "
                "present in the beam file"
            )
        return out

    beams, note = {}, ""
    if "bowtie" in wanted:
        beams["bowtie"] = on_grid(bow_freqs_hz, bow_bm, "bowtie")
    if "isotropic" in wanted:
        beams["isotropic"] = isotropic_beam(freqs_mhz.size, bow_bm.shape[1:])
    if "vivaldi" in wanted:
        vp = Path(vivaldi_path)
        if not vp.exists():
            raise SystemExit(
                f"{vp} not found -- pass --vivaldi or set EIGSEP_VIVALDI_BEAM"
            )
        viv = np.load(vp)
        viv_bm = on_grid(viv["freqs"], viv["bm"], "vivaldi")
        nside = int(viv["nside"])
        step = max(1, len(viv_bm) // 5)
        frac = band_limited_power_fraction(viv_bm[::step], nside, lmax)
        note = f"vivaldi band-limited power fraction at lmax={lmax}: {frac.min():.9f}"
        print(f"  {note}")
        if frac.min() < 1 - 1e-4:
            raise SystemExit(
                "the Vivaldi is not resolved at the bowtie's band limit; the "
                "comparison would measure the grid, not the antenna"
            )
        print(f"  resampling vivaldi HEALPix (nside={nside}) -> MWSS...", flush=True)
        beams["vivaldi"] = healpix_to_mwss(viv_bm, nside, lmax)
    return beams, lmax, note


def main():
    args = parse_args()
    cfg = eigsim.load_config()
    freqs_mhz = np.array(cfg["frequencies"], dtype=float)

    hz_file = OUTPUT_DIR / "horizons_position.npz"
    if not hz_file.exists():
        raise SystemExit(f"{hz_file} not found - run make_horizons.py first")
    hz = np.load(hz_file, allow_pickle=True)
    names = [str(n) for n in hz["names"]]
    i_nom = names.index("nominal")

    OUTPUT_DIR.mkdir(exist_ok=True)
    ckpt = {t: OUTPUT_DIR / f"beam_{t}.npz" for t in TAGS}
    missing = [t for t in TAGS if not ckpt[t].exists()]

    t_start = Time(T_START, scale="utc")
    times = cro.utils.time_array(
        t_start=t_start, t_end=t_start + SIDEREAL_DAY_S * u.s, N_times=args.n_times
    )
    note = ""
    if missing:
        print(f"Loading beams for {', '.join(missing)}...")
        beams, lmax, note = load_beams(missing, args.vivaldi, freqs_mhz)
        thetas, phis = mwss_grid(lmax)
        W = open_sky_weight(hz["alpha_h"][i_nom], hz["az_grid"], thetas, phis)

        print("Generating sky model (GSM16)...")
        gsm = GlobalSkyModel16(
            freq_unit="MHz",
            data_unit="TRJ",
            resolution=cfg["sky"]["resolution"],
            include_cmb=cfg["sky"]["include_cmb"],
        )
        sky = cro.Sky(
            gsm.generate(freqs_mhz), freqs_mhz, sampling="healpix", coord="galactic"
        )
        sky_alm = eigsim.precompute_sky_alm(sky)
        for tag in missing:
            print(f"  {tag:10s} simulating...", flush=True)
            t0 = time.time()
            ts = eigsim.simulate(
                beams[tag],
                freqs_mhz,
                sky,
                times.jd,
                [0.0],
                [0.0],
                beam_kw={"horizon": W},
                sky_alm=sky_alm,
            )
            fg = eigsim.compute_fgnd(
                beams[tag], freqs_mhz, [0.0], [0.0], beam_kw={"horizon": W}
            )
            np.savez(ckpt[tag], t_sys=np.asarray(ts)[0], fgnd=np.asarray(fg)[0])
            print(f"       done in {time.time() - t0:.0f}s -> {ckpt[tag].name}")
    else:
        print("all per-beam checkpoints on disk; merging only")
        lmax = int(eigsim.load_beam()[2])

    tags = list(TAGS)
    t_sys = np.stack([np.load(ckpt[t])["t_sys"] for t in tags])
    fgnd = np.stack([np.load(ckpt[t])["fgnd"] for t in tags])

    assert t_sys.shape == (len(tags), args.n_times, freqs_mhz.size)

    out = OUTPUT_DIR / "beam_sims.npz"
    np.savez_compressed(
        out,
        t_sys=t_sys,
        fgnd=fgnd,
        beams=np.array(tags),
        freqs_mhz=freqs_mhz,
        times_jd=times.jd,
        t_start=T_START,
        n_times=args.n_times,
        t_ground=cfg["ground"]["temperature"],
        t_receiver=cfg["receiver"]["temperature"],
        lon=cfg["location"]["lon"],
        lat=cfg["location"]["lat"],
        alt=cfg["location"]["alt"],
        sky_model=cfg["sky"]["model"],
        beam_lmax=lmax,
        vivaldi_source=str(Path(args.vivaldi).name),
        band_limit_note=note,
        eigsim_version=eigsim.__version__,
    )
    print(f"\nwrote {out}")
    for tag, f in zip(tags, fgnd):
        print(f"  {tag:10s} ground fraction {f.mean():.4f}  (eta {1 - f.mean():.4f})")


if __name__ == "__main__":
    main()
