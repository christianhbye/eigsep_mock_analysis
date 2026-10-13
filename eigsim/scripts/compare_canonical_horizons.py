"""How much the D5 horizon changes the canonical simulation.

Runs ``eigsim.simulate`` twice on the same beam, sky, times and a few
orientations: once with the legacy boolean mask of ``horizon_mwss.npz``
(site 1P, 114 m, grid azimuth) and once with the fractional D5 weight
from the team's horizon (``eigsim.d5_open_sky_weight``, 91 m era).  It
prints the change in system temperature by frequency, and the open-sky
fraction of each horizon.  For eigsep_mock_analysis issue #30 and
eigsep_analysis queue M8; it is a check, not a product.

Usage
-----
uv run python eigsim/scripts/compare_canonical_horizons.py
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import croissant as cro  # noqa: E402
import numpy as np  # noqa: E402
from astropy import units as u  # noqa: E402
from astropy.time import Time  # noqa: E402
from pygdsm import GlobalSkyModel16  # noqa: E402

import eigsim  # noqa: E402

N_TIMES = 48  # half-hour cadence over one sidereal day
ORIENTATIONS = [(0.0, 0.0), (30.0, 0.0), (30.0, 90.0), (60.0, 180.0)]


def open_fraction(W, lmax):
    thetas, _ = eigsim.mwss_grid(lmax)
    w = np.sin(thetas)[:, None]
    W = np.asarray(W, dtype=float)
    return float((w * W).sum() / (w * np.ones_like(W)).sum())


def main():
    cfg = eigsim.load_config()
    freqs_hz, beam, lmax = eigsim.load_beam()
    freqs_mhz = freqs_hz / 1e6
    legacy, _ = eigsim.load_horizon()
    legacy = np.isnan(legacy)
    d5, info = eigsim.d5_open_sky_weight(lmax, era="91m")
    d5 = np.asarray(d5)
    print(
        f"open-sky fraction: legacy {open_fraction(legacy, lmax):.4f}, "
        f"D5 91 m {open_fraction(d5, lmax):.4f} ({info['stamp']})"
    )

    gsm = GlobalSkyModel16(
        freq_unit="MHz",
        data_unit="TRJ",
        resolution=cfg["sky"]["resolution"],
        include_cmb=cfg["sky"]["include_cmb"],
    )
    sky = cro.Sky(
        gsm.generate(freqs_mhz), freqs_mhz, sampling="healpix", coord="galactic"
    )
    t0 = Time("2026-07-01 06:00:00", scale="utc")
    times = cro.utils.time_array(
        t_start=t0,
        t_end=t0 + cro.constants.sidereal_day["earth"] * u.s,
        N_times=N_TIMES,
    )
    sky_alm = eigsim.precompute_sky_alm(sky, times.jd)
    el = np.array([o[0] for o in ORIENTATIONS])
    az = np.array([o[1] for o in ORIENTATIONS])

    out = {}
    for name, h in (("legacy", legacy), ("d5", d5)):
        out[name] = np.asarray(
            eigsim.simulate(
                beam,
                freqs_mhz,
                sky,
                times.jd,
                el,
                az,
                beam_kw={"horizon": h},
                sky_alm=sky_alm,
            )
        )
    dT = out["d5"] - out["legacy"]  # (n_ori, n_times, n_freqs)
    rel = dT / out["legacy"]
    print("change D5 - legacy in T_sys, over orientations and times:")
    print(
        f"{'MHz':>8} {'median dT [K]':>14} {'max |dT| [K]':>13} "
        f"{'median dT/T':>12} {'max |dT/T|':>11}"
    )
    for k in range(0, freqs_mhz.size, 6):
        print(
            f"{freqs_mhz[k]:8.2f} {np.median(dT[..., k]):14.2f} "
            f"{np.abs(dT[..., k]).max():13.2f} {np.median(rel[..., k]):12.4f} "
            f"{np.abs(rel[..., k]).max():11.4f}"
        )
    for i, (e, a) in enumerate(ORIENTATIONS):
        print(
            f"  el {e:4.0f} az {a:5.0f}: median dT/T {np.median(rel[i]):+.4f}, "
            f"max |dT/T| {np.abs(rel[i]).max():.4f}"
        )


if __name__ == "__main__":
    main()
