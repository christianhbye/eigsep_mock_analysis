#!/usr/bin/env python
"""Build eigsim's D5 horizon from the team's ``horizon_profiles_v0005``.

The source is the EIGSEP team's curated horizon for the Marjum D5
campaign, ``marjum-2026-07/curation/horizon_profiles_v0005.npz`` with its
``.json`` sidecar (generator ``data-analysis/scripts/marjum-2026-07/
build_horizon_profiles.py`` @ ``9a06df5``, 2026-10-21; eigsep-hub branch
``horizon-v0005-pointing-v4`` @ ``c1fa926``). It is a ray march from the
antenna of geometry release v0005 on DEM v0004 (v0003 extended 6 km
south), on 3600 bearings 0.1 deg apart, for four height eras: 30 m
(field-note difference), 87.5 m (S11 terrain echo, 1.659 m below the
91 m anchor), 91 m (the release's antenna) and 91m-reraise (S11, 0.572 m
above the 91 m era: the configuration after the 07-17 re-raise).
It supersedes v0003 (geometry v0004, DEM v0003, field-note heights) as
eigsep_analysis's D5 horizon (MOD-95).

Its bearings are UTM *grid* bearings, counter-clockwise from grid East
(``atan2(dN, dE)``). eigsim's ``open_sky_weight`` wants compass azimuth
``az = atan2(E, N)`` in the true frame, ascending on ``[0, 2 pi)``, so each
curve is converted with ``A_true = 90 deg - theta_grid - 1.5208 deg``
(``eigsim.horizon.curve_from_grid_bearings``; grid convergence from
eigsep_analysis issue #8) and sorted. Elevations are unchanged, in rad.

Output, in eigsim/data: ``horizon_d5_v0005.npz`` with ``az`` (rad),
``alpha_h_30m``, ``alpha_h_87p5m``, ``alpha_h_91m``,
``alpha_h_91m_reraise`` (rad), ``stamp`` (the team's version stamp),
``source_sha256``, ``convergence_deg``, ``description`` and
``provenance``. Read it with
``eigsim.data.load_horizon_curve(era, path=...)``.

The source must match the pinned sha256 (the file published 2026-10-21);
a newer team version gets its own script and file.

Usage
-----
    uv run python eigsim/scripts/make_horizon_d5_v0005.py --src PATH
"""

import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from eigsim.horizon import GRID_CONVERGENCE_DEG, curve_from_grid_bearings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_NAME = "horizon_d5_v0005.npz"
#: sha256 of the team's horizon_profiles_v0005.npz as published 2026-10-21.
SOURCE_SHA256 = "b497cebd1324291af7c7abc58266edb57a4db198add35392d89fb3e69049c2e9"
#: source key -> output key, per height era
ERAS = {
    "elev_rad_30m": "alpha_h_30m",
    "elev_rad_87.5m": "alpha_h_87p5m",
    "elev_rad_91m": "alpha_h_91m",
    "elev_rad_91m-reraise": "alpha_h_91m_reraise",
}


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


def build(src, out, expect_sha256=SOURCE_SHA256):
    """Convert the team file *src* into eigsim's format at *out*."""
    src, out = Path(src), Path(out)
    sha = _sha256(src)
    if expect_sha256 is not None and sha != expect_sha256:
        raise ValueError(
            f"source sha256 {sha} is not the pinned v0005 file {expect_sha256}"
        )
    meta = json.loads(src.with_suffix(".json").read_text())
    if "atan2(dN,dE)" not in meta.get("frame", ""):
        frame = meta.get("frame")
        raise ValueError(f"unexpected bearing frame in the sidecar: {frame!r}")
    d = np.load(src)
    bearings = d["bearings_deg"]
    arrays = {}
    for key_in, key_out in ERAS.items():
        az, alpha = curve_from_grid_bearings(bearings, d[key_in])
        arrays[key_out] = alpha
    here = Path(__file__).resolve().parent
    status = _git(here, "status", "--porcelain")
    provenance = (
        f"source {src} (sha256 {sha}; team stamp {meta['stamp']}); built "
        f"{datetime.date.today().isoformat()} by "
        f"eigsim/scripts/make_horizon_d5_v0005.py at mock_analysis "
        f"{_git(here, 'rev-parse', '--short', 'HEAD')}"
        + (" with uncommitted changes" if status not in ("", "unknown") else "")
    )
    description = (
        "The EIGSEP team's D5 horizon (curation/horizon_profiles_v0005: ray march "
        "from the geometry-v0005 antenna on DEM v0004; eras 30 m (field-note "
        "difference), 87.5 m and 91m-reraise (S11 terrain-echo offsets from the "
        "91 m anchor), 91 m), "
        "converted from UTM grid bearings (CCW from grid East) to true compass "
        f"azimuth az = atan2(E, N) with grid convergence {GRID_CONVERGENCE_DEG} "
        "deg (A_true = 90 - theta_grid - convergence), sorted ascending on "
        "[0, 2 pi). az and alpha_h_* in rad. For eigsim.horizon.open_sky_weight."
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        az=az,
        **arrays,
        stamp=np.array(meta["stamp"]),
        source_sha256=np.array(sha),
        convergence_deg=np.array(GRID_CONVERGENCE_DEG),
        description=np.array(description),
        provenance=np.array(provenance),
    )
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--src",
        required=True,
        help="the team's horizon_profiles_v0005.npz (its .json beside it)",
    )
    parser.add_argument("--out", default=str(DATA_DIR / OUT_NAME))
    args = parser.parse_args()
    out = build(args.src, args.out)
    d = np.load(out)
    print(f"Wrote {out}")
    for key in ERAS.values():
        a = np.degrees(d[key])
        print(f"  {key}: {a.size} az, elevation {a.min():.2f} to {a.max():.2f} deg")


if __name__ == "__main__":
    main()
