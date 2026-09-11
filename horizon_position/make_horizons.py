"""Compute horizon elevation profiles alpha_h(az) for the 19 positions.

Runs in the eigsep_terrain environment (it imports eigsep_terrain, which
is NOT available in the mock_analysis env):

    PYTHONPATH=/home/christian/Documents/research/eigsep/eigsep_terrain \
    uv run --project /home/christian/Documents/research/eigsep/eigsep_terrain \
        python horizon_position/make_horizons.py

Note: PYTHONPATH is required because eigsep_terrain uses a flat (non-src)
layout and is not installed as an editable package in its venv — uv only
adds the project dir to sys.path when running with -c, not a script path.

`calc_horizon` assigns each DEM pixel to a whole number of azimuth bins and
takes the max per bin, so a coarse grid does not resolve how the horizon
responds to a horizontal antenna move: moving the antenna re-sorts which pixel
wins each bin (108 of 720 bins flip for a 0.1 m East shift, against 2 for
0.1 m Up) and each flip is a discontinuous jump. N_AZ is therefore set fine
enough to converge, and consumers reduce as they need.

Converged at N_AZ = 46080: refining to 92160 moves the +1 m curves by 6 per
cent, corr(+d,-d) for a +/-0.1 m pair reaches -0.989 (geometry: -1) and the
RMS ratio 1 m / 0.1 m reaches 9.94 (geometry: 10). At 720 those are -0.058 and
4.69. 46080 is 64 x masks.N_AZ_MASK so it reduces onto the mask grid exactly.

WHO REDUCES, AND WHY: `masks.open_sky_weight` point-samples the curve at the
~256 MWSS azimuths, so the simulations must band-limit it first --
`masks.reduce_azimuth`, which is the only place that reduction happens. The
figures plot alpha_h as it is stored here, because averaging hides 55-60 per
cent of the peak at cliff edges. Nothing derived is stored: the reduced curve
is a function call, not an array in this file.

Output: output/horizons_position.npz with
  names     (19,)        position names
  enu       (19, 3)      antenna ENU positions [m]
  az_grid   (n_az,)      azimuths [rad], = atan2(E, N), North->East
  alpha_h   (19, n_az)   horizon elevation [rad] per position, float64
  n_az      scalar
  pos_sha   hash of enu  (staleness guard for run_sims.py)

CAUTION: pos_sha covers the POSITIONS ONLY. It does not change when the DEM or
N_AZ changes, so run_sims.py's guard will silently accept stale
pos*_batch_*.npz files after either. Delete them by hand when you rerun this.
"""

import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
from eigsep_terrain.marjum_dem import MarjumDEM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from positions import build_positions  # noqa: E402

N_AZ = 46080  # = 64 * masks.N_AZ_MASK; see the module docstring
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEM_CACHE = OUTPUT_DIR / "marjum_dem.npz"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("Building / loading Marjum DEM...")
    dem = MarjumDEM(cache_file=str(DEM_CACHE))
    print(f"  DEM {dem.data.shape} {dem.data.dtype}, {float(dem.res)} m/px")

    positions = build_positions()
    names = [n for n, _ in positions]
    enu = np.array([e for _, e in positions], dtype=np.float64)
    az_grid = np.linspace(0.0, 2 * np.pi, N_AZ, endpoint=False)

    alpha_h = np.empty((len(positions), N_AZ), dtype=np.float64)
    for i, (name, e) in enumerate(positions):
        hangles, _ = dem.calc_horizon(float(e[0]), float(e[1]), float(e[2]), n_az=N_AZ)
        alpha_h[i] = np.asarray(hangles, dtype=np.float64)
        deg = np.degrees([alpha_h[i].min(), np.median(alpha_h[i]), alpha_h[i].max()])
        print(
            f"  [{i:2d}] {name:10s} alpha_h(min,med,max) deg = "
            f"{deg[0]:6.2f} {deg[1]:6.2f} {deg[2]:6.2f}"
        )

    pos_sha = hashlib.sha256(np.ascontiguousarray(enu).tobytes()).hexdigest()
    out = OUTPUT_DIR / "horizons_position.npz"
    np.savez(
        out,
        names=np.array(names),
        enu=enu,
        az_grid=az_grid,
        alpha_h=alpha_h,
        n_az=N_AZ,
        pos_sha=pos_sha,
    )
    print(f"Saved {out}  alpha_h shape {alpha_h.shape}")


if __name__ == "__main__":
    main()
