"""Compute horizon elevation profiles alpha_h(az) for the 19 positions.

Runs in the eigsep_terrain environment (it imports eigsep_terrain, which
is NOT available in the mock_analysis env):

    PYTHONPATH=/home/christian/Documents/research/eigsep/eigsep_terrain \
    uv run --project /home/christian/Documents/research/eigsep/eigsep_terrain \
        python horizon_position/make_horizons.py

Note: PYTHONPATH is required because eigsep_terrain uses a flat (non-src)
layout and is not installed as an editable package in its venv — uv only
adds the project dir to sys.path when running with -c, not a script path.

Output: output/horizons_position.npz with
  names     (19,)        position names
  enu       (19, 3)      antenna ENU positions [m]
  az_grid   (n_az,)      azimuths [rad], = atan2(E, N), North->East
  alpha_h   (19, n_az)   horizon elevation [rad] per position
  n_az      scalar
  pos_sha   hash of enu  (staleness guard for run_sims.py)
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

N_AZ = 720
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEM_CACHE = OUTPUT_DIR / "marjum_dem.npz"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("Building / loading Marjum DEM...")
    dem = MarjumDEM(cache_file=str(DEM_CACHE))

    positions = build_positions()
    names = [n for n, _ in positions]
    enu = np.array([e for _, e in positions], dtype=np.float64)
    az_grid = np.linspace(0.0, 2 * np.pi, N_AZ, endpoint=False)

    alpha_h = np.empty((len(positions), N_AZ), dtype=np.float64)
    for i, (name, e) in enumerate(positions):
        hangles, _ = dem.calc_horizon(float(e[0]), float(e[1]), float(e[2]), n_az=N_AZ)
        alpha_h[i] = np.asarray(hangles, dtype=np.float64)
        deg = np.degrees([hangles.min(), np.median(hangles), hangles.max()])
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
