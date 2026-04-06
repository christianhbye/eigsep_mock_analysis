"""Run a canonical EIGSEP simulation for one sidereal day.

Loads beam, horizon, and sky model from the default config, runs the
simulation over the full orientation grid, adds radiometer noise, and
saves the result to ``output/canonical_sim.npz``.

Orientations are processed in batches to keep memory usage bounded.
Intermediate batches are saved to ``output/batch_*.npz`` and merged
at the end.  If the script is interrupted, completed batches on disk
are reused on the next run.

Usage
-----
uv run python scripts/run_canonical_sim.py
"""

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

cfg = eigsim.load_config()

# Observation start: July 1, 2026 00:00 local time (Mountain Time, UTC-6)
T_START = "2026-07-01 06:00:00"  # UTC
SIDEREAL_DAY_S = cro.constants.sidereal_day["earth"]
N_TIMES = 1436  # ~1 minute cadence
RNG_SEED = 20260701
BATCH_SIZE = 100  # orientations per batch
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# ---------------------------------------------------------------------------
# Load beam and horizon
# ---------------------------------------------------------------------------
print("Loading beam and horizon data...")
beam_freqs_hz, beam_data, lmax = eigsim.load_beam()
horizon, _ = eigsim.load_horizon()

# Config frequencies are the source of truth — select matching channels.
freqs_mhz = np.array(cfg["frequencies"])
freqs_hz = freqs_mhz * 1e6
beam_freqs_mhz = beam_freqs_hz / 1e6
freq_idx = np.isin(beam_freqs_mhz, freqs_mhz)
beam_data = beam_data[freq_idx]
print(f"  Selected {beam_data.shape[0]}/{len(beam_freqs_mhz)} beam channels")

# ---------------------------------------------------------------------------
# Generate sky model
# ---------------------------------------------------------------------------
print("Generating sky model (GSM16)...")
sky_cfg = cfg["sky"]
gsm = GlobalSkyModel16(
    freq_unit="MHz",
    data_unit="TRJ",
    resolution=sky_cfg["resolution"],
    include_cmb=sky_cfg["include_cmb"],
)
sky_map = gsm.generate(freqs_mhz)  # (N_freqs, N_pix) healpix
sky = cro.Sky(sky_map, freqs_mhz, sampling="healpix", coord="galactic")

# ---------------------------------------------------------------------------
# Build time array
# ---------------------------------------------------------------------------
print("Building time array...")
t_start = Time(T_START, scale="utc")
t_end = t_start + SIDEREAL_DAY_S * u.s
times = cro.utils.time_array(t_start=t_start, t_end=t_end, N_times=N_TIMES)
times_jd = times.jd

# ---------------------------------------------------------------------------
# Build orientation grid
# ---------------------------------------------------------------------------
ori = cfg["orientations"]
elev_vals = np.array(ori["elevations"], dtype=float)
az_vals = np.array(ori["azimuths"], dtype=float)
elev_grid, az_grid = np.meshgrid(elev_vals, az_vals, indexing="ij")
elevations = elev_grid.ravel()
azimuths = az_grid.ravel()
n_ori = len(elevations)
print(f"Orientation grid: {len(elev_vals)} x {len(az_vals)} = {n_ori} orientations")

# ---------------------------------------------------------------------------
# Noise parameters
# ---------------------------------------------------------------------------
delta_freq_hz = np.median(np.diff(freqs_hz))
delta_time_s = SIDEREAL_DAY_S / N_TIMES
rng = np.random.default_rng(RNG_SEED)

# ---------------------------------------------------------------------------
# Run simulation in batches
# ---------------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
n_batches = int(np.ceil(n_ori / BATCH_SIZE))
print(
    f"Running simulation "
    f"({n_ori} orientations x {N_TIMES} times x {len(freqs_mhz)} freqs) "
    f"in {n_batches} batches of {BATCH_SIZE}..."
)

wall_start = time.time()
batch_files = []

for b in range(n_batches):
    i0 = b * BATCH_SIZE
    i1 = min(i0 + BATCH_SIZE, n_ori)
    batch_file = OUTPUT_DIR / f"batch_{b:04d}.npz"
    batch_files.append(batch_file)

    if batch_file.exists():
        # Advance the RNG state to stay consistent with a fresh run.
        rng.normal(size=(i1 - i0, N_TIMES, len(freqs_mhz)))
        print(f"  Batch {b + 1}/{n_batches} [{i0}:{i1}] — found on disk, skipping")
        continue

    print(f"  Batch {b + 1}/{n_batches} [{i0}:{i1}] ...", end=" ", flush=True)
    t0 = time.time()

    t_sys = eigsim.simulate(
        beam_data,
        freqs_mhz,
        sky,
        times_jd,
        elevations[i0:i1],
        azimuths[i0:i1],
        beam_kw={"horizon": horizon},
    )
    t_sys = np.asarray(t_sys)
    noise = eigsim.radiometer_noise(t_sys, delta_freq_hz, delta_time_s, rng=rng)
    t_obs_batch = t_sys + noise

    np.savez(batch_file, t_obs=t_obs_batch)
    dt = time.time() - t0
    print(f"done in {dt:.0f}s")

wall_elapsed = time.time() - wall_start
print(f"All batches complete in {wall_elapsed / 3600:.1f} hours")

# ---------------------------------------------------------------------------
# Merge batches into final output
# ---------------------------------------------------------------------------
print("Merging batches...")
t_obs = np.concatenate([np.load(f)["t_obs"] for f in batch_files], axis=0)
assert t_obs.shape == (n_ori, N_TIMES, len(freqs_mhz))

outfile = OUTPUT_DIR / "canonical_sim.npz"
print(f"Saving to {outfile}...")
np.savez_compressed(
    outfile,
    # Simulation output
    t_obs=t_obs,
    # Axes
    freqs_mhz=freqs_mhz,
    times_jd=times_jd,
    elevations=elev_vals,
    azimuths=az_vals,
    # Config / metadata
    t_start=T_START,
    n_times=N_TIMES,
    rng_seed=RNG_SEED,
    delta_freq_hz=delta_freq_hz,
    delta_time_s=delta_time_s,
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
)
size_mb = outfile.stat().st_size / 1e6
print(f"Done. Output size: {size_mb:.0f} MB")

# Clean up batch files
for f in batch_files:
    f.unlink(missing_ok=True)
print("Batch files cleaned up.")
