# Horizon Chromaticity Comparison — Design

**Date:** 2026-06-10
**Status:** Approved

## Goal

Quantify, for a paper, how the horizon model affects the spectral
chromaticity of simulated EIGSEP spectra. Three horizon models are
compared with otherwise identical simulations. This is an analysis
project that *uses* the `eigsim` package but does not ship with it.

## Simulation parameters

- **Orientations:** full canonical grid from the eigsim config
  (elevations 0–350° × azimuths 0–350° in 10° steps = 1296).
- **Times:** one sidereal day, 1436 samples (~1 min cadence),
  start 2026-07-01 06:00:00 UTC (same as canonical sim).
- **Frequencies:** 50–250 MHz, 1 MHz steps (201 channels, from config).
- **Sky:** GSM16 (TRJ, lo resolution, CMB included, from config).
- **Noise:** none. Scripts save raw noiseless `t_sys`. Noise can be
  added later in notebooks if needed.
- **Chromaticity metrics:** computed in notebooks, not in scripts.
  Scripts only produce raw spectra.

## Horizon cases

All cases are boolean masks on the MWSS grid `(N_theta, N_phi)`,
**True = open sky** (the convention `croissant.Beam(horizon=...)`
expects).

1. **`nohorizon`** — θ = π cut: all pixels open, ground fraction 0.
   Azimuthally symmetric baseline.
2. **`quarry`** — constant-θ cut at θ_c: open for θ ≤ θ_c, blocked
   below. θ_c is solved so that the **blocked solid angle** matches
   the EIGSEP horizon (case 3). Solid angle, not raw pixel count,
   because MWSS rings have equal pixel counts but unequal areas.
   Since a constant-θ cut on MWSS blocks whole rings, θ_c is the ring
   boundary minimizing |ΔΩ|; both target and achieved blocked solid
   angle are recorded in the output.
3. **`eigsep`** — realistic horizon: `np.isnan(horizon)` of the array
   in `eigsim/data/horizon_mwss.npz` (file stores distance-to-terrain,
   finite = blocked, NaN = open sky).

Solid angles are computed with the same s2fft MWSS quadrature weights
used inside `eigsim.simulate()`.

## Project layout

Self-contained top-level folder in the monorepo (not a package):

```
horizon_chromaticity/
  README.md            # purpose, how to run, output format
  make_horizons.py     # builds the 3 masks, solves quarry θ_c,
                       # saves output/horizons.npz
  run_sims.py          # CLI: --case {nohorizon,quarry,eigsep}
  output/              # gitignored npz outputs
  notebooks/           # analysis notebooks (chromaticity metrics)
```

Notebooks live inside the project folder (deviation from the
`notebooks/` top-level convention, chosen for paper cohesion) and load
only the npz outputs — no imports from local script paths.

## Scripts

### `make_horizons.py`

Loads `horizon_mwss.npz` via `eigsim.load_horizon()`, builds the three
boolean masks, solves θ_c for the quarry case, and saves
`output/horizons.npz` containing the three masks plus metadata
(θ_c in degrees and radians, target and achieved blocked solid angle
per case). Prints a summary table.

### `run_sims.py`

Modeled on `eigsim/scripts/run_canonical_sim.py`:

- `--case {nohorizon,quarry,eigsep}` selects the mask from
  `output/horizons.npz` (run `make_horizons.py` first).
- Loads beam, builds GSM16 sky and time array, precomputes sky ALM
  once, runs `eigsim.simulate()` in orientation batches with
  checkpoint/resume (`output/<case>_batch_*.npz`), merges to
  `output/chromaticity_<case>.npz`, then deletes batch files.
- No noise step.
- One invocation per case so runs resume independently. Expected cost:
  roughly one canonical-sim runtime and a ~GB-scale output per case.

Output npz per case: `t_sys (N_ori, N_times, N_freqs)`, axes
(`freqs_mhz`, `times_jd`, `elevations`, `azimuths`), horizon metadata
(case name, θ_c where applicable, blocked solid angle), and the usual
config metadata (location, world, temperatures, sky model, beam file).

## eigsim bug fix (separate commit)

Passing the raw `load_horizon()` array (NaN = open sky, finite =
distance) into `cro.Beam(horizon=...)` poisons `simulate()` with NaN —
verified: `eigsim/output/batch_0000.npz` from the canonical run is
100% NaN. Minimal fix: convert with `np.isnan(horizon)` at the call
site in `run_canonical_sim.py`, and delete the stale NaN batch file.

## Testing

Smoke test (small: 1 orientation, few times, few frequencies) for each
case verifying:

- output is finite (catches the NaN mask bug class);
- `nohorizon` has fgnd = 0; `quarry` and `eigsep` have fgnd > 0;
- quarry blocked solid angle ≈ eigsep blocked solid angle.
