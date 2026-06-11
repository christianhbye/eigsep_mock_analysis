# Horizon Chromaticity Comparison

Analysis project (for a paper) comparing the spectral chromaticity of
simulated EIGSEP spectra under three horizon models. Uses the
`eigsim` package; does not ship with it.

Design spec: `../docs/superpowers/specs/2026-06-10-horizon-chromaticity-design.md`

## Horizon cases

All cases are boolean masks on the MWSS grid, True = open sky:

| Case        | Description                                                        |
|-------------|--------------------------------------------------------------------|
| `nohorizon` | θ = π cut: everything open, ground fraction 0                     |
| `quarry`    | Constant-θ cut; θ_c solved to match the blocked solid angle of the EIGSEP horizon (nearest MWSS ring boundary) |
| `eigsep`    | Realistic horizon: `np.isnan(horizon)` of `eigsim/data/horizon_mwss.npz` |

## Workflow

From the monorepo root:

```bash
# 1. Build the three horizon masks -> output/horizons.npz
uv run python horizon_chromaticity/make_horizons.py

# 2. Run the simulation for each case (hours each; checkpoint/resume
#    via output/<case>_batch_*.npz, safe to interrupt and rerun —
#    but resume with the SAME flags: batches don't record CLI args)
uv run python horizon_chromaticity/run_sims.py --case nohorizon
uv run python horizon_chromaticity/run_sims.py --case quarry
uv run python horizon_chromaticity/run_sims.py --case eigsep

# 3. Analyze in notebooks/ (chromaticity metrics live there)
```

## Simulation parameters

Full canonical grid (36 elevations x 36 azimuths = 1296 orientations),
one sidereal day (1436 times), 201 channels (50-250 MHz), GSM16 sky,
**noiseless** (raw t_sys; add noise in notebooks if needed).

## Outputs (`output/`, gitignored)

- `horizons.npz` — the three masks + quarry θ_c + blocked solid angles
- `chromaticity_<case>.npz` — `t_sys (N_ori, N_times, N_freqs)`, axes
  (`freqs_mhz`, `times_jd`, flat per-orientation `elevations`/`azimuths`
  plus grid axes `elev_vals`/`az_vals`), horizon + config metadata

## Smoke tests

Gated behind an env var so plain `uv run pytest` stays fast:

```bash
EIGSEP_SMOKE=1 uv run pytest horizon_chromaticity/test_smoke.py -v
```

Takes a few minutes (JAX compilation). Requires the eigsim data files.
