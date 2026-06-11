# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
in this directory.

## What this is

Self-contained analysis project (for a paper) comparing the spectral
chromaticity of simulated EIGSEP spectra under three horizon models.
It **uses** the `eigsim` workspace package but is not a package itself —
plain scripts + notebooks, no pyproject.toml, not shipped or released.

Design spec: `../docs/superpowers/specs/2026-06-10-horizon-chromaticity-design.md`
Implementation plan: `../docs/superpowers/plans/2026-06-10-horizon-chromaticity.md`

## Commands

Run from the monorepo root, always via `uv run`:

```bash
uv run python horizon_chromaticity/make_horizons.py        # build masks -> output/horizons.npz
uv run python horizon_chromaticity/run_sims.py --case eigsep   # one sim case (hours)
EIGSEP_SMOKE=1 uv run pytest horizon_chromaticity/test_smoke.py -v   # smoke tests (~5 min)
```

Smoke tests are gated behind `EIGSEP_SMOKE=1` so plain `pytest` skips
them (they spawn subprocesses and compile JAX).

## Architecture

**make_horizons.py -> output/horizons.npz -> run_sims.py (x3 cases) ->
output/chromaticity_<case>.npz -> notebooks/**

Three horizon cases, all boolean masks on the MWSS grid (130, 258),
**True = open sky** (the convention `croissant.Beam(horizon=...)` expects):

- `nohorizon` — all open, ground fraction 0
- `quarry` — constant-θ ring cut (ring 52, θ_c ≈ 71.16°), solved by
  matching blocked **solid angle** (not pixel count) to the eigsep case
- `eigsep` — `np.isnan(raw)` of `eigsim/data/horizon_mwss.npz`

Chromaticity metrics live in `notebooks/`, never in the scripts — the
scripts only produce raw noiseless `t_sys` (no radiometer noise).

## Critical conventions

- **Horizon mask trap:** `eigsim.load_horizon()` returns the RAW file
  array — distance-to-terrain, finite = blocked, NaN = open sky. It must
  be converted (`np.isnan(...)`) before any use as a mask. Passing it
  raw makes `eigsim.simulate()` return all-NaN.
- **Solid angles** use the same s2fft MWSS quadrature weights as
  `eigsim.simulate()`: per-pixel weight = `w[ring]`,
  `w = s2fft.utils.quadrature_jax.quad_weights(L=lmax+1, sampling="mwss")`.
- Set `os.environ.setdefault("JAX_ENABLE_X64", "1")` **before** any
  jax/s2fft/croissant/eigsim import (scripts and test modules do this).
- `output/` is gitignored — never commit npz outputs.
- Notebooks load only npz files from `output/`; they must not import
  from the script files.

## run_sims.py specifics

- Checkpoint/resume: batches saved as `output/<case><tag>_batch_*.npz`,
  merged then deleted. Safe to interrupt — but **resume with the same
  CLI flags**; batches record the horizon-mask SHA (stale `horizons.npz`
  aborts the run) but not the other CLI args.
- `--n-times/--max-orientations/--freq-stride/--batch-size/--output-tag`
  exist for smoke tests only; production runs use defaults.
- In `chromaticity_<case>.npz`: `elevations`/`azimuths` are FLAT
  per-orientation arrays (length 1296). Use `elev_vals`/`az_vals`
  (36 each) for grid reshaping. (Note this differs from
  `eigsim/output/canonical_sim.npz`, where those keys hold grid axes.)
