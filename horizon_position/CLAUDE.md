# CLAUDE.md

Guidance for Claude Code in this directory.

## What this is

Self-contained analysis (for a paper): how antenna-position error
changes the EIGSEP antenna temperature vs freq/LST, as a spec on
position knowledge. Uses `eigsim` and `eigsep_terrain`; not a package.
Sibling of `horizon_chromaticity/`. Zenith pointing only.

Spec:  `../docs/superpowers/specs/2026-06-13-horizon-position-sensitivity-design.md`
Plan:  `../docs/superpowers/plans/2026-06-13-horizon-position-sensitivity.md`

## Two environments (important)

- `make_horizons.py` imports `eigsep_terrain` (NOT in the mock_analysis
  env). Run it with `PYTHONPATH=<eigsep_terrain path> uv run --project
  <eigsep_terrain path> python ...` (the PYTHONPATH is required because
  eigsep_terrain uses a flat layout and is not installed into its venv).
- `run_sims.py`, the pure modules, and the tests use `eigsim`/`s2fft`
  in the default env: `uv run python ...` / `uv run pytest ...`.

## Method / critical conventions

- The horizon is a **continuous** elevation curve `alpha_h(az)` from
  `calc_horizon` (azimuth = `atan2(E, N)`, North->East). It is turned
  into an **anti-aliased (fractional)** open-sky mask `W in [0,1]` on the
  MWSS beam grid (`masks.open_sky_weight`). Fractional weighting is what
  lets sub-pixel (0.1 m) horizon shifts register — a boolean mask floors
  them to zero.
- **Frame map:** croissant beam/grid azimuth `phi` is from ENU East;
  `calc_horizon` azimuth is from North. They are related by
  `phi = pi/2 - az`. Verified against the nominal `horizon_mwss.npz` in
  `test_validation.py` (~99.8% mask agreement).
- `eigsim.simulate`/`compute_fgnd` apply the horizon as a float multiply
  (croissant stores it as-is, no booleanization) and normalize by the
  full-sphere beam integral, so a fractional `W` flows through correctly.
- Set `os.environ.setdefault("JAX_ENABLE_X64", "1")` before any
  jax/s2fft/croissant/eigsim/eigsep_terrain import.
- `output/` is gitignored. The notebook loads npz + `analysis.py` only;
  it never imports the scripts.

## Files

- `positions.py` / `masks.py` / `analysis.py` — pure, unit-tested.
- `make_horizons.py` -> `output/horizons_position.npz` (eigsep_terrain env).
- `run_sims.py` -> `output/position_sims.npz` (eigsim env; resumable
  per-position checkpoints `pos*_batch_*.npz`; `pos_sha` guards against a
  stale `horizons_position.npz`).
- `notebooks/horizon_position.ipynb` — modes, spec curve, reductions.
