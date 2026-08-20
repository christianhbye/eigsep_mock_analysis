# CLAUDE.md

Guidance for Claude Code in this directory.

## What this is

Generates the citable 21 cm global-signal ensemble for the EIGSEP
instrument paper. Not a package; a self-contained generator, like
`horizon_position/`.

## Two environments (important)

- `priors.py`, `selection.py`, `provenance.py` are **pure** — numpy/scipy/
  stdlib only, never `zeus21`. Their tests run in the main env with
  `uv run pytest models_21cm/`. Keep them that way.
- `generate.py` and `verify_ensemble.py` need the pinned env:
  `uv run --project models_21cm python ...`. It carries `classy`, which
  compiles CLASS, and is excluded from the uv workspace so it never
  reaches the workspace lock or CI.

## Critical conventions

- **`zmin = 4.65`, not 4.681623.** The 250 MHz endpoint must sit inside
  the spline domain, not on its boundary.
- **Never zero-pad above the native band.** Late-reionization models are
  still at ~14 mK at 237 MHz; a step there survives a smooth-mode filter
  and corrupts the retained-RMS statistic the figure reports.
- **Interpolate with a cubic spline in log z, never `np.interp`.** Linear
  kinks are non-smooth structure and survive the filter.
- **Pop III needs `zeus21.Correlations(...)` during setup** and
  `USE_RELATIVE_VELOCITIES=True`, or you get `KeyError: 'xi_RR_CF'`.
- **Zeus21's `setup.py` omits `powerbox` and `pyfftw`** though
  `zeus21.maps` imports them at package import time.
- The reionization cut is **posterior** and applied in `load_t21()`, once,
  so the figure and the statistics cannot disagree.
- The cut has **two parts**, both required: `xHI(z=5.9) < 0.1` (the
  McGreer reference point) AND `xHI(z=4.6816) < 0.05` (the band's top
  edge, `z(250 MHz)`). The second exists because Zeus21's Q-based
  reionization ODE can re-neutralize at very low escape fractions
  (recombination outruns the ionizing supply) — a real model limitation,
  not a numerical artifact, confirmed by re-running offenders at three
  different `zmin` values with identical results. The `0.05` threshold is
  a chosen value, not a derived one; alternatives are documented and
  compared in `README.md`. Do not tighten or loosen it without recording
  the new survivor count and re-running `verify_ensemble.py`.
- **Worker count is bounded by memory, not cores.** Each `generate.py`
  worker costs ~2.78 GB private peak RSS at `precisionboost = 3`. Sizing
  `--processes` off `nproc` instead of available memory is how an 8-worker
  run gets OOM-killed on a 15 GB machine.
- `output/` is gitignored; the npz goes to Zenodo at publication.
