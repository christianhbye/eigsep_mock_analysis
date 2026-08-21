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
- `output/` is gitignored. The notebooks load npz + `analysis.py` +
  `paper.py` + `../models_21cm/selection.py`, and nothing else.

## Files

- `positions.py` / `masks.py` / `analysis.py` — pure, unit-tested.
- `make_horizons.py` -> `output/horizons_position.npz` (eigsep_terrain env).
- `run_sims.py` -> `output/position_sims.npz` (eigsim env; resumable
  per-position checkpoints `pos*_batch_*.npz`; `pos_sha` guards against a
  stale `horizons_position.npz`).
- `notebooks/horizon_shift.ipynb`, `notebooks/signal_loss.ipynb` — **the**
  analysis. See below.
- `paper.py` — artifact locations + the constants the two figures share
  (`N_ANCHOR`, `N_SHOW`, `N_MODELS`). No analysis, by design.
- `signal_loss_text.tex.in` — LaTeX template for the draft prose.
- `reionization_sensitivity.py` — one-off audit of `models_21cm`'s
  reionization threshold; settled, not part of the figure pipeline.
- `rotation_dimensionality.py` — mode count over the full drive grid vs
  zenith; a by-product, not a figure.

## The notebooks are the only figure producers

Two notebooks produce all three paper figures. There are no
`make_*_figure.py` scripts — do not add one. Each goes raw inputs ->
derived quantities -> figures -> paper-repo export, deriving everything
in-notebook rather than importing it.

- `horizon_shift.ipynb` -> `horizon_perturbations_1col.pdf` and
  `horizon_shift.pdf`. Inputs: `output/position_sims.npz`,
  `output/horizons_position.npz`, the Zeus21 ensemble.
- `signal_loss.ipynb` -> `signal_loss.pdf` and `signal_loss_text.tex`.
  Inputs: the paper's `foreground_svd.npz`, the Zeus21 ensemble, and
  `horizon_shift.npz` (§7 only, for the caption blocks).
- **Run `horizon_shift.ipynb` first** — `signal_loss.ipynb` asserts its
  eigenbasis against the `Vh` that one publishes.
- Neither notebook imports the other. Shared values live in `paper.py` as
  constants; each notebook re-derives them and asserts. Do not move a
  derivation into `paper.py` — that is what makes the assert meaningful.
- Exports into the paper's `notebooks/` dir:
  `horizon_shift.{npz,ipynb,pdf}`, `horizon_perturbations.{npz,ipynb}` +
  `horizon_perturbations_1col.pdf`, `signal_loss.{npz,ipynb,pdf}` and
  `signal_loss_text.tex`.
- The exported notebooks must be **standalone** (Zenodo convention: they
  load the committed npz and import nothing from this repo). Their code is
  lifted from the live kernel with `inspect.getsource`, so there is exactly
  one copy of every plotting function. Change a figure by editing the
  function in the notebook and re-running — never by editing the export.
- Several asserts keep the figures consistent; do not remove them. Both
  notebooks check their survivor count against `paper.N_MODELS` and their
  recomputed anchor against `paper.N_ANCHOR`; `horizon_shift` checks its
  baseline against `foreground_svd.npz`; `signal_loss` checks its
  eigenbasis against `horizon_shift.npz` and gates the "costs one mode"
  phrasing its caption uses.
- `N_ANCHOR` is *recomputed*, not imported: the smallest N at which the
  foreground floor drops below the median retained 21 cm signal **and stays
  below** for every larger N. A first-crossing rule gives a different,
  over-optimistic answer — both curves cross more than once.
- Do not use `plt.rc_context` in a figure cell. It restores the `backend`
  rcParam on exit, which resets the inline backend's post-execute hook and
  silently stops every *later* cell from displaying its figure. Set font
  sizes per-artist instead.
