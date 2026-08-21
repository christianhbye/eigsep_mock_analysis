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
- `output/` is gitignored. The notebook loads npz + `analysis.py` +
  `../models_21cm/selection.py`, and imports `make_paper_signal_loss_figure`
  only to assert consistency with Fig. 1 (see below).

## Files

- `positions.py` / `masks.py` / `analysis.py` — pure, unit-tested.
- `make_horizons.py` -> `output/horizons_position.npz` (eigsep_terrain env).
- `run_sims.py` -> `output/position_sims.npz` (eigsim env; resumable
  per-position checkpoints `pos*_batch_*.npz`; `pos_sha` guards against a
  stale `horizons_position.npz`).
- `notebooks/horizon_shift.ipynb` — **the** analysis. See below.
- `make_paper_signal_loss_figure.py` — the paper's Fig. 1
  (`signal_loss.pdf`), a separate figure. Reads `horizon_shift.npz` for
  `Vh`, so the notebook must run first.
- `reionization_sensitivity.py` — one-off audit of `models_21cm`'s
  reionization threshold; settled, not part of the figure pipeline.

## The notebook is the only figure producer

`notebooks/horizon_shift.ipynb` is the paper trail for both horizon
figures. There are no `make_*_figure.py` scripts — do not add one. It goes
raw inputs -> `dT_ant` -> foreground eigenbasis -> operating point ->
figures -> paper-repo export, deriving everything in-notebook rather than
importing it.

- Inputs: `output/position_sims.npz`, `output/horizons_position.npz`,
  `../models_21cm/output/zeus21_models.npz`. Nothing else is *derived* from
  another module; the one script import is assert-only.
- Exports into the paper's `notebooks/` dir: `horizon_shift.{npz,ipynb,pdf}`
  and `horizon_perturbations.{npz,ipynb}` + `horizon_perturbations_1col.pdf`.
- The exported notebooks must be **standalone** (Zenodo convention: they
  load the committed npz and import nothing from this repo). Their code is
  lifted from the live kernel with `inspect.getsource`, so there is exactly
  one copy of every plotting function. Change a figure by editing the
  function in the notebook and re-running — never by editing the export.
- Three asserts keep this figure and Fig. 1 consistent; do not remove them:
  the 21 cm selection matches `make_paper_signal_loss_figure.load_t21`, the
  baseline waterfall matches the paper's `foreground_svd.npz`, and the
  recomputed `N_ANCHOR` matches `make_paper_signal_loss_figure.N_ANCHOR`.
- `N_ANCHOR` is *recomputed*, not imported: the smallest N at which the
  foreground floor drops below the median retained 21 cm signal **and stays
  below** for every larger N. A first-crossing rule gives a different,
  over-optimistic answer — both curves cross more than once.
- Do not use `plt.rc_context` in a figure cell. It restores the `backend`
  rcParam on exit, which resets the inline backend's post-execute hook and
  silently stops every *later* cell from displaying its figure. Set font
  sizes per-artist instead.
