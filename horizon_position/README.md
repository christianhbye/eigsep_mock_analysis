# Horizon Position Sensitivity

How much does the EIGSEP antenna temperature change when the antenna
position is wrong by 0.1, 1, or 10 m? This quantifies the spec on how
well the suspended antenna's position must be known, for forward
modelling. Zenith pointing only.

Design: `../docs/superpowers/specs/2026-06-13-horizon-position-sensitivity-design.md`
Plan:   `../docs/superpowers/plans/2026-06-13-horizon-position-sensitivity.md`

## Method (one line)

Continuous horizon curve from `eigsep_terrain.calc_horizon` per position
-> anti-aliased (fractional) open-sky mask -> `eigsim.simulate`
(zenith-only) -> difference vs nominal. Fractional masks resolve
sub-pixel horizon shifts that a boolean mask would floor to zero.

## Pipeline (two environments!)

`make_horizons.py` needs `eigsep_terrain`; everything else needs
`eigsim`. They are separate uv envs. `make_horizons.py` also needs
`PYTHONPATH` set because eigsep_terrain uses a flat (non-src) layout and
is not installed into its venv.

```bash
# 1. horizon curves (eigsep_terrain env)
PYTHONPATH=/home/christian/Documents/research/eigsep/eigsep_terrain \
uv run --project /home/christian/Documents/research/eigsep/eigsep_terrain \
    python horizon_position/make_horizons.py        # -> output/horizons_position.npz

# 2. per-position waterfalls (mock_analysis env, from monorepo root)
uv run python horizon_position/run_sims.py          # -> output/position_sims.npz

# 3. the analysis + both paper figures
uv run jupyter lab horizon_position/notebooks/horizon_shift.ipynb
#    or headless:
uv run jupyter nbconvert --to notebook --execute --inplace \
    horizon_position/notebooks/horizon_shift.ipynb

# 4. Fig. 1 -- reads the Vh that step 3 writes, so it must come after
uv run python horizon_position/make_paper_signal_loss_figure.py

# tests
uv run pytest horizon_position/ -v                  # pure-module unit tests
EIGSEP_SMOKE=1 uv run pytest horizon_position/test_smoke.py -v
```

## The notebook

`notebooks/horizon_shift.ipynb` is the paper trail for both horizon
figures, end to end: it loads the raw simulation output and the Zeus21
21 cm ensemble, computes `dT_ant`, builds the foreground eigenbasis
explicitly, re-derives the operating point `N_ANCHOR`, renders the
figures, prints every number the paper quotes, and exports the paper
repo's committed npz + standalone notebook + PDF.

There are no figure scripts. It imports `make_paper_signal_loss_figure`
in one place only, to assert that Fig. 1 and this figure cut the same
21 cm models and agree on `N_ANCHOR` — nothing is derived from it.

The plotting code is lifted into the exported standalone notebooks with
`inspect.getsource`, so the version archived with the paper is always
byte-identical to the version that produced the PDF.

Figures produced (written into the paper's `notebooks/` directory):

- `horizon_perturbations_1col.pdf` — the horizon geometry: `alpha_h(az)`
  and `Delta alpha_h(az)` for +1 m East/North/Up.
- `horizon_shift.pdf` — `dT_ant(nu)` at 24 LSTs (top) and its residual
  after filtering the leading N foreground modes, against the retained
  21 cm signal (bottom).

## Outputs (`output/`, gitignored)

- `horizons_position.npz` — `alpha_h(az)` per position (+ enu, az grid).
- `position_sims.npz` — `t_sys (19, n_times, n_freqs)`, `fgnd (19, n_freqs)`, axes, metadata.

## Analysis modes

`analysis.delta_waterfall(..., mode)` with `mode` in:
- `uncorrected` (default) — raw `t_sys` difference. What the paper
  figure uses: the position error is by hypothesis unknown, so the
  observer cannot apply the displaced position's own `fgnd`.
- `oracle` — ground-loss corrected with each position's true `fgnd`.
- `miscorrected` — corrected with the *nominal* `fgnd` (position error
  unknown); equals `uncorrected / (1 - fgnd_nominal)`.

## Other scripts

- `make_paper_signal_loss_figure.py` — the paper's Fig. 1
  (`signal_loss.pdf`), a different figure. Reads `horizon_shift.npz`
  for the foreground modes, so run the notebook first.
- `reionization_sensitivity.py` — one-off audit of how much
  `models_21cm`'s reionization threshold moves the reported numbers.
