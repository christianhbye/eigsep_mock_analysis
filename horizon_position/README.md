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

# 2b. one nominal-horizon day per antenna, for the beam comparison.
#     --vivaldi (or EIGSEP_VIVALDI_BEAM) points at the HEALPix Vivaldi beam,
#     which lives outside this repo. Resampling it to MWSS takes a few minutes.
uv run python horizon_position/run_beam_sims.py     # -> output/beam_sims.npz

# 3. the analysis + all three paper figures, in this order
#    (signal_loss reads the Vh that horizon_shift writes)
uv run jupyter nbconvert --to notebook --execute --inplace \
    horizon_position/notebooks/horizon_shift.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace \
    horizon_position/notebooks/signal_loss.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace \
    horizon_position/notebooks/beam_comparison.ipynb
#    or open any of them in Jupyter and run it top to bottom

# tests
uv run pytest horizon_position/ -v                  # pure-module unit tests
EIGSEP_SMOKE=1 uv run pytest horizon_position/test_smoke.py -v
```

## The notebooks

Three notebooks, four paper figures, no figure scripts. Each is a paper
trail: it loads the raw inputs, derives everything in-notebook, renders
its figures, prints every number the paper quotes, and exports the paper
repo's committed npz + standalone notebook + PDF.

- `notebooks/horizon_shift.ipynb` — the horizon geometry
  (`horizon_perturbations_1col.pdf`) and the antenna-position systematic
  (`horizon_shift.pdf`): loads the simulation output and the Zeus21
  ensemble, computes `dT_ant`, builds the foreground eigenbasis, and
  re-derives `N_ANCHOR`.
- `notebooks/signal_loss.ipynb` — Fig. 1 (`signal_loss.pdf`) and the
  draft prose (`signal_loss_text.tex`): pushes the 21 cm ensemble through
  the identical projection and reads the retained signal off the same
  axes as the foreground residual. One single-column panel, two curves,
  no dimension marked — `N_ANCHOR` is where the *prose* quotes numbers,
  not something either figure draws.

- `notebooks/beam_comparison.ipynb` — `beam_comparison.pdf` and its draft
  prose: the same sky, horizon and filter run for three beams (isotropic,
  EIGSEP bowtie, Vivaldi feed), each with its own eigenbasis and its own
  open-sky attenuation, to separate what the antenna costs from what the sky
  does. Measures the design claim of `subsec:eig_antenna`.

Run `horizon_shift` first — `signal_loss` reads the `Vh` it publishes.
`beam_comparison` is independent of both, but asserts its bowtie panel against
`foreground_svd.npz` and `paper.N_ANCHOR`.

Neither notebook imports the other. What they must agree on lives in
`paper.py` as constants (`N_ANCHOR`, `N_SHOW`, `N_MODELS`, and where the
paper's files are); each notebook re-derives those from the raw inputs
and asserts against them, so a regenerated ensemble fails loudly in both
rather than quietly moving one figure.

The plotting code is lifted into the exported standalone notebooks with
`inspect.getsource`, so the version archived with the paper is always
byte-identical to the version that produced the PDF.

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

## Other files

- `paper.py` — where the paper's artifacts live, plus the constants the
  two figures share. No analysis. `EIGSEP_PAPER_NOTEBOOKS` overrides the
  output location, so a run can target a worktree instead of the live
  paper directory.
- `signal_loss_text.tex.in` — the LaTeX template `signal_loss.ipynb`
  substitutes into `signal_loss_text.tex`. Edit the prose here.
- `reionization_sensitivity.py` — one-off audit of how much
  `models_21cm`'s reionization threshold moves the reported numbers.
