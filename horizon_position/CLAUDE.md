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
- `notebooks/horizon_shift.ipynb`, `notebooks/signal_loss.ipynb`,
  `notebooks/beam_comparison.ipynb` — **the** analysis. See below.
- `beams.py` — pure, unit-tested: HEALPix->MWSS beam resampling and the
  isotropic reference beam. `NITER = 3` is load-bearing; `test_beams.py`
  pins it by reproducing eigsim's stored MWSS bowtie bit-for-bit, and
  asserts that niter=0 does *not*.
- `run_beam_sims.py` -> `output/beam_sims.npz` (eigsim env; one
  nominal-horizon sidereal day per beam, resumable per-beam checkpoints
  `beam_<tag>.npz`). The Vivaldi HEALPix beam lives outside this repo:
  `--vivaldi` / `EIGSEP_VIVALDI_BEAM`.
- `paper.py` — artifact locations + the constants the two figures share
  (`N_ANCHOR`, `N_SHOW`, `N_MODELS`). No analysis, by design.
- `paper_text.tex.in` — LaTeX template for the whole draft prose, all
  six blocks in manuscript order. Substituted once, by
  `beam_comparison.ipynb`, which runs last; `signal_loss.ipynb`
  publishes its values to `output/signal_loss_vals.npz` instead of
  writing a file. Replaced `signal_loss_text.tex.in` and
  `beam_comparison_text.tex.in` on 2026-08-26.
- `reionization_sensitivity.py` — one-off audit of `models_21cm`'s
  reionization threshold; settled, not part of the figure pipeline.

## The notebooks are the only figure producers

Three notebooks produce all four paper figures. There are no
`make_*_figure.py` scripts — do not add one. Each goes raw inputs ->
derived quantities -> figures -> paper-repo export, deriving everything
in-notebook rather than importing it.

- `horizon_shift.ipynb` -> `horizon_perturbations_1col.pdf` and
  `horizon_shift.pdf`. Inputs: `output/position_sims.npz`,
  `output/horizons_position.npz`, the Zeus21 ensemble.
- `signal_loss.ipynb` -> the analysis behind blocks 1, 4, 5 and 6 of
  `paper_text.tex`, published as `output/signal_loss_vals.npz`. **It no
  longer contributes a paper figure**: `signal_loss.pdf` was retired on
  2026-08-26 when `beam_comparison.pdf` became Fig. 1, because its content is
  that figure's central panel. The notebook still derives `N_ANCHOR`,
  `N_HAND` and every millikelvin number the prose quotes. Do not delete it.
  It still writes `signal_loss.pdf` and `signal_loss.npz`, which are simply
  no longer included by the manuscript. Formerly ->
  Inputs: the paper's `foreground_svd.npz`, the Zeus21 ensemble, and
  `horizon_shift.npz` (§7 only, for the caption blocks).
- `beam_comparison.ipynb` -> `beam_comparison.pdf` and
  `beam_comparison_text.tex`. Inputs: `output/beam_sims.npz`, the Zeus21
  ensemble, and `foreground_svd.npz` as a cross-check. Three panels — isotropic
  / bowtie / Vivaldi — each the Fig. 1 axes for one antenna. **Each antenna is
  filtered in its own eigenbasis and attenuated by its own `eta`**, so the
  three grey bands are not one band repeated; that is why it is three panels
  and not three curves. Do not put them on one panel, and do not plot retention
  against mode count: at fixed N the antenna that suppressed *less* retains
  more, which inverts the conclusion. Quote retained *fraction* at matched
  foreground suppression.
- **Run order is `horizon_shift` -> `signal_loss` -> `beam_comparison`.**
  The last one merges the other's substitution values and writes the single
  `paper_text.tex`; it fails with a pointer if `signal_loss_vals.npz` is
  absent. Run `horizon_shift.ipynb` first — `signal_loss.ipynb` asserts its
  eigenbasis against the `Vh` that one publishes. `beam_comparison.ipynb` is
  independent of both but asserts its bowtie panel reproduces
  `foreground_svd.npz` and `paper.N_ANCHOR`.
- **The Vivaldi is not a rejected candidate.** It is a HERA Phase II feed, it
  is what EIGSEP uses for the ground antennas (`sec:vivaldis`), and it is the
  antenna the October 2024 suspension flew. Block 1 frames the comparison
  through **design intent, not defence**: "used in isolation, without a dish,
  as for the ground antennas", then one sentence of history — a prototype
  design, flown on the first suspension, never optimised for low chromaticity,
  with the bowtie developed afterwards against that criterion. "Optimised"
  attached to the bowtie carries the whole contrast. An earlier revision
  defended the Vivaldi in its own right ("where low chromaticity is not
  required", "the ground antennas have a different job"); that read as a talk
  aside and as protesting too much, and was cut on 2026-08-27.
- **The aperture point is bookkeeping, not an argument.** `ETAVIV > ETABOW` is
  still asserted, and the fact is stated in the units paragraph (the Vivaldi
  carries proportionally more 21-cm amplitude into its filter) *before* the
  retained fractions land, where it is needed anyway to explain the
  attenuation. It used to be a standalone sentence ("begins with more signal
  and still ends with less") — persuasive, but in a voice the paper does not
  otherwise use. Never say the Vivaldi ends with less in **absolute** retained
  amplitude: `eta` divides out of the fractions and in absolute terms the two
  antennas are nearly tied.
- **Khalichi et al. lives in block 1's caveat paragraph**, and it is the
  paper's *first* mention (`subsec:covariance` is at `rasti_template.tex:204`,
  `subsec:eig_antenna` at 344). It therefore carries the full deferral plus a
  forward `\ref` to `subsec:eig_antenna`; the sentence at
  `rasti_template.tex:354` should shrink to a back-reference so the same
  phrase is not printed twice. That edit is in the paper repo, not here.
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
- **The 21 cm ensemble is in antenna temperature.** Both figures work in
  uncorrected antenna temperature (ground pickup in, receiver out), so both
  notebooks multiply the models by the beam-weighted open-sky fraction
  `eta = 1 - fgnd` (0.36–0.55 over the band, mean 0.446) before filtering —
  an isotropic signal reaches that observable attenuated. Each notebook
  derives `eta` from its own input (`foreground_svd.npz` / `position_sims.npz`
  row 0, verified identical); `paper.N_ANCHOR` is the assert that catches them
  disagreeing. Do NOT filter the sky-referred `T21_mK`: that compares a
  sky-referred signal against an antenna-temperature residual and overstates
  retention by 2.2x. Ground-loss correcting everything instead is
  self-consistent and gives the same answer, but needs the beam knowledge
  block 1 claims not to use, and makes the foregrounds less compressible.
- **`eta` is per beam, and every value the paper quotes is tokened.** The
  0.36–0.55 above is the *bowtie*. `beam_comparison.ipynb` carries three:
  0.36–0.55 (bowtie), 0.59–0.80 (Vivaldi), 0.35 (isotropic, achromatic — its
  standard deviation across the band is identically zero, which the assert in
  section 1.2 guarantees and which is why the paper quotes it as one number
  rather than a range). Until 2026-09-02 only the bowtie range reached the
  manuscript through tokens (`ETALO`/`ETAHI` from `signal_loss.ipynb`) and the
  other two were typed by hand — the same failure mode that took out the
  rotation paragraph. The loop now emits `ETALO{ISO,BOW,VIV}` and
  `ETAHI{ISO,BOW,VIV}`. **Never hand-type an eta into the paper text.**
- `T21_sky` (unattenuated) is kept alongside `T21` in `signal_loss` for trough
  depth, trough width and the selections built on them — those describe the
  *model*, not the observation. Everything filtered or compared against a
  foreground residual uses `T21`. Do not mix them.
- Several asserts keep the figures consistent; do not remove them. Both
  notebooks check their survivor count against `paper.N_MODELS` and their
  recomputed anchor against `paper.N_ANCHOR`; `horizon_shift` checks its
  baseline against `foreground_svd.npz`; `signal_loss` checks its eigenbasis
  and `N_HAND` against `horizon_shift.npz`, gates the systematic-to-signal
  ratio at `N_HAND`, and gates the caption's claim that the vertical clears
  the median at exactly the mode carrying its leftover spike.
- `N_ANCHOR` is *recomputed*, not imported: the smallest N at which the
  foreground floor drops below the median retained 21 cm signal **and stays
  below** for every larger N. Keep the stays-below rule — it is the
  conservative choice and it is what the caption's inequality asserts — even
  though with the attenuated ensemble it happens to pick the same N as a
  first-crossing rule.
- **`N_ANCHOR` and `N_HAND` are different depths.** `N_HAND` (= `N_ANCHOR - 1`
  = 9) is where an unmodelled 1 m vertical error overtakes the foreground
  floor: a foreground-vs-*systematic* crossing. `N_ANCHOR` (10) is a
  foreground-vs-*signal* one. They coincided before the ensemble was
  attenuated. The **anatomy** of the position systematic — the leftover-spike
  mode, its amplitude, the cosine similarity — is quoted at `N_HAND`, because
  99% of what a vertical error leaves there is the single mode `N_ANCHOR`
  filters, so an anatomy read at `N_ANCHOR` describes the remainder after that
  mode rather than the error. The **induced floors** `floor_own`/`floor_nom`
  are NOT: they moved to `N_ANCHOR` on 2026-09-03. `N_HAND` is where the
  perturbation looks worst against the retained signal (2.59x the median,
  against 0.36x one mode later), so a tolerance quoted there — one mode short
  of the depth section 2.1 sets the filter at — is indefensible even though
  every individual number was correct. Sections 2.1 and 5.5 of the manuscript
  now read at the same depth. Everything else quotes at `N_ANCHOR`.
- **The two landmarks use different crossing rules, deliberately.** `N_ANCHOR`
  takes the *stays-below* rule; `N_HAND` takes *first crossing*. They are not
  interchangeable: the +1 m vertical overtakes the foreground floor at N = 9
  (3.01 against 1.82 mK), dips back under at N = 10 and 11, and is above for
  good only from N = 12 — so a stays-above rule would put the handover at 12
  and break the `N_HAND == N_ANCHOR - 1` assert. First crossing is right here
  because the crossing at 9 and the dip at 10 are the same fact: mode 10
  carries 99% of what the vertical leaves, and that spike is block 4's closing
  argument rather than a defect in the landmark. `print_handover` returns both
  (`first`, `stays`) and prints the pair; keep it that way.
- **`N_HAND` is the vertical's crossing, not the position error's.** Of the 18
  simulated displacements only six cross the foreground floor within N ≤ 18 at
  all: Up ±1 m at 9, Up ±10 m at 5–6, East ±1 m at 16, East ±10 m at 8, North
  ±10 m at 9–12. Nothing at 0.1 m in any direction crosses, and neither does
  north at 1 m. Block 4's appositive therefore says "the vertical error first
  overtakes"; do not generalise it back to "the position error", which is false
  for the east and north floors quoted in the same sentence.
- **Neither figure draws anything at `N_ANCHOR`.** It is the dimension the
  *prose* quotes millikelvin numbers at, and the common dimension at which
  `horizon_shift` compares the position systematic against the 21 cm
  ensemble. `signal_loss.pdf` deliberately marks no dimension: a labelled N
  in the figure reads as an adopted operating point, which is the misreading
  the referee already made of the old 10 mK residual. Do not add a vertical
  line, a colour scale keyed to one N, or a frequency-space panel drawn at
  one — that is what the single-panel version exists to avoid.
- The 21 cm ensemble is a grey (`C_21 = "0.40"`) 5–95 band with a dashed
  median in **both** figures. Keep them identical; the reader learns the
  artist in `signal_loss.pdf` and reads it again in `horizon_shift.pdf`.
- **Vocabulary: filtering, not geometry.** Both residual axes are
  `Foreground modes filtered` / `Residual RMS [K]` — the axes of the
  `foreground_svd_residual.pdf` these curves were added to — and the prose,
  the captions and `signal_loss_text.tex.in` all describe the operation as
  filtering foreground eigenmodes, which is the manuscript's own wording. An
  earlier revision purged "filter" in favour of subspace/orthogonal-complement
  language; that was reversed. Where geometry *is* the argument (block 5's
  own-basis floors) "basis" and "subspace" stay. What the wording must still
  not do is present N as a depth anyone proposes to apply to data.
- Paper style: legend entries are capitalized (`Beam-weighted foregrounds`)
  and 21-cm is hyphenated when adjectival (`21-cm models`).
- No 10 mK anywhere. The manuscript's "to filter foregrounds to 10 mK we need
  the first eight modes", and the 10 mK line the old figure drew, are what the
  referee read as a sensitivity claim. The residual at eight modes is still
  quoted, as the dynamic range the foregrounds are described to; neither
  figure draws a threshold.
- `signal_loss.pdf` is single-column (3.4 in) and belongs in a `figure`, not
  the `figure*` its three-panel predecessor needed.
- Do not use `plt.rc_context` in a figure cell. It restores the `backend`
  rcParam on exit, which resets the inline backend's post-execute hook and
  silently stops every *later* cell from displaying its figure. Set font
  sizes per-artist instead.
