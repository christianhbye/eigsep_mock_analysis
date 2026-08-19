# Zeus21 21 cm Model Ensemble — Design

**Date:** 2026-08-19
**Status:** Draft (awaiting review)

## Goal

The EIGSEP instrument paper's signal-loss figure (PR #5) pushes an
ensemble of global 21 cm models through the same projection onto the
leading foreground modes as the foregrounds themselves, so that the
residual and the retained signal are read off the same axes. It exists
to answer a referee who objected that the eigenmode analysis
established only that *simulated foregrounds* are low-dimensional, not
that the cosmological signal survives the same filter.

The ensemble currently used,
`~/Documents/research/eigsep/normalizing_flows/models_21cm.npz`, cannot
be cited. It is 1135 models on a 50–249 MHz grid, dated August 2022,
with **no surviving generating script** anywhere on disk (searched
2026-08-19). Its origin is plausibly 21cmGEM — the 21cmVAE training set
— but that is unconfirmed. An ensemble nobody can trace is a referee
liability on a figure whose entire job is answering a referee.

This project **regenerates the ensemble with Zeus21**, so that every
model traces to published code at a pinned commit, a written parameter
range, and a fixed random seed.

## Why Zeus21

Zeus21 (Muñoz 2023a, arXiv:2302.08506) encodes an effective model for
the cosmic-dawn 21-cm signal that exploits the approximate
log-normality of the star-formation rate density. It agrees with
semi-numerical simulations to ~10% while costing seconds per model, and
it is a published, maintained, citable code. Population III stars with
Lyman-Werner feedback and relative velocities come from Cruz et al.
2024 (arXiv:2407.18294).

Alternatives considered and rejected:

- **21cmFAST** — the reference semi-numerical code, but minutes to
  hours per model makes a 4096-model ensemble impractical, and we need
  only the global signal, not the fields it computes to get there.
- **21cmGEM / 21cmVAE** — emulators, and therefore themselves derived
  products carrying their own provenance chain. Regenerating from an
  emulator whose training set we would also have to trace does not
  solve the problem that motivated this work.

## Scope

**In scope:** a generator package with its own pinned environment, the
regenerated ensemble, the swap of `load_t21()` in
`make_paper_signal_loss_figure.py`, and recomputation of the operating
point that the swap invalidates.

**Out of scope:** 21-cm power spectra, varying cosmology, and enforcing
UVLF or CMB-tau consistency on the priors.

## Model space

### Fixed

| setting | value | note |
|---|---|---|
| cosmology | Planck 2018 | Zeus21 defaults: `omegab=0.0223828`, `omegac=0.1201075`, `h=0.67810`, `As=2.100549e-9`, `ns=0.9660499`, `tau=0.05430842` |
| `astromodel` | 0 | GALUMI-like |
| `accretion_model` | 0 | exponential |
| `HMF_CHOICE` | `"ST"` | Sheth-Tormen |
| `USE_RELATIVE_VELOCITIES` | `True` | required for the Pop III / LW machinery |
| `A_vcb`, `beta_vcb` | 1.0, 1.8 | Cruz+2024 calibration against simulations; not an astrophysical free parameter |
| `Mturn_fixed` | `None` | Pop II turnover follows the atomic-cooling mass `M_atom(z)` |
| `Nalpha_lyA_II/III` | 9690 / 17900 | BL05; largely degenerate with `epsstar` for the coupling |
| `betastar`, `Mc` | −0.5, 3e11 | high-mass rollover of f⋆(Mₕ); drives the UVLF bright end, minor for the global signal |

### Varied — 14 parameters

Priors are deliberately **broad and agnostic**: wide flat ranges that
span weak-to-extreme signals, rather than ranges tied to what current
observations allow. The figure is a stress test of the filter, not a
forecast, so including signals already disfavoured makes the claim
stronger rather than weaker.

| # | parameter | range | default | sets |
|---|---|---|---|---|
| 1 | log₁₀ `epsstar` | −2.5 … −0.5 | −1.0 | Pop II SFR efficiency → coupling/heating timing |
| 2 | `alphastar` | 0.0 … 1.0 | 0.5 | low-mass slope of f⋆(Mₕ) |
| 3 | `dlog10epsstardz` | −0.5 … 0.5 | 0.0 | redshift evolution of efficiency |
| 4 | log₁₀ `fesc10` | −2.5 … 0.0 | −1.0 | reionization timing → 130–250 MHz |
| 5 | `alphaesc` | −1.0 … 1.0 | 0.0 | mass slope of f_esc |
| 6 | log₁₀ `L40_xray` | −2.0 … 2.0 | 0.48 | X-ray heating → trough depth, emission |
| 7 | `alpha_xray` | −2.0 … 0.0 | −1.0 | X-ray SED hardness |
| 8 | log₁₀ (`E0_xray`/eV) | 2.0 … 3.2 | 2.70 | soft X-ray absorption cutoff |
| 9 | log₁₀ `fstar_III` | −4.0 … −1.5 | −2.5 | Pop III efficiency |
| 10 | log₁₀ (`Mc_III`/M⊙) | 5.5 … 8.0 | 7.0 | minihalo mass scale |
| 11 | log₁₀ `L40_xray_III` | −2.0 … 2.0 | 0.48 | Pop III X-ray heating |
| 12 | log₁₀ `fesc7_III` | −2.5 … −0.5 | −1.35 | Pop III escape fraction |
| 13 | `A_LW` | 0.0 … 4.0 | 2.0 | LW feedback strength (0 = none) |
| 14 | `beta_LW` | 0.3 … 1.0 | 0.6 | LW feedback mass dependence |

The four decades on both X-ray luminosity axes are deliberate: that
range spans fully-cold-absorption models through saturated-heating
models, which is most of the signal's dynamic range and the main axis
the figure tests against.

### Sampling

**Scrambled Sobol sequence** (`scipy.stats.qmc.Sobol`, fixed seed),
`random_base2(m=12)` = **4096 models**.

Sobol rather than a Latin hypercube because it is *extensible*: 4096 →
8192 preserves every existing model and stays low-discrepancy, whereas
an LHS is fixed-N and would have to be discarded and regenerated. Since
the ensemble size is settled by a convergence check that may say "more"
(below), extensibility is worth more than LHS's marginal stratification,
and Sobol's projection properties in 14 dimensions are better regardless.

### On what 4096 models can and cannot claim

4096 points do **not** fill a 14-dimensional space, and no achievable
number would — a 3-point-per-axis grid is 4.8 million models. Volume
filling is the wrong bar.

The right bar is whether the statistics the figure actually reports —
the 5/50/95 percentiles of retained RMS, and the fraction of models
above the foreground floor — are converged. Those are one-dimensional
summaries of a map from parameters to retained RMS whose *effective*
dimensionality is far below 14: the global signal is governed by
roughly amplitude, timing, and heating, with the 14 parameters feeding
those few combinations. Summary statistics converge on effective
dimension, not nominal dimension.

This is testable rather than assertable, and the validation section
requires it to be tested.

## Redshift and frequency coverage

The paper grid is **50–250 MHz at 1 MHz** (201 points), taken from
`foreground_svd.npz`.

Zeus21 is advertised for z = 5–35, i.e. 39.5–236.7 MHz, which does not
reach the top of the band. Running with `zmin_CLASS=4.5` and
`zmin=4.68` extends it to **250.1 MHz** and covers the whole grid with
computed values. This is a stated assumption: it dips just below the
code's advertised validity floor.

**Zero-padding above 236.7 MHz was considered and rejected as wrong.**
For weak-X-ray, low-efficiency models reionization finishes late and
`T21` is still +13.6 mK at 236.7 MHz (measured; up to 15.6 mK above
200 MHz). Padding zeros would plant a ~14 mK step discontinuity
mid-band. A step is maximally non-smooth, so it would survive any
smooth-mode filter and inflate the retained RMS — the exact quantity
the figure reports — with an artifact indistinguishable from signal.
The models where padding would matter are precisely the
late-reionization ones that cannot be padded.

### Interpolation

Zeus21's native grid is log-spaced in *z*, so in frequency the spacing
runs from 0.77 MHz at the bottom of the band to 4.08 MHz at the top
(at `precisionboost=1`). It never lands on a uniform 1 MHz grid at any
precision setting, so **interpolation is unavoidable**; `precisionboost`
only sets how far it has to reach.

Interpolation is by **cubic spline** (`scipy.interpolate.CubicSpline`)
in log z, where the native samples are equally spaced and the problem
is well-conditioned, applied once at generation time. Linear
interpolation — what the current `load_t21()` does — leaves curvature
discontinuities at every node, and in an analysis about what survives
projection onto a smooth basis those kinks are exactly the non-smooth
structure that survives.

### Precision setting

`precisionboost` controls the native grid: it scales both the number of
redshift samples and the number of smoothing radii `NRs`. Measured on
8 cores, Pop III enabled, `zmin=4.68`:

| `precisionboost` | s/model | `NRs` | n_z | native dnu at 250 MHz | 4096 models | wall-clock, 8 cores |
|---|---|---|---|---|---|---|
| 1 | 0.33 | 45 | 102 | 4.08 MHz | 0.37 core-h | ~3 min |
| 2 | 2.59 | 90 | 203 | 2.05 MHz | 2.95 core-h | ~22 min |
| 3 | 8.25 | 135 | 303 | 1.37 MHz | 9.38 core-h | ~70 min |

Cost grows superlinearly (1x, 7.8x, 25x against a 1x, 4x, 9x growth in
n_z x NRs), implying something quadratic in `NRs`.

**Production runs at `precisionboost = 3`.** Seventy minutes, once, for
a data product the paper cites is not a meaningful constraint, and it
buys a native spacing of 1.37 MHz against the 1 MHz target at the top of
the band. The spline is then resampling a nearly-matched grid rather
than reaching across 4 MHz gaps, which makes residual interpolation
error negligible rather than merely bounded.

## Post-generation selection

Broad priors produce models in which reionization is incomplete at
z ≈ 5, which the Lyα forest rules out. This cut is necessarily
**posterior** — `xHI(z)` only exists after the model runs — so the
order of operations is generate → cut → subsample.

The cut is applied **once, to the whole ensemble**, and both the quoted
statistics and the drawn curves use the survivors. Applying it to the
figure but not the numbers would leave the two disagreeing, which a
referee would find.

- **Threshold:** `xHI(z = 5.9) < 0.1`. The McGreer et al. 2015
  dark-pixel limit is `xHI < 0.06` at z = 5.9; 0.1 is deliberately
  looser, so the cut removes only models that are unambiguously
  excluded rather than models sitting near the limit. Because `xHI` is
  stored per model, tightening it to 0.06 later costs nothing.
- **Reported:** the number and fraction of models cut.

This remains consistent with the broad-priors choice. It is not an
observationally-informed *prior*; it is one hard, uncontested
observational fact.

Because the cut is posterior, per-model `xHI(z)` is **stored in the
output**, so the threshold can be retuned — or a referee's preferred
value adopted — without regenerating anything.

## Output format

`models_21cm/output/zeus21_models.npz` (gitignored; Zenodo at
publication, following the instrument paper's npz→Zenodo convention).

| key | contents |
|---|---|
| `freqs_MHz` | the paper grid, 50–250 MHz at 1 MHz |
| `T21_mK` | `(n_model, n_freq)`, spline-interpolated onto `freqs_MHz` |
| `T21_native_mK`, `z_native` | uninterpolated output, so the ensemble can be regridded |
| `xHI`, `z_xHI` | per-model neutral fraction, for the retunable cut |
| `params`, `param_names` | `(n_model, 14)` and names, in header column order |
| `provenance` | JSON string; the self-describing header, below |
| `generator_source` | literal text of `priors.py` + `generate.py` as run |
| `env_lock` | literal text of `models_21cm/uv.lock` |
| `regenerate_recipe` | human-readable instructions, readable without this repo |

Written with `np.savez_compressed` (~18 MB uncompressed: 4096 x 201
gridded, 4096 x 303 native, plus `xHI`).

Storing `T21_mK` pre-interpolated reduces `load_t21()` to a load, a
grid assertion, and a mK→K conversion, and removes the GHz/MHz trap in
the current loader. Non-finite or failed draws are dropped before
writing, with the count and the offending parameters recorded.

### The self-describing header

This ensemble will be referenced by future papers and proposals, not
just PR #5. The failure mode being fixed is precisely that the *file
outlived the script*, so the requirement is stronger than recording
metadata:

> Someone holding only this npz, with no access to this repository,
> must be able to regenerate it.

That is why `generator_source` and `env_lock` are embedded rather than
referenced. Tens of kB against an ~18 MB file is a trivial cost, and it
is the difference between a file that *describes* its provenance and
one that *carries* it. A git SHA is a pointer, and pointers are what
failed last time.

`provenance` is a JSON string (not a pickled dict — it must be readable
without `allow_pickle`) containing:

- **When and where:** `created_utc`, `hostname`, `platform`,
  `python_version`, the invoking command line.
- **Code identity:** Zeus21 remote URL, commit SHA, and dirty flag;
  the same three for this repository; the generator script path.
- **Package versions:** `numpy`, `scipy`, `classy`, `astropy`, `mcfit`,
  `powerbox`, `pyfftw`.
- **Full Zeus21 configuration:** every `User_Parameters`,
  `Cosmo_Parameters_Input`, and fixed `Astro_Parameters` keyword as
  actually passed — including `precisionboost = 3`, `zmin_CLASS = 4.5`,
  `zmin = 4.68`, `USE_POPIII`, `USE_LW_FEEDBACK` — not just the ones
  that differ from defaults, since defaults drift between versions.
- **Sampler:** kind, scrambling, seed, `m`, `n_models`, and the
  `varied` list of `{name, transform, lo, hi}` in the column order of
  `params`.
- **Interpolation:** method, variable, and target grid.
- **Selection:** the recommended cut and its reference, recorded as
  *not applied* to the stored arrays.
- **Citations:** the papers a user of this file owes.

## Components

```
models_21cm/
  README.md          what the ensemble is, how to regenerate, what to cite
  pyproject.toml     standalone uv project (NOT a workspace member);
                     zeus21 pinned by git SHA, plus classy/powerbox/pyfftw
  priors.py          pure: PARAMS spec (name, log/linear, lo, hi) + Sobol sampler
  generate.py        CLI; one cosmology setup, multiprocessing over samples,
                     resumable batch checkpoints, writes output/zeus21_models.npz
  select.py          pure: applies the xHI cut and the seeded figure subsample
  test_priors.py     pure unit tests, run in the main env
  output/            gitignored
```

`priors.py` and `select.py` import no Zeus21, so their tests run under
the normal `uv run pytest` — the same split `horizon_position/` uses for
`positions.py` / `masks.py`. `generate.py` gets `run_sims.py`-style
resumable checkpoints, so a draw that hangs or blows up does not cost
the whole run.

Zeus21 pulls in `classy`, which compiles CLASS from source; it has no
place in the mock_analysis workspace lock or in CI. The generator
therefore gets its own pinned environment, following the two-environment
pattern already documented in `horizon_position/CLAUDE.md` for
`eigsep_terrain`. The root `pyproject.toml` will likely need
`exclude = ["models_21cm"]` under `[tool.uv.workspace]` so uv does not
trip over a nested project that is not a member.

## Validation

1. **`priors.py` / `select.py` unit tests** (main env): sample shape,
   bounds respected, seed reproducibility, log/linear transform
   round-trip, cut applied to the right axis.
2. **Fiducial regression** — default parameters reproduce Zeus21's own
   tutorial global signal, confirming the wrapper drives the code
   correctly.
3. **Ensemble sanity** — all `T21` finite; trough depths span a range
   consistent with the literature; models that reionize approach
   `T21 → 0` at the top of the band.
4. **Interpolation error budget** — run ~20 models at
   `precisionboost = 4` (~18 s/model *extrapolated* from the measured
   scaling, so a few minutes for the subset; confirm before relying on
   it),
   treat as truth, and compare the production `precisionboost = 3`
   result against it **in induced error on retained RMS after filtering
   10 modes**, not in mK. That is the only unit that matters here. The
   requirement is that it sits well below the 0.62 mK foreground floor.
   The same comparison run against `precisionboost = 1` quantifies what
   the extra compute bought, and belongs in the README.
5. **Percentile convergence** — recompute the 5/50/95 retained-RMS
   percentiles from random subsets of 256/512/1024/2048/4096. If they
   are still drifting at 4096, extend the Sobol sequence to 8192.
6. **Header sufficiency — machine-checked, not asserted.** A populated
   header rots as silently as an absent one, so sufficiency is tested
   directly:
   - **Sampling half:** rebuild the Sobol draw from the stored
     `provenance` header *alone* and assert bit-for-bit equality with
     the stored `params`. An incomplete or wrong sampler spec fails.
   - **Physics half:** re-run 3 randomly chosen models using only the
     configuration in the header, and match them against their stored
     rows to machine precision. Roughly 25 s at pb = 3.
   Together these prove the header is sufficient rather than merely
   present. This test is the direct countermeasure to the problem that
   motivated the whole project and should be treated as a release gate
   for the npz, not an optional check.

## Downstream changes

`make_paper_horizon_figure.py` imports `load_t21` and `retained_pct`
from `make_paper_signal_loss_figure`, so both figures move together
from a single edit.

- `load_t21()` collapses to load + grid assertion + mK→K.
- `MODELS_NPZ` points at the new file; the `TODO(provenance)` block is
  replaced by real provenance.
- The figure draws a fixed-seed subsample of the survivors for
  legibility (~1000 curves as a starting point; the exact count is
  tuned during figure work, but the seed and count must be recorded so
  the figure is reproducible), while all statistics use the full
  surviving ensemble. The caption must say so.

### The operating point must be recomputed

PR #5's operating point — `N_ANCHOR = 10`, and the table arguing
8 → 10 → 12 — is derived *from the ensemble*: "the smallest N at which
both floors fall below the median retained signal." A new ensemble
changes the median retained signal, so those numbers change and N = 10
may no longer be the answer. `RET_EDGES_MK` and the Spearman
correlations quoted in the PR description are affected the same way.

Recomputing the table and re-justifying the operating point is
therefore **in scope**. Swapping the ensemble without it would leave the
PR asserting numbers its own data no longer supports.

## Assumptions and risks

- **z < 5 is below Zeus21's advertised validity floor.** Accepted, and
  stated in the caption. The alternative (padding) is worse and
  demonstrably so.
- **Broad priors include models inconsistent with UVLFs and CMB tau.**
  Deliberate. Only the reionization-completeness cut is applied;
  other diagnostics are recorded, not enforced.
- **The operating point may move**, and if it moves the paper text
  moves with it. Accepted as in scope.
- **`precisionboost = 3` raises memory per worker.** `NRs = 135` and
  n_z = 303 enlarge the SFRD meshgrids, and `generate.py` forks 8
  workers. Batch size may need tuning, and the resumable checkpoints
  exist partly so an OOM does not cost the run.
- **Cost scales superlinearly in `precisionboost`**, so extending the
  Sobol sequence to 8192 at pb = 3 is a ~2.3 hour job, not a ~70 minute
  one. Still affordable, but worth knowing before promising it.

## Citations

- Muñoz 2023a, arXiv:2302.08506 — Zeus21, required.
- Cruz et al. 2024, arXiv:2407.18294 — Pop III, LW feedback, relative
  velocities. Required because Pop III is on.
- McGreer et al. 2015 — dark-pixel `xHI` limit behind the cut.
- Planck 2018 (Aghanim et al.) — the fixed cosmology.
- A link to the Zeus21 GitHub repository, per its citation policy.
