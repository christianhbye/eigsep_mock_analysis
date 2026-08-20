# models_21cm

A 4096-model ensemble of global 21 cm signals generated with
[Zeus21](https://github.com/JulianBMunoz/Zeus21), used by the EIGSEP
instrument paper's signal-loss figure.

It replaces `normalizing_flows/models_21cm.npz` (1135 models, Aug 2022),
which had no surviving generating script and therefore could not be cited.

## The ensemble

- **4096 models drawn** (scrambled Sobol, `m = 12`, seed `20260819`).
  **0 failed draws** — every sampled parameter combination produced a
  finite Zeus21 run.
- **1782 survive the reionization cut** (43.5%) — see "The reionization
  cut" below for exactly what that cut is and why it has two parts.
- `T21_mK` spans **−262.9 to +38.3 mK** across the whole ensemble;
  per-model minima (trough depths) range **−262.9 to −12.1 mK**.
- Native Zeus21 grid: **304 points over z ∈ [4.65, 35.0]**. Output grid
  (`freqs_MHz`): **50–250 MHz at 1 MHz, 201 points**, built with a cubic
  spline in log z (see "Critical conventions" in `CLAUDE.md` for why not
  `np.interp`).
- `output/zeus21_models.npz` is **22 MB** compressed.
- Pinned environment: Zeus21 `0.1.dev0` @ commit `9f2d210`, `classy`
  `3.3.4.0`, `numpy` `2.5.2`, `scipy` `1.18.0`. The complete lock (every
  package, exact version) is stored verbatim in the npz's `env_lock` key
  and in `models_21cm/uv.lock`.

## The reionization cut

Applied once, in `selection.reionized_across_band`, requiring **both**:

- `xHI(z=5.9) < 0.1` — the McGreer et al. 2015 dark-pixel reference point.
- `xHI(z=4.6816) < 0.05` — the band's *top* edge (`z(250 MHz)`).

The reference-redshift check alone keeps 1812/4096 (44.2%). The band-top
check removes 30 more models that pass at z=5.9 and then re-neutralize
before 250 MHz; 27 of those 30 carry more than 1 mK of residual signal at
250 MHz (worst case 4.76 mK) if the band-top check is skipped. Together,
**1782/4096 (43.5%) survive.**

**The `0.05` threshold is a choice, not a derivation.** It was selected
from a small table of candidates:

| criterion | survivors | max \|T21(250 MHz)\| among kept |
|---|---|---|
| monotonicity of xHI below z=5.9 | 1771 | 1.031 mK |
| `xHI(4.6816) < 0.05` (adopted) | 1782 | 0.963 mK |
| `xHI(4.6816) < 0.01` | 1769 | 0.103 mK |

All three agree to within 0.2% on every reported statistic (retained-RMS
percentiles, `N_ANCHOR`, the above-floor fraction). Per-model `xHI` and
`z_xHI` are stored in the npz precisely so this threshold can be retuned
— or replaced with the monotonicity criterion — **without regenerating
anything.**

The adopted threshold's margin against `verify_ensemble.py`'s `< 1.0 mK`
gate is thin (0.963 vs. 1.000 mK). If that margin is ever a concern, the
`< 0.01` variant gives 5x more headroom at the cost of 13 more excluded
models (1782 → 1769) and changes no reported statistic beyond the third
decimal.

**Why the band-top check exists at all.** Zeus21 solves reionization as
an ionized-fraction ODE, `dQ/dt = ndot_ion - Q/t_rec`, integrated with a
fixed clumping factor. At very low escape fractions the recombination
term outruns the ionizing supply, so `Q` — and hence `xHI` — can rise
again after peaking: the model IGM re-neutralizes. **This is a known
limitation of simple Q-based reionization models, not a bug in this code
and not a numerical artifact.** It was confirmed by re-running the worst
offenders at `zmin = 4.65`, `4.0`, and `3.4` and getting identical
`xHI(z)`/`T21(z)` at fixed physical redshift.

One more honest caveat: a model that *genuinely* completes reionization
near z≈5 legitimately still carries ~1 mK of signal at 250 MHz. The
`< 1.0 mK` sanity bound in `verify_ensemble.py`'s gate is itself a chosen
threshold, not a physical law — it is calibrated to the cut adopted
above and would need revisiting if that cut changes.

## The output carries its own provenance

`output/zeus21_models.npz` embeds everything needed to regenerate it —
a JSON `provenance` header, the `generator_source` verbatim, and the
`env_lock`. A git SHA is a pointer, and a pointer is exactly what failed
last time.

    import numpy as np, json
    d = np.load("models_21cm/output/zeus21_models.npz")
    header = json.loads(str(d["provenance"]))
    print(str(d["regenerate_recipe"]))

## Two environments

The pure modules (`priors.py`, `selection.py`, `provenance.py`) import no
Zeus21 and are tested in the main workspace env:

    uv run pytest models_21cm/ --ignore=models_21cm/test_zeus21_fiducial.py

`generate.py` and `verify_ensemble.py` need the pinned env, which carries
`classy` and therefore compiles CLASS. It is deliberately **not** a uv
workspace member:

    uv sync --project models_21cm
    uv run --project models_21cm python models_21cm/generate.py --help

## Regenerating

    uv run --project models_21cm python models_21cm/generate.py \
        --n-log2 12 --precisionboost 3 --seed 20260819 \
        --out models_21cm/output/zeus21_models.npz

~4 hours at 3 workers on the production run. **Worker count is bounded
by memory, not cores:** each worker needs ~2.78 GB of private peak RSS
at `precisionboost = 3` (plus 0.47 GB shared via fork). An 8-worker
attempt on a 15 GB machine was OOM-killed immediately; 3 workers
(~8.8 GB total) is what worked. Size `--processes` to
`(available_GB - 1) / 2.8`, **not** by core count.

Resumable: rerunning the identical command skips completed batches. The
work directory is keyed on seed and batch size as well as model count and
precision, so a resume with different flags cannot silently reuse stale
batches.

Then the gates, which must all pass before the npz is used or archived:

    uv run --project models_21cm python models_21cm/verify_ensemble.py \
        models_21cm/output/zeus21_models.npz

On the current `output/zeus21_models.npz` this takes about 5.5 minutes
and prints, among other things:

- **Header rebuilds the Sobol draw bit-for-bit**, and 3 models re-run from
  the header alone match their stored rows exactly: `max |diff| = 0.00e+00
  mK`.
- **Interpolation error budget:** max induced retained-RMS error
  `0.0458 mK`, against the `0.62 mK` foreground floor the paper figure
  uses at `N = 10`.
- **Percentile convergence:** the retained-RMS percentiles at `N = 10`
  converge by a genuine 1024-model subsample compared against the full
  1782-survivor set.
- `All gates passed.`

## Operating point in the paper figure

The figure's `N_ANCHOR` (how many leading foreground modes get filtered
before comparing signal to residual) came out to **`N = 10`** on this
ensemble — unchanged from the old one, but for a different reason. At
`N = 10`, retained RMS across the 1782 survivors is p5/p50/p95 =
**0.773 / 2.176 / 7.821 mK**, and **98.1% of models retain more signal
than the 0.62 mK foreground residual** (the old, uncited ensemble gave
69%).

`N_ANCHOR` is the smallest `N` at which *both* the foreground residual
and the worst-case position systematic fall below the median retained
signal. At `N = 9` the position systematic (3.20 mK) is still above the
3.01 mK median even though the foreground floor alone (1.82 mK) is
already below it — it's that conjunction, not the foreground alone, that
pushes the anchor to `N = 10`.

## What to cite

- Muñoz 2023a, [arXiv:2302.08506](https://arxiv.org/abs/2302.08506) — Zeus21.
- Cruz et al. 2024, [arXiv:2407.18294](https://arxiv.org/abs/2407.18294) —
  Pop III, Lyman-Werner feedback, relative velocities.
- McGreer et al. 2015 — the xHI limit behind the reionization cut.
- Planck 2018 — the fixed cosmology.
- A link to the Zeus21 GitHub repository, per its citation policy.

## Design

`../docs/superpowers/specs/2026-08-19-zeus21-model-ensemble-design.md`
