# models_21cm

A 4096-model ensemble of global 21 cm signals generated with
[Zeus21](https://github.com/JulianBMunoz/Zeus21), used by the EIGSEP
instrument paper's signal-loss figure.

It replaces `models_21cm.npz` from the sibling `normalizing_flows`
project — **outside this repository**, at
`~/Documents/research/eigsep/normalizing_flows/models_21cm.npz` on the
machine this was written on — 1135 models, Aug 2022, which had no
surviving generating script and therefore could not be cited.

## The ensemble

- **4096 models drawn** (scrambled Sobol, `m = 12`, seed `20260819`).
  **0 failed draws** — every sampled parameter combination produced a
  finite Zeus21 run.
- **1769 survive the reionization cut** (43.2%) — see "The reionization
  cut" below for exactly what that cut is and why it has two parts.
- `T21_mK` spans **−262.9 to +38.3 mK** across the whole ensemble;
  per-model minima (trough depths) range **−262.9 to −12.1 mK**.
- Native Zeus21 grid: **304 points over z ∈ [4.65, 35.0]**. Output grid
  (`freqs_MHz`): **50–250 MHz at 1 MHz, 201 points**, built with a cubic
  spline in log z (see "Critical conventions" in `CLAUDE.md` for why not
  `np.interp`).
- `output/zeus21_models.npz` is **22 MB** compressed.
- **`zmin = 4.65` (251.4 MHz) sits below Zeus21's advertised z = 5–35
  validity range.** This is deliberate, so the 250 MHz band edge is
  covered by computed values rather than extrapolation. The alternative
  — zero-padding above Zeus21's native top of range (z = 5, 236.7 MHz) —
  was rejected: late-reionization models still carry up to ~14 mK of
  signal at 237 MHz, and padding zeros there would plant a step
  discontinuity that survives any smooth-mode filter and would inflate
  the retained-RMS statistic the signal-loss figure reports.
- Pinned environment: Zeus21 `0.1.dev0` @ commit `9f2d210`, `classy`
  `3.3.4.0`, `numpy` `2.5.2`, `scipy` `1.18.0`. The complete lock (every
  package, exact version) is stored verbatim in the npz's `env_lock` key
  and in `models_21cm/uv.lock`.

## The reionization cut

Applied once, in `selection.reionized_across_band`, requiring **both**:

- `xHI(z=5.9) < 0.1` — the dark-pixel reference redshift.
- `xHI(z=4.6816) < 0.01` — the band's *top* edge (`z(250 MHz)`).

The reference-redshift check alone keeps 1812/4096 (44.2%). The band-top
check removes 43 more models that pass at z=5.9 and then re-neutralize
before 250 MHz; 27 of those 43 carry more than 1 mK of residual signal at
250 MHz (worst case 4.76 mK) if the band-top check is skipped. Together,
**1769/4096 (43.2%) survive.**

### The z = 5.9 limit has moved, and `0.1` is now the conservative side

The threshold at the anchor redshift was chosen against McGreer et al.
2015, `xHI ≤ 0.06 + 0.05` (1σ) at z = 5.9 from 6 quasar sightlines.
**That measurement has since been superseded.** Davies et al. 2025
([MNRAS 545](https://academic.oup.com/mnras/article/545/2/staf1862/8305915),
[arXiv:2510.25829](https://arxiv.org/abs/2510.25829)) redo it with 34
E-XQR-30 spectra and report *weaker* fiducial 1σ upper limits on
⟨xHI⟩ from the optimally sensitive Lyβ + Lyγ combination:

| z̄ | ⟨xHI⟩ ≤ |
|---|---|
| 5.481 | 0.030 + 0.048 |
| 5.654 | 0.095 + 0.037 |
| 5.831 | 0.191 + 0.056 |
| 6.043 | 0.199 + 0.087 |

They measure a dark fraction of 0.19 at z ≈ 5.8 against McGreer's 0.06 at
z = 5.9, attributing the difference to cosmic variance across the smaller
sample, and conclude that "the bulk of reionization must be finished at
z > 6, while leaving room for an extended *soft landing* … down to
z ∼ 5.4" — consistent with reionization ending by z = 5.3 (Bosman et al.
2022).

**The direction of the move matters more than the number.** The current
limit is looser than the one this cut was built against, so `0.1` is now
roughly half of what the data require: the cut errs toward excluding
histories that are still allowed, never toward admitting ones that are
ruled out. `0.1` is kept rather than relaxed because — see the next
section — the threshold changes no reported statistic.

**The band-top limb is untouched by this**, and if anything better
supported: reionization finishing near z ≈ 5.3–5.4 sits comfortably above
the band's top edge at z = 4.6816, so requiring a kept model to be
reionized *there* is a weaker demand now than when the cut was written.

### Sensitivity to the threshold

Regenerate this section's numbers with:

    uv run python horizon_position/reionization_sensitivity.py

`xHI(z=5.9)` is strongly bimodal, so the threshold sits in a nearly empty
valley: 1765 of 4096 models fall below 0.06, 1812 below 0.10, 1878 below
0.20, and 1925 below 0.25. Only **71** models lie in `[0.1, 0.25)` *and*
pass the band-top limb.

Retained-RMS percentiles [mK] and the above-floor fraction, both quoted
at the paper's `N = 9`; `N*` is the operating point each variant would
select on its own under the same stays-below rule:

| cut | keep | N* | p5 | p50 | p95 | above floor |
|---|---|---|---|---|---|---|
| McGreer+2015 strict, `x(5.9) < 0.06` | 1738 | 9 | 1.066 | 3.023 | 10.513 | 75.6% |
| **adopted, `x(5.9) < 0.1`** | **1769** | **9** | **1.046** | **3.014** | **10.488** | **75.4%** |
| Davies+2025 ladder, fiducial | 1750 | 9 | 1.059 | 3.018 | 10.503 | 75.6% |
| Davies+2025 ladder, +1σ | 1800 | 9 | 1.040 | 2.998 | 10.481 | 75.2% |
| relaxed, `x(5.9) < 0.2` | 1814 | 9 | 1.037 | 2.993 | 10.434 | 75.1% |
| relaxed, `x(5.9) < 0.25` | 1840 | 9 | 1.035 | 2.969 | 10.360 | 74.8% |
| band-top limb only, no z ≈ 6 anchor | 2218 | 9 | 0.884 | 2.828 | 10.136 | 71.7% |

`N* = 9` under every variant, **including dropping the z ≈ 6 anchor
entirely**. Across every variant the current data allow, the above-floor
fraction stays within 0.8 points (74.8–75.6%), and it stays within 3.7
points of the adopted 75.4% even with no anchor at all. The
"Davies+2025 ladder" rows apply all four limits above as a conjunction
rather than testing a single anchor redshift.

**One honest asymmetry.** The 71 models the adopted cut drops but
Davies+2025 would allow retain *less*, not more: p5/p50/p95 =
0.84 / 2.38 / 6.80 mK against the kept ensemble's 1.05 / 3.01 / 10.49 mK,
and 59.2% of them clear the foreground floor against the kept 75.4%. The
adopted cut is therefore marginally *favourable* to the fraction the
paper reports — by 0.6 points at `x(5.9) < 0.25`. Their worst
`|T21(250 MHz)|` is 0.107 mK, well inside `verify_ensemble.py`'s
`< 1.0 mK` gate, so nothing about them is unphysical at the band edge;
they simply reionize later.

**The `0.01` threshold is a choice, not a derivation.** It was selected
from a small table of candidates:

| criterion | survivors | max \|T21(250 MHz)\| among kept |
|---|---|---|
| monotonicity of xHI below z=5.9 | 1771 | 1.031 mK |
| `xHI(4.6816) < 0.05` | 1782 | 0.963 mK |
| `xHI(4.6816) < 0.01` (adopted) | 1769 | 0.103 mK |

All three agree to within 0.2% on every reported statistic (retained-RMS
percentiles, `N_ANCHOR`, the above-floor fraction). Per-model `xHI` and
`z_xHI` are stored in the npz precisely so this threshold can be retuned
— or replaced with the monotonicity criterion — **without regenerating
anything.**

**This project shipped with `0.05` first, then moved to `0.01`.** A
whole-branch review pointed out that `0.05`'s margin against
`verify_ensemble.py`'s own `< 1.0 mK` gate was thin (0.963 vs.
1.000 mK) — and that the closeness was not independent evidence the
choice was sound, since the same gate and the same threshold table came
from the same author. `0.01` was preferred over the monotonicity
candidate specifically because it requires no change to that gate,
unlike monotonicity (1.031 mK, which would exceed it). Moving to `0.01`
costs 13 more excluded models (1782 → 1769, 0.7%) and changes no other
reported statistic beyond the third decimal, for roughly 10x more
headroom against the gate (0.103 vs. 1.000 mK).

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

## `Mc_III` is sampled but has no effect

Of the 14 varied parameters (`priors.PARAMS`), **`Mc_III`** (the Pop III
minihalo mass scale, log₁₀ 5.5–8.0) **is inert**, given the fixed
`astro_fixed` values `alphastar_III = betastar_III = 0`. Zeus21 computes
the Pop III star-formation efficiency as (`zeus21/sfrd.py`,
`fstarofz_III`):

    fstarofz_III = 2 * OmegaB/OmegaM * epsstar_ofz_III /
        ((Mh/Mc_III)**-alphastar_III + (Mh/Mc_III)**-betastar_III)

With both exponents fixed at zero, `(Mh/Mc_III)**-0 = 1` for every halo
mass, so the denominator is exactly `1 + 1 = 2` regardless of `Mc_III` —
the term `Mc_III` was meant to control never enters the calculation.
This was confirmed empirically as well as algebraically: a partial rank
regression of `Mc_III` against trough depth gives `t = -0.10`
(`p = 0.92`), and the marginal `|Spearman|` between `Mc_III` and each of
five summary statistics is at most `0.009` — indistinguishable from the
null a genuinely fixed parameter would give.

**This makes the ensemble effectively a 13-parameter family, not 14.**
No statistic reported in this README, in `verify_ensemble.py`, or in the
paper figure is affected, because none of them ever depended on `Mc_III`
having an effect — the column was sampled but along for the ride. Pop
III's halo-mass dependence is not entirely absent from the ensemble: it
is still varied, through Lyman-Werner feedback (`A_LW`, `beta_LW`,
varied parameters 13-14) acting on Zeus21's `Mmol()` molecular-cooling
threshold mass, which sets the minimum halo mass able to host Pop III
star formation at all. `Mc_III` was meant to be a second, independent
handle on that mass dependence, but is not one in this ensemble.

**No regeneration and no change to the parameter set are planned as
part of this finding** — that decision belongs to the user. The npz's
`provenance["varied"]` entry for `Mc_III` carries a `note` field
recording this (added by `models_21cm/patch_npz_header.py`), so the
caveat travels with the file even without this README.

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
        --out models_21cm/output/zeus21_models.npz \
        --processes 3 --batch-size 64

**`--processes` is not optional — it defaults to 8, which is an
immediate OOM-kill on a 15 GB machine.** Worker count is bounded by
memory, not cores: each worker needs ~2.78 GB of private peak RSS at
`precisionboost = 3` (plus 0.47 GB shared via fork), so 8 workers wants
~22.7 GB. 3 workers (~8.8 GB total) is what worked, at ~4 hours wall
clock. Size `--processes` to `(available_GB - 1) / 2.8`, not by core
count.

Resumable: rerunning the identical command skips completed batches. The
work directory is keyed on seed and batch size as well as model count and
precision, so a resume with different flags cannot silently reuse stale
batches.

Then the gates, which must all pass before the npz is used or archived:

    uv run --project models_21cm python models_21cm/verify_ensemble.py \
        models_21cm/output/zeus21_models.npz

On the current `output/zeus21_models.npz` this takes about 6 minutes and
prints, among other things:

- **Header rebuilds the Sobol draw bit-for-bit**, and 3 models re-run from
  the header alone match their stored rows exactly: `max |diff| = 0.00e+00
  mK`.
- **Reionized models vanish at the top of the band:** `max |T21(250 MHz)|
  = 0.103 mK`, against the `0.01` band-top threshold's own gate of
  `< 1.0 mK` -- see "The reionization cut" above for why this margin
  changed from thin (0.963 mK) to wide.
- **Header drift gate:** the header's recorded `selection.recommended`
  is checked against `selection.py`'s actual constants at verify time, so
  the two cannot silently diverge again the way they did before this
  fix.
- **Interpolation error budget:** max induced retained-RMS error
  `0.0458 mK`, against the `0.62 mK` foreground floor at `N = 10`.
- **Percentile convergence:** the retained-RMS percentiles at `N = 10`
  converge by a genuine 1024-model subsample compared against the full
  1769-survivor set.

`N = 10` in those two gates is a **fixed reference point**, deliberately
not tracking the figure's `N_ANCHOR` (now 9, see below): this file
validates the ensemble, not the figure, and the gates describe the point
they were set at. `verify_ensemble.py` says so at its own constants.
- `All gates passed.`

## Operating point in the paper figure

The figure's `N_ANCHOR` (how many leading foreground modes get filtered
before comparing signal to residual) is **`N = 9`** on this ensemble. At
`N = 9`, retained RMS across the 1769 survivors is p5/p50/p95 =
**1.046 / 3.014 / 10.488 mK**, and **75.4% of models retain more signal
than the 1.82 mK foreground residual**.

`N_ANCHOR` is the smallest `N` at which the foreground residual falls
below the median retained signal **and stays below for every larger `N`**.
Only the foreground residual enters the criterion. The stays-below clause
is load-bearing: both curves fall with `N` and cross more than once, so a
first-crossing rule can select an `N` the floor later climbs back above.

**This moved from `N = 10`.** The earlier anchor was a two-floor
conjunction that also required the worst-case position systematic to fall
below the median retained signal. That floor was dropped from the figure,
which is scoped to foreground dimensionality alone; the position
systematic is shown against the same benchmark in `horizon_shift.pdf`
instead. `horizon_position/recompute_operating_point.py` re-derives the
anchor independently of any hardcoded constant.

## What to cite

- Muñoz 2023a, [arXiv:2302.08506](https://arxiv.org/abs/2302.08506) — Zeus21.
- Cruz et al. 2024, [arXiv:2407.18294](https://arxiv.org/abs/2407.18294) —
  Pop III, Lyman-Werner feedback, relative velocities.
- Davies et al. 2025,
  [arXiv:2510.25829](https://arxiv.org/abs/2510.25829) — the current
  dark-pixel `xHI` limits the reionization cut is justified against.
- McGreer et al. 2015 — the earlier dark-pixel limit the cut's threshold
  was originally chosen against, superseded by the above.
- Bosman et al. 2022,
  [MNRAS 514, 55](https://ui.adsabs.harvard.edu/abs/2022MNRAS.514...55B/abstract)
  — reionization ends by z = 5.3, cited for the band-top limb.
- Planck 2018 — the fixed cosmology.
- [Zeus21 GitHub repository](https://github.com/JulianBMunoz/Zeus21), per its citation policy.

`pyproject.toml` pins the git source as `github.com/ZeusCosmo/Zeus21`
rather than `github.com/JulianBMunoz/Zeus21` above. These are the same
code under a GitHub organization rename, not two different codebases —
`ZeusCosmo/Zeus21` is where the repository now lives; the citation
policy and paper still point at the `JulianBMunoz` URL.

## Design

Spec: `../docs/superpowers/specs/2026-08-19-zeus21-model-ensemble-design.md`
Plan: `../docs/superpowers/plans/2026-08-19-zeus21-model-ensemble.md`
