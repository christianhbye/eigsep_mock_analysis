# C51 / notebook 057 — compact-source content and a fixed direct-source pilot

**Status:** author revision after C48's one independent physics review,
2026-11-07. Root allocated C51 and reserved notebook 057; no branch, notebook,
fit or simulation has been started. **Product label: diagnostic.** This
document specifies executable work in two phases with a source-accounting
gate between them. It does not authorize a terrain-fringe calculation.

## 1. Question, scope and decision

Does the exact sky model used by canonical 034 retain the catalogue-strength
compact sources, and does correcting Cyg A's direct-source content improve
prediction of the unchanged July 17 night data? Cyg A is the primary target. Account
for Vir A, Cas A, Tau A and Her A so that partial or already-present sources
cannot be counted twice; do not add every named source automatically. The primary
correction changes **Cyg A only**. The other sources receive a flux budget
and remain unchanged in that primary comparison. A combined correction is
a separately labelled fixed secondary variant only if its source budgets
are externally resolved and frozen before scoring; it cannot replace the
primary because it scores better. In equations(3)–(8), the primary source
sum therefore runs over`{Cyg A}`.

The saved C48 aperture/peak inspection motivates this test, but has not
measured a zero retained Cyg A flux. The planning direct scale is about
2.2 K times directivity per 10^4Jy at 109MHz, against034's approximately 0.83K
per-bin `sigma_tot`. The training fit may absorb most of that shape. This
task measures that absorption and the held-out prediction, rather than
declaring detectability from the unprojected temperature alone.

No free source amplitude, source position, diffuse-background coefficient,
geometry, beam or calibration scale is added to the night fit. Sky-map
content is estimated from the external map, not from the night residual.
No new data cut, inpainting model, coherent cross term, reflected-source
column, absolute calibration or physical fringing identification is in scope.
Existing inpainting is useful where it is part of a declared estimator;
its signal/noise transfer must be measured and original support retained.
The Sun was below −10 degrees over the frozen night; exclude it here. Moon
emission/occultation and reflected RFI require separate inputs and are not
added on the strength of this planning note.

Positive means a fixed source-content correction merits incorporation into
the sky-column generator after review/canonical checks. A negative parks
that correction as an explanation of the tested residual, not the physical
existence of the source. Ambiguous source content, nuisance absorption or
indecisive controls gives an explicit inconclusive follow-up.

## 2. Frozen inputs and gates

1. Canonical034 `data_v0025.npz`, `fits.npz`, `results.json`, `columns/`,
   `PROVENANCE.json`: frequency/channel arrays, raw-channel counts/weights,
   calibrated data, fixed masks/wind cut, `sigma_tot`, train/test block IDs,
   nominal beam, loss draws and saved bin rotations. Verify input hashes
   against this product, not whatever the team tree has since replaced.
   The map source SHA256 is
   `746c45f905a888676df4036e78019db076e3eb3f861d66fdbe73b953e41a78c4`.
2. Recreate the actual map call:
   `GlobalSkyModel16(freq_unit='MHz', data_unit='TCMB', resolution='hi',
   include_cmb=False)`; the high-resolution output is nside 1024 and034
   degrades it to nside 64, then nside 8 where its operator uses it. Assert
   coordinates/order, units, pixel area, degradation settings and map hash.
   Inspect `023_eigsim_columns.gsm_maps` and the exact native dependency pin.
3. Retrieve source-specific catalogue coefficients, flux units, valid
   frequency range, epoch, covariance/scale error, source extent and
   polarization bounds. Start with Perley & Butler 2017,
   [doi:10.3847/1538-4365/aa6df9](https://doi.org/10.3847/1538-4365/aa6df9)
   ([arXiv:1609.05940](https://arxiv.org/abs/1609.05940)); use Baars et al. 1977
   only within its actual source-specific range and epoch convention.
   Copy coefficients from the retrieved table with table/page and source
   file hash, not from memory or a software default. Cas A's secular change
   must be evaluated for the 2026 night with its uncertainty, or its
   catalogue-replacement result remains conditional. A common flux-scale
   error is correlated across frequency; do not turn it into independent
   channel noise. Missing covariance gets an explicit documented systematic
   envelope, not an invented posterior.
4. Use034's saved nominal rotation/beam for the controlled primary test.
   This is its IMU-tilt/v5-bin convention. Pointingv6 and the new HFSS1 MHz
   run are separate science changes, not prerequisites for phase 1. Any
   later upgrade needs its own reproduction bridge before source scores.

Catalogue positions are ICRS. Transform them using the same frozen 034
Galactic-to-true-ENU sky rotation and body rotation as the diffuse map;
record the frame convention and compare with an independent UTC Astropy
direction check. Do not copy the paper notebook's naive-time/manual-offset
recipe. Convert true/grid bearings only where the horizon operator requires
it, using the pinned MOD-95 convention. All frequencies passed to(1) are Hz;
saved output axes retain the D5 MHz/native-channel contract.

**Execution gates.** Phase 1 may start now from the frozen inputs; retrieving
and pinning the catalogue is part of it. Phase 2 waits until the external
retained-flux/shape budget and valid catalogue inputs are frozen. It also
requires a supported frequency/beam operator for its matched comparison.
Canonical041, a bistatic kernel and reflected-source ray tracing are not
needed. No missing gate can be filled by inspecting the night residual.

**Native support gate (CHB).** Preserve exact source-file/hash, row, native
frequency/channel and `time_best` identity; original measured/filled/excluded
masks and reason bits; calibration availability; per-cell estimator weights;
and every mapping from native cells to reported bins. Retain measured raw
values by immutable input reference and hash, with any model-filled value
identified separately. Export native masks even when the fit uses binned
data. Counts, effective fractions and aggregate coverage supplement rather
than substitute for them. A missing original mask is a recovery gate, not
permission to infer that a pixel was measured from its finite value.

## 3. Phase 1 — source accounting in the exact map

Let`q_cat,k(nu,Omega)` be a source profile normalized by
`sum_p q_p*Omega_p=1` (units sr^-1). It must include the effective resolution
and the same pixel/degradation response as the map being inspected.
For thermodynamic map units, define

    g(nu) = x^2 exp(x)/(exp(x)-1)^2,  x=h*nu/(k_B*T_CMB),
    C_nu = 1e-26*c^2/[2*k_B*nu^2*g(nu)]       [K_CMB sr / Jy].       (1)

This follows from`dB_nu/dT=(2*k_B*nu^2/c^2)*g` and one Jy's SI value.
Use the actual pygdsm conversion/constants for bit-level map closure;
cross-check(1) and its Rayleigh–Jeans limit. Do not subtract an extra CMB
from a map whose loader already used`include_cmb=False`.

At each frequency and resolution, fit the **map patch only**, in common
intensity or map units, with source profile plus local diffuse background:

    M_p = C_nu*S_ret,k*q_ret,k,p + B_p.                            (2)

The retained profile need not be a delta function. Pin the source response
from the parent maps/model construction, or bracket plausible smoothing
responses. Do not fit an arbitrary width/position until it erases the
distinction between a compact source and Cygnus X. If the response cannot
be bounded independently, report a retained-flux interval and stop short
of a unique correction.

Predeclare three map-only background checks before loading night residuals:
the original 3 degree aperture/4–8 degree annulus; a constant-plus-gradient
background on that patch with bright neighboring compact regions masked;
and an expanded local fit out to 10 degrees with quadratic background.
The compact response uses a 5 degree support initially; show its encircled
flux and extend it before freezing if its known response extends farther.
Record the chosen masks/profile from external maps/coordinates. For Cygnus X,
report the variation across these prescribed background choices, not just
a fit covariance. The annulus statistic is a diagnostic cross-check, not a
proof that `S_ret=0`.

If the formal retained-flux estimate is negative, retain that fitted result
as a background-model warning. For a physical source budget, report the
nonnegative compatible range and its upper bound; do not silently clip the
central value and call the entire catalogue missing. If background/profile
choices leave no common compatible range, phase 1 is inconclusive.

Freeze, per source/frequency, `S_cat`, its epoch/spectral uncertainty,
`S_ret`, the retained profile, the background/profile envelope and
`delta_S=S_cat-S_ret`. Do this for Cas A/Tau A too, even when the intended
correction is zero. The correction is removal plus replacement:

    M_corr = M - sum_k C_nu*S_ret,k*q_ret,k
                 + sum_k C_nu*S_cat,k*q_cat,k.                     (3)

Show the removed map, replacement and net correction separately. A retained
broad map feature and a true compact source can couple differently to the
beam; their correction is not always just `delta_S*D(source)`.

**Acceptance before phase 2.** Reproduced maps and degradation agree with the
stored034 map to numerical precision; source-profile flux is conserved to
1% through normalization/degradation; injected known compact profiles are
recovered to 1% on each declared background model in noiseless closure tests.
Those synthetic tests validate the operator, not a real Cygnus X background.
Report external map/profile uncertainty as a systematic envelope. Freeze
one primary external estimate and its envelope before scoring; if no
defensible primary estimate exists, stop with the interval and queue the
specific missing map/profile input. No catalogue addition is then adopted.

## 4. Phase 2 — direct columns through a matched estimator

The direct antenna-temperature operator for normalized power beam b is

    C_dir[M](nu,t) = integral b(nu,n,t)*V(n,t)*M(nu,G(t)^T*n) dOmega,
    integral b dOmega = 1,  D=4*pi*b.                               (4)

`V` is the frozen skyline visibility and`G` rotates Galactic coordinates
to true topocentric ENU. The source's receive direction is transformed by
the **actual saved bin rotation**. In the unresolved Rayleigh–Jeans limit,

    C_dir,k = 1e-26*S_cat,k*lambda^2*D(s_k,t)*V(s_k,t)/(8*pi*k_B).    (5)

Equation(5) is a normalization check. The operational column uses the same
map-unit convention as034 via(1)/(4); print its ratio to the RJ limit.
Keep the baseline map interpretation unchanged in both models. Report the
small thermodynamic/RJ conversion difference rather than silently mixing
the two conventions. Normalize on the full sphere, without renormalizing
over only visible sky. Bound finite source extent and polarization before
using the unpolarized point approximation; omit a source only with its
maximum possible omitted direct contribution stated beside `sigma_tot`.

Make two explicit bridges:

- **Historical reproduction:** exactly recover034's stored centre-frequency
  `S`, `f`, rotation and predictions before changing source content. Do not
  rerun an unrelated upstream fit or silently replace a historical tilt file.
- **Matched comparison:** apply the same native offset-channel and time/bin
  operator to the original and corrected maps. With raw count weights
  `n_bjk`, the channel part is

      E_bj[Z] = sum_k n_bjk*Z(nu_jk,t_b)/sum_k n_bjk.                (6)

  Use the actual supported offsets/counts, including truncated endpoint
  support, rather than an assumed six equal weights. `t_b` and its saved
  pointing are034's bin approximation; retain it identically in both
  candidates and quantify its direct-source time-averaging error before
  adoption. If actual native times/pointing are used instead, rebuild both
  baseline and candidate through that operator and name it a separate bridge.

Use the existing beam's documented frequency evaluation/interpolation; no
new unconstrained spectral basis or waiting for the acme2 beam. Validate
frequency interpolation on held-out existing beam slices, and propagate its
column error. If the present beam cannot support native-offset evaluation,
stop at phase 1 or a clearly labelled centre-frequency diagnostic; that
diagnostic cannot pass the matched-estimator adoption gate. This limitation
does not justify comparing an averaged candidate with an unmatched baseline.

Define`A0=E[C_dir[M]+T_CMB*(1-f)]` and
`A1=A0+E[C_dir[M_corr-M]]`. Ground fraction and thermal-ground treatment
are identical. Each model predicts the calibrated data as

    m_v,bj = theta_v,j*A_v,bj + P_v,j
             + (1-theta_v,j/Gbar_j)*cdev_bj,                       (7)
    cdev = c_L - weighted_mean_train(c_L).

Use the same`Gbar`, loss draws and load pivot in both models. The source
column is at the free-space antenna plane and receives the existing common
`theta=k*G` once. Do not apply an S11 round-trip transfer, an additional gain
or a new source temperature-calibration factor.

Fit theta/P and load coefficient using **training rows only**, including
all centering and each jackknife refit. Freeze predictions on held-out rows.
The existing`iterate_load_term` may be used with training `rows`; require
convergence. Check it independently against the exact fixed-Gbar linear
rewrite `m=theta*(A-cdev/Gbar)+P+cdev`. Match the documented solver tolerance;
stop scoring any frequency whose iteration or rank check fails, reporting
its coverage rather than quietly dropping it for a favorable result.

### 4a. Inpainting and source-power transfer for any delay comparison

The primary calibrated 034 comparison retains its actual measured-data
mask/weight/calibration operator. Do not add a new DPSS step to that fit.
If a source template, source-power estimate or injection is compared with
056 or another inpainted residual, use **that** product's complete operator.
The source absent from the fill model is also absent from its filled pixels.

For a frozen model M and good-cell mask g, deterministic fill satisfies

    P_fill = g*P+(1-g)*M,    r_fill=P_fill-M=g*(P-M).                (6a)

Thus source contribution removed by flags is not restored by inpainting.
Build the per-row BH7-weighted mask response before frequency transformation;
use the actual row-total normalization, DPSS model absorption and any SVD
projection, then the same exposure-weighted time-power estimator as the data.
Do not assume a diagonal delay response. Preserve native support through
each operation, including uncertainty/noise correlations. Filled values may
support a transform; they add no independent measured-sky likelihood weight.
Calibrator/switch visits are excluded sky intervals and remain so; do not
time-inpaint them into observations.

Run paired, small physical-source injections before and after that complete
pipeline, with the same measured masks and external source budget, to measure
the transfer. Freeze trained operators when that is the real estimator; if
the estimator relearns DPSS/SVD, repeat that relearning in the injection and
report its absorption separately. Include normalization changes. A fixed
linear approximation may summarize the response only after agreement with
the full operator is demonstrated. Keep multi-delay/source coupling and
frequency/time correlations rather than a single global efficiency.

As a **limiting check** only, a fixed-normalization independent random mask
gives coherent amplitude attenuation approximately`1-f_eff`, for the
corresponding window-weighted missing fraction. Its coherent power scales
approximately as`(1-f_eff)^2`. Mask leakage variance and noise power remain
in the measured squared amplitude, so the same scalar is not a correction
for total delay power. It is especially unsuitable for structured
integer-MHz masks. Never divide by the nominal unweighted flag fraction.

Validate recovery and its uncertainty on withheld real unflagged samples
and disjoint real time/frequency support, using representative artificial
holdout masks before inspecting the source statistic. Show injected-source
recovery with the real structured masks, alongside zero-source/noise transfer,
and report radiometer expectation versus actual held-out scatter. Withheld
real samples remain identifiable as measured truth and cannot enter the
model training/fill that predicts them. Report singular values/condition
and supported delay combinations of any correction/inversion. If attenuation
cannot be stably inverted within the retained source/response uncertainty,
report an attenuated conditional quantity or bound and an inconclusive
outcome. No corrected source power is claimed from an unstable inverse.

## 5. Score, controls, absorption and outcome

Freeze data, `sigma_tot`, masks, wind cut and034's1,800 second block origin.
Even blocks train and odd blocks test, using the stored complete-bin time
axis to define the origin. Count the actual test blocks. No new noise-floor
fit may lower the primary loss. For model v, the **summed** loss is

    Q_v = sum_(heldout b,j) [(y_bj-m_v,bj)/sigma_bj]^2,
    d = Q_1-Q_0.                                                   (8)

Negative d favors the correction. Also print per-cell mean losses for
readability, clearly labelled; never mix them with summed-loss errors.
Keep per-frequency and per-test-block differences and coverage. Refit on
each delete-one-block training subset and score the corresponding remaining
test subset, repeating load centering/iteration. For replicate differences
`d_(-l)`, use

    SE_J = sqrt[(L-1)/L * sum_l(d_(-l)-mean_l d_(-l))^2].            (9)

Resample whole time blocks jointly across frequencies for a10,000-draw paired
block bootstrap; report it beside the jackknife. Empirical block scatter is
the uncertainty reference, not an independent radiometer-noise likelihood.
The thresholds below are inherited operational criteria, not a calibrated
detection probability.

**Controls:** rotate the correction's source hour angles by ±1/±2/±3 sidereal
hours in the continuous celestial-to-topocentric transformation. Keep the
actual bin pointing, horizon, data, masks, channel weights and diffuse
baseline fixed; shift both removed and replacement source components.
Use the same externally frozen spectra, template support and theta/P/load
freedom. Re-evaluate the direction, not an array-index roll. This isolates
source timing without shifting the antenna or selecting different data.

**Primary pass rule:** retain MOD-99's implemented criterion: the correction
wins every test block or`d < -2.33*SE_J`; it must also beat **each** timing
control by more than1 SE of that paired summed-loss difference. Explicitly
pass`multiples=(2.33,1.0)` if using`skyground.block_test`; its helper has the
opposite improvement-sign convention. This C51 rule does not silently adopt
G2's proposed 1 SE operational variant. Report zero/near-zero error ties.

**Sensitivity and nuisance absorption:** compute the correction's actual
train-fit absorption before reading its held-out score. Project its
prediction change onto the same theta/P/load nuisance design on training
rows and propagate the fitted nuisance change to test rows. For a linear
check with prediction perturbationu and nuisance designN,

    beta_abs = (N_tr^T W_tr N_tr)^+ N_tr^T W_tr u_tr,
    u_surv,te = u_te - N_te*beta_abs.                               (10)

Report the training and test weighted norms before/after absorption, design
rank and correlations with the sky/load columns. Confirm any detectability
claim with the **actual** train/refit/held-out operator at fixed physical
flux, including load iteration. A bounded deterministic prediction/recovery
or empirical-block injection check is sufficient; any synthetic generator
lives in mock_analysis. Compute expected `Q0-Q1` on that same scale and
compare with the empirical control/block loss scatter. Do not compare an
unprojected SNR with a standard error of Q or claim broad physical exclusion
from a weak test. A material nonlinear disagreement with(10) makes the exact
refit response authoritative, and the discrepancy must be shown.

Score the predeclared external-flux/background/beam sensitivity envelope
without choosing its best member on test data. A primary improvement that
reverses under an independently admissible source budget or beam-error
envelope is inconclusive for adoption. A winning sum identifies only that
combined prediction; show fixed source-removal ablations before attributing
an improvement to Cyg A alone. No per-source amplitude fit is added.

**Outcome:** positive → proposed adoption only if source accounting, matched
estimator closure, held-out rule/controls and material uncertainty checks
pass; carry the fixed correction into the relevant sky-column generator and
downstream fit after review/merge/canonical rerun. Negative → park this
audited direct correction as the explanation of the tested residual only
when the fixed prediction is informative but fails the rule/controls.
Inconclusive → queue the specific source-content, response, covariance or
power limitation. Controls that tie identify no alternative cause. No result
here rejects or establishes a coherent terrain fringe.

## 6. Assumptions, known limitations and falsifiable checks

| ID | Assumption or limit | Check or consequence |
|---|---|---|
| C51-A1 | The pinned map and retrieved catalogue represent this epoch within their declared errors | Recover coefficients/epoch; bound variability, especially Cas A; no unsupported extrapolation |
| C51-A2 | A retained compact profile can be separated from diffuse emission locally | Prescribed background/profile sensitivity; interval/inconclusive if not identifiable |
| C51-A3 | Unpolarized point approximation is adequate | Quantify extent/polarization contribution using external information/beam; do not fit night residual to establish it |
| C51-A4 | Frozen034 pointing/beam and native spectral evaluation are adequate | Reproduction bridge and held-out-slice interpolation error; upgrades are separate variants |
| C51-A5 | Fixed-Gbar load formalism is the existing nuisance model | Training-only convergence and exact-linear closure; paired loss-draw sensitivity |
| C51-L1 | Direct-only pilot omits source incoherent reflection and interference | No fringing identification or complete source-power claim; later coherent baseline needs both paths |
| C51-L2 | Map background errors and blocks may be correlated | Systematic budget plus whole-block statistics; no Gaussian source-detection significance |
| C51-L3 | Centre-bin pointing/time is an inherited approximation | Bound direct-column averaging error; if material, rebuild both models with native times |
| C51-L4 | Model-filled channels lack the candidate source and can attenuate/mix delay modes | Preserve exact native identities/masks; measure complete operator response and correction conditioning; no unweighted scalar correction |

Required checks before a real-data score: zero flux/correction returns the
baseline; fully retained catalogue source returns zero correction when its
profile/operator matches; a partially retained source gets exactly the
residual budget; present Cas A/Tau A are not doubled; source hidden by the
horizon contributes zero; an isotropic beam recovers(5); profile integration
agrees with point evaluation in the converged limit; known injected map
profiles and unit conversion close; native weighted averaging is performed
on both models; changing only held-out data cannot change training fits,
load centering or externally determined source flux. Tests must fail under
factor-of-two, wrong-unit, wrong-rotation and held-out-leakage mutations.
For any inpainted comparison, additionally test reconstruction of native
measured/filled/excluded masks after averaging; zero-exposure sky gaps;
withheld-real-data prediction; structured-mask mode mixing; and recovery
and uncertainty/conditioning of the claimed source-transfer correction.

## 7. Code, products and review boundary

Proposed locations, to be registered by root before creation: a small
mock_analysis/eigsim module for compact-source profiles/beam projection and
synthetic closure, with tests; a D5 adapter/scorer in the private code repo
plus `notebooks/057_compact_source_content.ipynb`. Reuse existing sky,
horizon, rotation and `skyground`/`night_fit` solvers. Read both repositories'
instructions and `docs/interface.md`/eigsep_cal API before crossing packages.
No receiver/calibration or file-I/O logic goes into eigsim. Do not build on
the archived standalone rotis or `eigsep_data.sim.GlobalSim`.

Products via`paths.product_dir("057_compact_source_content")`:
`source_budget.csv/json`, catalogue coefficients/citations/hashes,
`map_profiles.npz`, `direct_columns.npz`, `scores.json`, per-block tables,
reproduction/closure checks and `PROVENANCE.json`. Record all masks/weights,
frames, map/beam/rotation hashes, code pins, flux/background envelopes,
decision thresholds and random seeds. Caches must fingerprint the inputs
actually executed. No canonical input is overwritten or branch-draft path
embedded as a permanent requirement.
Include the exact native support masks/reasons/identities and estimator
weights as a retained product or a pinned immutable product dependency;
aggregate support arrays are additional outputs. Preserve any transfer
matrix/operator, injections, held-out-real truth/recovery and conditioning
used to correct source power, with their input hashes and uncertainty.

Notebook figures end each stage: source patch/data-profile/background and
flux budget; zero/partial/full-source operator closure; direct correction
versus time/frequency and absorption; baseline/candidate data/model/residual
against radiometer reference and actual scatter; controls and held-out
block scores; uncertainty/adoption limits. Commit executed outputs and a
register row; a paper/memo result needs that committed notebook.

Light spec/revision only here; implementation gets its own bounded
conformance review. Map generation/repeated column work is
**LIMBO-COMPUTE** with fresh CPU/RAM/I/O checks, nice0, private outputs and
the aggregate cbye cap. It can stop after phase 1 if its gates fail. Push
private computational PRs for CHB; do not ask for a separate workspace-spec
PR. Shared adoption requires the normal graduation/approval packet.

Math to check: (1) map-unit and Jy conversion; (2) flux/profile identifiability
and no double counting in(3); (3) beam/horizon normalization and estimator
parity in(4)–(6); (4) training-only load identity(7); (5) summed-score and
paired jackknife convention(8)–(10). C48's reviewed source-accounting finding
motivates this pilot; its new detailed implementation remains author-scoped
and must be checked by root before execution.
