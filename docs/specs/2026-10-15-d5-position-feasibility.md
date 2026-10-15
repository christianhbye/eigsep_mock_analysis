# D5 position emulator: numerical feasibility

Status: implementation proposal, 2026-10-15. CHB selected numerical feasibility
with the existing fixed sky/beam first. Numerical thresholds below are proposed
engineering criteria, not an approved D5 science-error budget. No new study has
run. Label: **diagnostic**. Run tag: **LIMBO-COMPUTE**.

Companion: [implementation plan](../plans/2026-10-15-d5-position-feasibility.md).
Baseline: mock-analysis PR #34, merged as `0467d04`; reviewed scientific
producer `3d6d09ec0f45172b202a43044360999ac3d80ef8`. Notebook 026 and private
M008 report the completed study. Code PR #24 and manuscript PR #5 are still
open at preparation. This proposal continues workspace issue #14. Repository
reorganization and lint repair are tracked separately in mock-analysis #35.

## 1. Decision this study must support

Determine which of these outcomes the retraced numerical terrain model supports:

1. Forward values and position slopes meet the declared numerical criteria at
   the tested positions and scales: proceed to a spatial emulator trial.
2. Forward values meet the criteria, but slopes remain unresolved or have
   distinct stable one-sided limits: proceed only with a forward-value trial;
   a later likelihood study can test inference that does not require gradients.
3. Forward values remain unresolved: improve the reference computation before
   investing in a dense spatial grid.

The first run tests reference resolution. Passing it does not validate a whole
box, establish physical terrain accuracy, or make a D5 likelihood ready.
Sampled derivative checks cannot certify differentiability everywhere.

## 2. Scope, assumptions, and limits

Retain the original spec's input manifest, DEM v0003 bilinear interpretation,
geometry, three height conventions, fixed pointing/beam/sky, ground temperature,
site, IERS policy, lmax 128, sky nside 64/niter 3, and four frequency values.
Read each from hashed inputs or the upstream result metadata. Use the existing
eight observed night coordinates and four explicitly synthetic tilts. Treat
87.5 m observed coordinates as the primary numerical target. Report the 30 m
and 91 m height controls and synthetic tilts separately; they cannot disappear
from a run's results because their tests fail.

Position offsets are grid East, grid North, Up in metres in the original DEM
frame/datum. The nominal position and era offset convention come from the
manifest, never a copied numerical constant. Bearings are degrees CCW from grid
East; use the existing conversion and common sorting permutation for true
compass azimuths. Temperatures are receiver-free, before balun and coax.

- ASSUMPTION A1: the frozen DEM and response define the numerical model being
  approximated. Beam/sky resolution, DEM errors, refraction, pointing errors,
  calibration and the physical position prior remain outside this test.
- ASSUMPTION A2: the existing +/-3.2 m box is an exploration domain. It is not a
  measured position distribution or an inference prior chosen by this work.
- ASSUMPTION A3: millimetre displacements probe the numerical bilinear surface;
  they do not establish millimetre physical terrain knowledge.
- LIMITATION L1: successive refinement differences are empirical resolution
  diagnostics, not rigorous continuum error bounds.
- LIMITATION L2: fixed-response tests do not cover changes in sky or beam during
  eventual inference. The likelihood milestone must test that additional scope.
- LIMITATION L3: a cubic horizon interpolant can be C1 while the complete mask
  and temperature chain has kinks. Autodiff alone cannot resolve that issue.

Keep these local assumption IDs with the implementation PR. Any new global
MOD/SI/queue IDs must be allocated by the workspace orchestrator.

## 3. Reuse and code ownership

| Existing code | Extension and responsibility |
|---|---|
| `horizon_position_d5/reference.py` | Keep the independent cell endpoint/interior maximum tracer; preserve selected-point and edge diagnostics. |
| `horizon_position_d5/cache.py`, `runtime.py`, `launch.py` | Reuse content-checked blocks and detached execution; add a reference-convergence phase with an explicit run plan. |
| `horizon_position_d5/temperature.py`, `pilot.temperature_model` | Reuse response construction; make bearing and mask-resolution sweeps explicit and retain every displaced temperature. |
| New `horizon_position_d5/convergence.py` | Pure refinement comparisons, one-sided slopes, per-cell outcome flags, and reports. |
| New `horizon_position_d5/reference_convergence.py` | Campaign selection, dry-run inventory, checkpoint orchestration and exports. |
| `eigsim/src/eigsim/position.py` | Reuse `PositionHorizonGrid` for the spatial phase. Generic evaluator changes belong here. |
| Analysis-code reader and notebook 026 | Strictly read the new schema; independent arithmetic, figures at each stage, dated memo exports. |

These are proposed paths under the current organization. If #35 lands first,
map them to the consolidated study directory without duplicating either driver.
D5 row selection, run gates and study reports stay in campaign code. Reusable
horizon interpolation and mask operators belong in eigsim. The independent
reference remains separately testable and cannot import the candidate it tests.
General terrain tracing changes would require coordination with eigsep_terrain.

## 4. First run: reference convergence

### 4.1 Freeze positions before computing new outcomes

The first deep test uses two development positions:

- the nominal offset `(0, 0, 0)`;
- the existing derivative-validation position with the largest absolute
  5^3-cubic versus 3 mm central-retrace discrepancy in the observed 87.5 m
  group, across E/N/U, the eight times and four frequencies. Break ties by
  lexicographic offset, axis, time, frequency. Derive this from the hashed
  ensemble arrays, save the selected coordinate and full selection record,
  and deduplicate against nominal before budgeting.

Selection based on old errors makes this a stress test, not an independent
validation set. New validation points are reserved for section 6.

For each position, retrace the center and both signs on every E/N/U axis at
half-steps 10, 3 and 1 mm. Save all center and displaced horizon curves and
T(p), T(p+h), T(p-h), not only central slopes. Do not force left/right agreement.
The dry-run derives counts after deduplication: two distinct centers normally
require 38 traces per bearing grid, 114 across three grids. One trace includes
all three heights; time/frequency evaluation does not require a new terrain
trace for each temperature cell.

### 4.2 Vary three controls independently

| Control | Initial levels | Hold fixed while comparing |
|---|---|---|
| Horizon bearing spacing | 0.1, 0.05, 0.025 degrees | Same position, mask subsampling and response |
| Azimuth samples per mask cell | 180, 360, 720 | Exactly the same traced horizon and response |
| Position half-step | 0.01, 0.003, 0.001 m | Same bearing and mask settings |

Evaluate the full bearing-by-mask cross-product on the saved stencil horizons.
This exposes cancellation between angular controls. A mask-only calculation
must not retrace terrain. Preserve the existing response's sky epoch and
validate any reused response against public simulation on independent masks.
Cross-commit reuse is allowed only as a verified, hashed upstream artifact;
never change a checkpoint's source identity to make it load.

If the last comparison remains unresolved, a separately costed extension can
add 0.0125 degree bearings and/or 1440 mask samples. It must preserve the same
positions, model and thresholds. Do not silently add work or select the
resolution with the smallest-looking slope. On an extension, apply the same
criteria to the finest three consecutive levels, retaining all earlier verdicts.
Do not select a nonconsecutive triplet. Store unrun levels as `not_run`.

### 4.3 Quantities and proposed numerical criteria

For each temperature cell and axis, in kelvin and metres, compute

```
D_plus(h)  = (T(p+h) - T(p)) / h
D_minus(h) = (T(p) - T(p-h)) / h
D_center(h) = (T(p+h) - T(p-h)) / (2*h)
B_D(a,b) = 0.001 K/m + 0.01 * max(abs(a), abs(b))
```

Keep signed differences, RMS, p95, maximum, sample counts and exact coordinates
of each maximum. Criteria apply per cell; averaging cannot hide a failure.

Proposed screening budget: 1 mK maximum temperature error for the later
fixed-model spatial trial, reserving 0.25 mK for each angular-control refinement
diagnostic and 0.75 mK for interpolation discrepancy against the selected
reference. These allocations guide screening; empirical reference differences
cannot be added to claim a certified 1 mK continuum bound.

For the reference test:

- **Forward resolved:** every center/displaced temperature in a claimed scope
  changes by <=0.25 mK in each of the two successive bearing refinements and
  two successive mask refinements. Apply bearing comparisons at every tested
  mask setting and mask comparisons at every bearing setting within the assessed
  triplet. Require the coarse-to-fine joint difference within that triplet to
  be <=0.25 mK too; report it separately to expose cancellation.
- **Slope numerically resolved at a stated h:** D_plus and D_minus each change
  by <=0.25*B_D in those angular comparisons. Require both adjacent comparisons;
  one small last difference is insufficient. Central slopes are reported too,
  but cannot override a one-sided failure.
- **Common slope supported on tested scales:** angular resolution passes, each
  side agrees across both adjacent h pairs within B_D, and the two sides agree
  within B_D at both smallest steps. This is evidence on 1--10 mm scales,
  not proof of a pointwise continuum derivative.
- **Distinct one-sided behavior:** each side resolves angularly and across h,
  but the two sides disagree at both smallest h. Preserve the slope pair and
  selected terrain points; report behavior consistent with a kink. Do not
  average it into a validated gradient or assert the physical cause solely
  from this numerical signature.
- **Unresolved:** other cases, nonfinite cells, missing comparisons or
  edge-limited traces in the proposed usable scope. Missing work never passes.

The constants are deliberately explicit, revisable engineering targets. Freeze
them in the reviewed pre-run manifest before evaluating the new results. A
later change produces a new decision profile and retains the old verdicts.
No real-data covariance or posterior-accuracy claim is implied by these values.

### 4.4 What happens if a control fails

Bearing failure: extend the bearing sweep within a newly measured budget;
inspect selected maxima and periodic interpolation near problematic bearings.
Mask failure: compare a small set against an independent integration oracle,
then consider an improved generic mask operator in eigsim. For a piecewise
linear horizon, a useful oracle splits azimuth intervals at horizon knots and
polar-cell crossings and integrates the clipped sin(elevation) contribution;
check it with independent high-accuracy quadrature on analytic fixtures.
Changing integration is a separate PR and a new numerical reference identity.

Stable forward values with unresolved slopes permit only the forward-value
path. Failure of the numerical reference stops a dense-grid launch. Report
which criterion failed and the cost of a targeted extension.

## 5. Broaden the reference before training a dense grid

A two-position pass is a pilot gate only. At the selected angular settings,
check the nominal point, both reference-development points after deduplication,
all six face-near and four corner-near positions already defined in
`ensemble.validation_positions`, and eight new reproducible interior positions
from seed 26101501 in [-3,3]^3 m. Freeze actual coordinates in the manifest.
These are development checks and may guide adaptation.

Compute direct forward values at all points. Apply the complete one-sided
stencil at the largest new forward-refinement discrepancy, a face-near point,
a corner-near point, and a fresh interior point. Freeze a deterministic
selection rule and ties before reading new results. Complete their required
angular/step comparisons; evaluating only at the selected resolution cannot
test convergence. Preserve the 30/91 m and tilt stress-control outcomes.

If the breadth checks fail, restrict the reported scope or improve the
reference. Do not shrink a physical prior around a surrogate's preferred fit.

## 6. Spatial emulator trial, conditional on the reference

Start with existing trilinear and cubic implementations. Regenerate the 3^3
and 5^3 interior grids at the accepted reference settings. Trial 9^3 only after
a measured cost gate. A one-cell halo at each grid's spacing surrounds the
+/-3.2 m query box, so cubic evaluation at its boundary uses traced neighbors.
Thus interior counts 3/5/9 imply stored axis counts 5/7/11; the dry-run must
count actual stored nodes and reuse only exact overlapping coordinates and
identities. The finest trial is 1331 stored nodes, not 729. A 17^3 trial is
not part of the initial allocation.

Use the development set for choosing spacing or locating a later adaptive
patch. Before the final assessment, freeze the candidate and generate 32 new
interior positions with seed 26101502 in [-3.19,3.19]^3 m, plus new face, edge,
and corner probes with different tangential offsets from the development set.
Persist exact coordinates and roles before evaluating them. Never reuse these
as training nodes. Any adaptation after seeing them needs another independent
assessment set and a new frozen candidate.

At those points test:

- temperature residual <=0.75 mK maximum against the selected direct reference
  in the declared primary scope, with all frequencies and rows retained;
- horizon residuals, cubic overshoot and physical elevation limits;
- derivative residual <=B_D against angularly resolved direct slopes where a
  common slope is supported, including both sides of spatial-cell joins;
- near-boundary, exact face/edge/corner, outside-domain and invalid queries;
- stability under an additional spatial refinement, using the same samples.

A sampled gradient pass must list tested steps and unresolved regions. A
forward-only pass can advance to the later likelihood study with that scope.
The query domain is distinct from the stored halo and from an eventual prior.
An adapter must enforce the query domain even though `PositionHorizonGrid`
accepts points throughout its stored box. No silent clipping or extrapolation.

## 7. Later likelihood and contribution milestones

Only after numerical feasibility, specify the intended time/frequency sample,
covariance C, priors and nuisance freedom. Keep fixed-nuisance and global
gain/offset synthetic controls as diagnostics; do not substitute them for the
intended inference model. For r=m_direct-d and e=m_interp-m_direct, evaluate
`Delta log L = -r.T C^-1 e - 0.5 e.T C^-1 e`, retaining coherent residuals.
Freeze likelihood, parameter-recovery and interval-stability criteria before
that phase. Null C means the likelihood gate is **not evaluated**, never passed.

If numerical and likelihood gates support the intended use, package traced
horizons, coordinate axes/frames, domain and halo, settings, hashes, error
reports and evaluator API. Preserve failures and one-sided behavior. A
successful synthetic numerical study remains **diagnostic** under the workspace
labels; physical validation requires held-out real-data prediction and
material systematics. Shared contribution still requires canonical main runs,
CHB review, a fresh Limbo survey, a shared-runnable generator and a reviewed
packet executed in a separately approved contribution session.

## 8. Output contract and completion

New versioned exports must contain:

- frozen plan, explicit coordinates and selection reasons, source/input/lock/
  environment and upstream hashes; trace, response and evaluation identities;
- raw trace horizons and selected-point/edge diagnostics;
- all signed stencil temperatures with bearing, mask, h, axis, sign, height,
  time and frequency coordinates; no inferred array-position coordinates;
- derived slopes, all refinement comparisons and separate forward/one-sided/
  common-slope outcomes, plus exact exclusions and scope masks;
- planned/completed/missing block inventories, timings, memory, attempt/unit/
  InvocationID, exit evidence and content-hashed terminal result manifest.

A clean process exit is operational success. Numerical gates have separate
statuses `pass`, `fail`, `unresolved`, `not_run`; any reporting code must preserve
that distinction. Notebook 026 independently reconstructs the finite changes,
shows temperatures and residuals beside slopes, and exports a dated M008
amendment. The accepted initial-study text and original products stay intact.
