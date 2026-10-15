# D5 position-to-temperature derivatives

Status: revised after independent math and fit reviews, 2026-10-14; ready for implementation.
Owner: separate Codex study, approved by CHB; Claude keeps the D5 trackers.
Branches: mock_analysis `feat/d5-position-derivatives` from `80e027d`;
analysis_code `nb/d5-position-derivatives` from `5400e59`, notebook 026.
Run tag: **LIMBO-COMPUTE**. Label: **diagnostic** throughout.

## Question and deliverables

Do E7, E7-prime, and a position interpolator predict the antenna-temperature
response to D5 position changes, and what numerical error would each introduce
into an inference calculation? Issue 14 reopened the skyline derivative; it did
not establish an error in eigsim's fixed-horizon temperatures.

1. Reusable JAX position-to-horizon interpolation in eigsim, with tests that
   distinguish interpolated values, derivatives, and behavior at grid boundaries.
2. A reproducible simulation study in `horizon_position_d5/` of mock_analysis:
   independent bilinear-DEM reference, convergence controls, temperature response,
   held-out position tests, and a clearly conditional inference diagnostic.
3. Executed notebook `026_position_derivatives.ipynb` in analysis_code, reading
   committed-generator outputs, plotting every stage, and recording provenance.
4. Two private PRs for CHB. Leave them open. The shared memo is a later
   contribution packet after review, merges, canonical reruns and a fresh Limbo
   comparison (`eigsep_analysis/docs/autopilot.md` section 11).

No modifications to live notebook 022, S1, their worktrees, the shared Limbo tree,
or published terrain/geometry products. No claim to infer a physical position
uncertainty, validate the terrain below its pixel scale, or establish an absolute
D5 calibration. A numerical failure is a reportable result, not a gate to tune
until a preferred method passes.

## Inputs and frames

Record file hashes and package/source commits in the run manifest, rather than
silently following a path that another task can update.

- DEM v0003, float32 samples on a 0.5 m raster, interpreted as a bilinear surface.
  Laptop input is the existing `limbo_sync/bulk/.../dem/v0003/marjum_dem.npz`.
  Copy the same hashed input to private Limbo storage before computing.
- Geometry release v0004: take `antenna_enu_m` from the team's
  `horizon_profiles_v0003.npz`; verify against the release metadata.
- Team horizon v0003 and 022's recorded candidate/march/bracket arrays are
  comparison inputs only; their generating hashes must be recorded. Existing
  bracket points may seed an emulator but cannot validate themselves.
- Beam `eigsep_bowtie_v002_mwss.npz`, native HFSS, existing eigsim config and
  croissant commit pinned by `uv.lock`. For data-coordinate comparisons exclude
  the 250 MHz endpoint and select by the frequency values, not array offsets.
- D5 time/pointing pairs from the existing `binned_v0001` plus `fit_v0009`
  manifest, using the coordinate mapping already checked in S1's
  `scripts/023_eigsim_columns.py`. Record its source SHA and the actual rows.
  Use `fit_v0009/fit.npz["fitted"]` only as a selection mask and verify its
  time/era/pointing arrays equal the binned arrays. Its 165 selected rows are
  all 87.5 m; recorded 91 m rows are not in this night selection. The 30 m and
  91 m horizons at these coordinates are synthetic height controls. The
  recorded 91 m unusual/inverted angles are separate implementation stress
  controls, without an assumed physical interpretation.
- GSM2016 via pygdsm 1.7.0 (high resolution, no CMB, ud_grade to nside 64,
  sky transform niter=3), held fixed during position perturbations. Reuse
  S1's hashed `gsm_maps.npz` and record its source-map hash and generator
  metadata. Pin both IERS tables and disable automatic downloads; record the
  reference epoch, actual selected times and any table prediction policy.
  Apply the same explicit Marjum site as S1: (39.247699, -113.402660, 1800 m).
  Reuse sky alms only when the first-time reference epoch is identical.
- Use receiver-free `eigsim.simulate_path`; fixed uniform ground temperature
  from the config. These are free-antenna temperatures, before balun/coax.

Position p=(E,N,U), metres, is on the DEM's UTM grid axes, with the existing
vertical datum. A ray has beta CCW from grid East, u=(cos beta,sin beta), and
c=(sin beta,-cos beta). Convert bearing to true compass azimuth with eigsim's
existing `grid_bearings_to_az`; sorted angles, horizons and tangents share one
permutation. All differentiation uses radians internally. Three era heights
follow the team convention: U_era=U_91-(91-h_era), h_era in {30,87.5,91} m.

ASSUMPTION A1: all eras share the nominal horizontal position, as in the team
product. This is an input convention, not evidence that the antenna stayed fixed.
ASSUMPTION A2: the box |delta E|,|delta N|,|delta U|<=3.2 m is MOD-102's
conservative bracket, not a sigma or a measured posterior (Q-ARP-02).
ASSUMPTION A3: bilinear interpolation and geometric Earth curvature R=6371 km,
without refraction, define the numerical terrain model under test.
ASSUMPTION A4: the initial temperature test fixes pointing, beam, ground, sky
and the topocentric/site frame. It isolates the terrain-mediated position
response; it does not vary latitude/longitude or a terrain-dependent beam.
Any parameter-recovery demonstration must state which nuisance parameters are
fixed or fitted; it cannot establish the uncertainty of the full staged fit.

## Independent horizon reference

For a fixed bearing, alpha(p)=atan(max_r q(r,p)), where
q=(h(E+r ux,N+r uy)-U)/r-r/(2R), r>0. Preserve the team's interpolable DEM domain
(the upper raster boundary excludes the last pixel) and its 0.25 m inner cutoff.
Report edge-limited bearings separately. The upper coordinate bound is
(nx-2-e0)*res (and the corresponding N bound), exactly as team `edge_range`;
this differs from the full mathematical bilinear domain by one pixel.
Hold the 0.25 m inner cutoff fixed when refining the sampled march step. The continuous endpoint limit at the
far tile boundary is part of this reference; the product's finite march omits
that endpoint. Neither model asserts that terrain outside the tile is absent.

Partition the ray at every crossed raster line. In each open cell use
 t=r-r0, h(t)=a*t^2+b*t+c0.
Bilinearity gives
 a=(z11-z10-z01+z00)*ux*uy/res^2,
 b=grad(h at r0 from this cell).u,
 c0=h at r0.
The curvature-corrected slope is q=(a-k)*r+(b-2*a*r0)+C/r, where
 k=1/(2R), C=c0-b*r0+a*r0^2-U.
Its stationary radii obey r^2=C/(a-k). Evaluate each cell endpoint and every
positive stationary radius within that cell, then take the global maximum.
When a-k=0 there is no isolated stationary candidate unless q is constant.
For numerical stability, an equivalent local form is
 D0=b*r0-c0+U-k*r0^2,
 t=-(D0/(a-k))/(sqrt(r0^2-D0/(a-k))+r0), when the square root is real.
Degenerate zero-width intervals must not generate divisions by zero. Use float64
arithmetic on the original DEM values. Preserve the nearest-radius tie rule and
record competing maxima/switches where relevant. Deduplicate shared cell
endpoints before interpreting ties; a diagnostic near-tie tolerance must
never replace the actual argmax or merge distinct skyline points silently.

This is a study reference, not a replacement for the shared terrain generator.
A separate sampled mode must reproduce the team's 0.25 m horizons before the
continuous calculation is interpreted. Verify the reference against dense
independent sampling on synthetic scenes and selected real rays, including
crease maxima, interior maxima, near ties, cardinal rays and tile boundaries.
Dense sampling gives a lower bound on the continuous maximum, not a certified
upper bound on reference error. Report the sampled-reference gap as a
convergence diagnostic; call an error bound certified only with a separate
proof or interval bound. The synthetic analytic cases and independent
within-cell optimization test completeness of the candidate calculation.

For p shifts, central differences are
 D_i(h)=[T(p+h e_i)-T(p-h e_i)]/(2h), in K/m.
Use h in {1,0.3,0.1,0.03,0.01} m and selected 4,2,1 mm controls. Separate local
limits from finite-scale slopes; where left and right limits differ, report a
kink and finite displacement response. Do not force a unique derivative.

## Candidate derivatives and chain checks

Retain the identities used by 022 as hypotheses with explicit reference models:
 d_alpha/dU=-cos(alpha)^2/r_star,
 d_alpha/ds=sin(alpha)*cos(alpha)/r_star+cos(alpha)^2/R,
 d_alpha/dc=slope/r_star,
 d_alpha/dE=cos(beta)*d_alpha/ds+sin(beta)*d_alpha/dc,
 d_alpha/dN=sin(beta)*d_alpha/ds-cos(beta)*d_alpha/dc.
E7 uses slope=-periodic_central_difference(alpha,beta); its physical smoothing
scale is r_star*delta_beta, which varies with bearing. E7-prime uses
slope=cos(alpha)^2*grad_DEM.c at the selected product sample. Neither is assumed
to be the continuous crease derivative. Vertical/along identities also require
checking their domain and boundary exceptions; E4 generally fails for a
fixed discrete march or a maximum constrained by the fixed inner cutoff.
A stable crease alone does not invalidate E4/E5. No geometry assertion is a test.

For each candidate, push d_alpha/dp through `open_sky_weight` and
`simulate_path` using JAX JVP. Test alpha->W->T separately with known smooth
perturbations and converged central differences. Include a uniform sky with
Tsky=Tground (horizon-independent T), a known nonzero sky-ground contrast, and
a perturbation with sign/frame sensitivity. A zero chain response must not pass
as a successful position gradient. Add a horizon exactly on a theta-cell
edge and compare both one-sided differences with the central difference.
The clipped cell weights are only piecewise differentiable for finite phi
subsampling: JAX can return a half-slope that agrees with a central difference
at a kink. C1 interpolation of alpha does not make the whole T chain C1.

For fixed sky, beam, pointing and ground, the implemented temperature is affine
in W: T(W)=T(0)+K.W. This follows from the masked beam transform and convolution
being linear and f_ground=1-integral(B W)/integral(B). A cached response operator
may accelerate position sweeps. Verify it on independent W perturbations against
full `simulate_path`, with relative/absolute errors printed, before using it.
Keep the operator specific to its full input fingerprint and write in blocks to
avoid storing every time/frequency by every sky cell in RAM.

Resolve three numerical controls independently: position step, horizon angular
sampling (0.1,0.05,0.025 deg on a subset), and W phi subsampling (180,360,720).
Beam/sky band limit is fixed for the initial implementation check and named in
all conclusions; a higher-resolution subset checks physical discretization
before a stronger claim. A local exact DEM maximum does not remove either angular
or W discretization. Use the actual implemented response as the first target.

## Position interpolator

New `eigsim/position.py` exposes a validated, fixed rectilinear uniform grid of
horizon curves and a JAX query at p. The grid's three spatial axes are metres;
last axis is the fixed sorted true-azimuth grid. Support trilinear interpolation
as a baseline and tensor-product cubic Hermite/Catmull-Rom as a C1 candidate.
Cubic weights for t in [0,1] are
 (-t/2+t^2-t^3/2, 1-5t^2/2+3t^3/2,
  t/2+2t^2-3t^3/2, -t^2/2+t^3/2).
Use linearly extrapolated ghost samples at domain edges so adjacent cells share
the same first derivative. This boundary convention is explicit and tested.
Do not extrapolate outside the trained position box: concrete calls raise;
traced calls return NaN and offer a validity predicate. Do not silently clip a
query or an overshooting horizon, since that would alter its gradient.

Tests: node reproduction; constant/affine truth including all boundaries;
interior quadratic truth where the cubic stencil supports it; JVP vs analytic
truth and off-node finite differences; C1 joining for cubic and documented
slope jumps for linear; axis order and unequal spacings; invalid inputs and
out-of-domain behavior under jit/vmap. A perturbed derivative implementation
must fail at least one analytic test.

D5: first grids of 5 and 9 points per axis over the bracket, with adaptation only
if timing and errors justify it. Reserve deterministic off-grid positions, a
second independent random seed, and face/corner-near positions before training.
Use newly traced continuous references; no training node is a validation point.
Check temperature values and position derivatives separately. A smooth surrogate
cannot remove physical or model kinks; list failures and the scale they affect.

## Pilot, gates and inference diagnostic

Pilot: observed-coordinate sample at 87.5 m, four native beam frequencies
spanning the supported data band, three displacement axes, and synthetic
analytic controls. Select chronological 87.5 m fitted rows with
np.rint(np.linspace(0,N-1,8)): rows [0,23,47,70,94,117,141,164] in the current
input, July 17 04:22-11:23 UTC. This samples time near one parked orientation,
not a raster. Apply S1 offsets and Rz(psi). Repeat those coordinates at 91 m
and 30 m as synthetic height controls. Include an explicitly synthetic
drive-angle grid at representative selected times to test non-zenith responses.
Do not describe the chosen night rows as broad D5 orientation coverage.
Measure one full-sized trace and temperature block, RSS, and actual CPU time.
A one-time timing gate records the remaining schedule and any scope drops before
the ensemble begins. Extend to three eras, all 51 supported native frequencies,
and more independent selected D5 night pairs within the measured cost. Keep every scope drop
in the run report and PR; do not describe a pilot as full-D5 validation.

Pre-registered numerical gates (engineering targets, not physical accuracies):
- Reference: synthetic scenes at double-precision accuracy; dense real-ray
  checks approach the exact maximum from below, with a reported error bound
  and the independent angular/W resolution controls.
- Affine temperature cache: matches full simulation to 1e-8 K absolute plus
  1e-10 relative on the held-out mask controls; otherwise bypass the cache.
- AD implementation: resolved alpha->T perturbations match a finite-difference
  plateau to 1e-4 relative RMS, with absolute K and K/rad errors also reported.
- Position candidates: report signed errors, RMS, p95 and maximum K/m, and
  finite-displacement prediction errors in K at all tested scales. The reference
  uncertainty must be <0.1 of a claimed candidate error, or label unresolved.
- Emulator: report both spatial resolutions and held-out errors; do not select
  solely on derivative agreement or solely on forward temperature agreement.
  Store the full residual arrays, worst-case coordinates and failure masks.

Inference demonstration uses direct-reference synthetic temperatures at held-out
positions. E7/E7-prime forward candidates are T(p0)+J(p-p0), a local linear
model whose domain is the tested position bracket; the interpolator supplies
a separate nonlinear forward model. This is a surrogate-likelihood test, not
a test using an approximate optimizer gradient with an exact forward model.
Fit with a candidate model over a local identifiable subspace;
compare to direct-reference likelihood evaluations. Test a fixed-nuisance case
and a single global multiplicative gain plus a single global additive K
offset across the selected times/frequencies, reporting Jacobian
rank/conditioning. Per-frequency freedom is an additional labeled variant
only if the global experiment is identifiable and the time budget permits. If a
D5 covariance/position prior is not established, use declared synthetic noise
levels and a bounded uniform position prior for a numerical experiment, clearly
separated from D5 uncertainty. Report results across noise scales (the numerical
error/noise ratio), rather than adopting a new D5 noise floor. Compare likelihood
changes and position bias/width; optimizer convergence alone is insufficient.
If the model is unidentifiable, report that and compare predictions instead.

## Execution and provenance

All heavy work runs in a new private Limbo checkout under
`/mnt/data02/cbye/research/eigsep/mock_analysis-d5-position-derivatives`.
Logs, caches and checkpoints stay in `/mnt/data02/cbye/research/`; no writes under
`/mnt/data02/eigsep`. Tracked code moves by private GitHub branches. Bulk input
copies are hashed; never fetch or change Git configuration in the team tree.

Launch with **nice 2**, detached systemd user unit, a runtime budget and a private
non-AVX JAX 0.9.2/jaxlib 0.9.2 environment, matching the mock lock, based
on the private astronova-noavx Python (conda jaxlib build cpu_py312hc81e8bd_0).
Keep it separate from the analysis-code JAX 0.10.2 environment. Require a
JIT/JVP smoke test and record x64, package/import paths and the conda build.
Verify the actual nice level and import origins. Use
CPU affinity/thread limits to bound this study, initially two logical CPUs, and
check every other cbye job so aggregate use stays <=one third of all 16 logical
CPUs. Reduce concurrency if host pressure grows. User-service CPUQuota is not
assumed effective on Limbo's cgroup-v1 configuration. Never modify another job.

Checkpoint each trace/response block atomically under a hash of inputs, parameters,
source tree and lockfile; failed/incomplete blocks cannot be reused. `--dry-run`
prints cache validity. Store `RUN_RECORD.json`, frozen `gate.json`, and DONE/FAILED
sentinels. Check the live systemd unit as well as sentinels on resume. Never sync
the run environment while its unit is active. Bound logs and serialized output.

Code notebook 026 loads these results through `paths.product_dir` and a small
validated result reader that rejects missing/incomplete or stale blocks,
inconsistent source/lock/input fingerprints and mixed run configurations.
Distinguish the simulation runtime from the analysis-code runtime.
Its first code cell calls `provenance.show()`. Save an
intermediate figure for every section, units/frames on axes, tables with maxima
and counts, a claim-to-evidence table, and `PROVENANCE.json` with upstream hashes.
Do not embed expensive simulation in notebook execution. The canonical rerun
waits for CHB's merges, as does any claim of graduation to the team.

## Math to review

1. Bilinear-ray stationary points, completeness of endpoint/interior candidates,
   roundoff, cardinal rays, near ties and the team's domain boundaries.
2. The slope/position identities, frame permutation, and crease/boundary exceptions.
3. Affineness in W and independent temperature controls, including normalization.
4. Cubic weights, C1 continuity, boundary extension and differentiable queries.
5. Separation of exact implementation gradients, continuous-model gradients,
   finite-scale sensitivity, and inference error under stated nuisance freedom.

## Review log

Fit review complete (fresh reviewer, 2026-10-14):
- F1, must-fix: actual night mask and parked pointing scope. Fixed; input manifest
  now includes fit_selection.npz SHA c3b1935935061aeecf57dd697bc83d5c6cc1b181c66a80bbd86943694eb3a3d0.
- F2, must-fix: JAX/jaxlib version mismatch on the common Limbo base. Fixed by
  the separate matching 0.9.2 base; JIT/JVP preflight remains required.
- F3-F6: complete IERS/GSM/time fingerprint, define linear surrogate and global
  nuisance freedom, enforce notebook input validation, label synthetic controls.
  All adopted above. Source and data hashes are part of the implementation.
- Limbo survey: main df83d94, memo-008 bcce4a5, memo-008-excess 0174cf2,
  memo-008-sky 22e1c3f; no derivative branch found in the focused survey.
  Tomorrow's contribution packet must correct the old skyline packet's claim
  of exact derivatives from skyline distances/heights in light of issue 14.
Math review complete (fresh reviewer, 2026-10-14), all findings adopted:
- M1: exact team outer boundary and fixed inner cutoff; addressed above.
- M2: finite-subsample W clip kinks; one-sided boundary control added.
- M3: preserve the synthetic crease counterexample to E7-prime as a regression.
- M4: dense sampling is a lower bound, not a certified reference error bound;
  wording corrected. Endpoint duplicates are distinct from competing maxima.
- M5: E4's fixed-inner-cutoff/discrete-march exception stated explicitly.
- M6: require x64 and explicit IERS configuration; already in the input/runtime gate.
- Retain an independent crease fixture h(x,y)=3+0.3y-|x-10|, p=(0.5,0.1,0),
  beta=0.45 rad, r in [0.25,50] m. Its actual across derivative has opposite
  sign to one adjacent-cell E7-prime value. The reference tests will compute
  expected derivatives from the ridge geometry, not copy a result table.
Review scratch: /tmp/d5-derivative-math-review/check_math.py,
math_results.json and check_jax.py. The implementation retains the falsifying
controls as tests. No D5 numerical conclusions yet.

## Implementation review and timing amendment (before the ensemble)

The first private Limbo reference run at `8827688` took 151.1 s for one
continuous full-bearing trace of all three heights, with 1.01 GiB peak RSS.
At that measured shared-host rate, a 9^3 training grid alone projects to
15.3 hours with two workers. The proposed overnight gate therefore compares
nested 3^3 and 5^3 grids (125 distinct training positions), instead of running
the proposed 9^3 grid. The response-only phase at `84e40be` supplies the
remaining timing measurement; freeze the final gate before starting the
position sweep. No interpolation accuracy has been observed when making this
scope decision. A 9^3 result must not be claimed. A higher beam/sky band-limit
comparison and expansion to all 51 frequencies remain outside a pilot unless
the measured gate explicitly adds them.

The ensemble freezes 18 validation positions before training: four from each
of seeds 192840 and 716933, six face-near and four corner-near points. Expensive
position derivative checks use indices [0,4,8,14], covering both seeds plus a
face and corner, at 10 mm and 3 mm, retaining both one-sided responses. The
other held-out positions test forward values only. Two noiseless inference
injections use indices [0,4], and only observed 87.5 m night coordinates.

Fit review of implementation, all findings adopted:
- Bind report readers to coordinate values, expected block names and finite
  report contents, as well as hashes. Validate the complete ensemble schema;
  an empty inventory is never a completed scientific result.
- Include untracked source files in the runtime clean-tree check.
- Require reference derivative resolution before interpreting reference local
  Gaussian widths. After the applicable nuisance projection, each column's
  step and half one-sided RMS changes must be <=1e-6 K/m + 1e-3 relative;
  their operator norms must be <0.1 of the smallest singular value. This is
  an engineering convergence diagnostic, not a certified error bound.
- Six deterministic optimizer starts include the known synthetic injection,
  zero and four signed 2 m corners. Keep all costs, failures and positions.
  Report recovered-minus-injected displacement and competing minima; a local
  success and rank 3 do not identify an interpolation-induced bias. An exact
  sine-model ambiguity is retained as a falsifying unit test.

A tiny analytic-horizon integration test runs the complete ensemble and
inference product path, including serialization. It checks axis order,
interpolated forward values and the directly evaluated reference residuals.

## Conformance revisions, 2026-10-15

The independent conformance review found no equation defect in the terrain
reference, candidate propagation, interpolation or nuisance projection. Its
independent optimizer matched 40 tiny DEM/ray scenes within 4.44e-15 in tan(alpha);
curvature-sign, transverse-frame and interpolation-derivative mutants were caught.
The following reporting and execution findings are addressed before completion:

- The reader now pins both reference JSON hashes and producer/input identities,
  and recomputes the numerical pass conditions. Notebook provenance includes them.
- Notebook 026 compares finite-displacement predictions in K at every declared
  pilot step, retaining sampled/continuous baseline and change-only errors.
- It compares recovered position errors with candidate and resolved-reference
  local widths across all declared noise scales, retaining validity/bound flags.
- A claim-to-evidence table ties quantitative summaries to the saved arrays.
- A uniform 1200 K sky, 300 K ground and uniform beam control checks the nonzero,
  negative horizon derivative against independent NumPy MWSS quadrature and the
  analytic cell width. It passed with atol=1e-8 K/rad and rtol=1e-10.
- `launch.py` records source/gate pins, output ownership, CPU reservations and
  quota, nice 2, detached invocation IDs, attempt logs, CPU time and peak RSS.
  `--dry-run` does not write. The operator reserves other jobs' peak CPU budgets;
  the launcher's unit quota does not govern other sessions.

The first full-resolution response attempt (84e40be) segfaulted during JAX
`jit__orient` compilation with an 8 MiB stack, while RAM remained available.
A retry with a 64 MiB stack keeps source and scientific parameters identical;
its outcome remains pending. The failed attempt is preserved, not relabelled
as a numerical result. The interrupted full test run had 99 passes, 8 skips
and 8 stale-IERS prediction failures; `test_suite.py` uses the study's verified
IERS inputs for a controlled rerun. No acceptance threshold was relaxed.

A collection-time `iers.conf.auto_download=False` in the new toy temperature
control contributed to the stale-table failures in unrelated 2026 frame tests.
It is now a scoped fixture restored after each toy test. The detached rerun
still uses pinned current tables for deterministic science validation. This
fix removes a test-isolation regression rather than relaxing table validity.
