# B8 pointing/window injection code home

Companion to christianhbye/eigsep_analysis_code PR #8 and its committed
`docs/specs/2026-09-17-b8-raster-estimator-fixes.md` revision `35aabc7`,
especially sections 4.2.7 and 8. Resumed analysis helpers are at `c675585`.
The workspace's 2026-10-08 overnight authority permits recorded provisional
implementation choices; scientific thresholds remain unchanged.

All synthetic work lives in mock_analysis. Keep this pointing-only generator
in its own NumPy-only workspace member rather than adding eigsim's sky/JAX/
s2fft pins to the analysis environment. It never imports eigsep_cal or the
analysis estimators, and does not replace the future instrument generator
(D5 interface Q-CHB-29). Analysis inverse/estimator code remains in PR #8.

The first component evaluates ordered, nonoverlapping constant-rate moves and
their exact continuous-window means. It is an idealized drive, not a recovery
of raw telemetry-update times. D5's update averages must separately pass the
model's prerequisite or be counted as unmatched under the original plan.

**Provisional quadrature representation:** a fixed 32/64-point midpoint rule
does not meet the 1e-6-degree identity at arbitrary start/stop phases. Use
analytic microbin moments for the drive, labeled as means with representative
midpoint timestamps. Their equal-weight row means are exact. Nonlinear play,
harmonics, transient and clock shifts must still be tested at both resolutions;
exact drive moments do not assert exact nonlinear integration. This is an
explicit numerical representation choice, not a relaxed identity threshold.

Current checks are test-first analytic partial-window values, 32/64-bin
identity, the failed midpoint counterpart, Unix-epoch stability and invalid
move/window rejection. The remaining injection effects, independent physical
reference fixtures, analysis adapter and real-data calibration remain pending;
this component alone does not make PR #8 merge-ready.

## Injection layer: bounded implementation

The play path is integrated analytically by shortening each monotone drive
move by its takeup duration; short moves may not take up the gap at all.
The default synthetic initial state arrived from the direction opposite the
first move, and its offset is returned explicitly. This is not a D5 hardware
fact. Positive report lag evaluates the physical path at report time + tau.
Play and drive window moments remain exact at both resolutions and at Unix
epochs (microbin construction uses relative coordinates to retain precision).

Transient and windup follow the declared drive advance. The transient rises
linearly over the supplied ramp length before its exponential decay, only
during motion; windup is present only in cruise outside the edge ramp lengths.
The angular transfer applies (1-eps)*attitude + Phi(attitude) to the physical
attitude including these effects, with Phi's columns cosine/sine coefficients.
Those nonlinear/edge terms still need resolution/recovery checks. Noise is
stationary AR(1) with marginal sd, added after row averaging. The caller owns
sensor wrapping and records its RNG seed beside the returned truth parameters.

Six focused tests pass: the three moment checks plus exact play takeup/short
moves/zero-play, zero-effect and lagged identity, and AR(1) scale/correlation
and seeded repeatability. Remaining work is independent physical-reference
fixtures, the analysis adapter/inverse checks, burst sensitivity and real-data
calibration. No real calibrated result or ready-to-merge claim follows.

The azimuth helper adds a declared lagged unit ramp to externally supplied
per-row elevation contamination. Its ramp/window moments are analytic, and
noise is explicitly per-row Gaussian. The independent reference exporter
uses a parked-middle/elevation-moving-outer geometry, 0.243-nat elevation
structure versus a 0.113-nat azimuth step, 50 realizations for each of lags
0/0.2/0.4 seconds. Neither generator nor exporter imports analysis. This
fixture tests removal of linear elevation contamination; a real neighboring-
leg contamination and quadratic sensitivity remain separate requirements.

Batch sensitivity expands a declared cycle-average macro motion into 72-ms
active ramps every 179 ms, with a shortened last batch preserving total angle.
The first batch starts at the macro onset. These approximate timings are a
synthetic sensitivity convention, not firmware replay or a measured sampling
phase. Batch fixture effects are restricted to lag/play with zero transient
and windup; the current transient generator resets on each supplied move and
has not defined an inter-batch mechanical response. Do not use it silently
for nonzero batch transient/windup. Independent burst references are exported
by summing analytic ramp-window overlaps, not the analysis inverse formula.
