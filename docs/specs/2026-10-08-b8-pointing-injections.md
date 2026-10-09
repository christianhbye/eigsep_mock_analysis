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
