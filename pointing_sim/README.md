# Pointing simulation helpers

Private NumPy-only drive/window generators for B8 (analysis-code PR #8).
This workspace member has no sky, receiver, calibration, analysis-package or
file-I/O dependency. The receiver/instrument generator described in the D5
interface remains a separate future component.

`drive.uniform_drive` evaluates a declared piecewise constant-rate drive.
`uniform_window_means` integrates it exactly through partial start/stop rows.
`uniform_micro_means` returns exact sub-window moments and midpoint
representative timestamps. Those values are **means**, not instantaneous
midpoint samples. Their row averages preserve the drive identity at 32 or
64 sub-windows; nonlinear transformations still need resolution checks.

These are idealized model quantities. D5's stored motor positions are sparse
update averages, with update timestamps/counts absent from the saved payload.
An exact model identity is not proof that this observation model describes D5.
Count unmatched real moves and report the limitation explicitly.

From the workspace root:

```bash
uv run pytest pointing_sim/tests
uv run ruff check pointing_sim
uv run ruff format --check pointing_sim
```

`inject.play_model` converts a monotone constant-rate drive to exact box moves
with play takeup, including a declared synthetic initial condition.
`simulate_windows` adds lag, a ramped reversal transient, cruise windup,
harmonics/linear gain and stationary per-row AR(1) noise. Drive/play moments
are exact; nonlinear effects are evaluated at microbin resolution. Returned
angles are unwrapped, with all truth parameters beside them.

The independent analysis reference fixtures, real-count reconstruction/adapter,
burst sensitivity and real-data calibration remain in development. These
functions do not establish a calibrated D5 result.
