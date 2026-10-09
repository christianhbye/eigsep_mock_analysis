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

The lag/play/transient/noise injection layer and analysis adapter are still in
development. These first functions do not establish a calibrated D5 result.
