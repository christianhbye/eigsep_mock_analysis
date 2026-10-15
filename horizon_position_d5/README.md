# D5 position derivatives

This diagnostic study compares independently retraced bilinear-DEM horizons,
finite-scale skyline formulas, and differentiable position interpolation through
eigsim's antenna-temperature model. It does not infer a D5 position uncertainty.

The approved scope and mathematical controls are in
[`../docs/specs/2026-10-14-d5-position-derivatives.md`](../docs/specs/2026-10-14-d5-position-derivatives.md),
with the implementation plan beside it. Analysis notebook026 lives in the
private eigsep_analysis_code repository and records the executed results.

- `reference.py`: independent cell endpoint/interior maximum and sampled controls.
- `inputs.py`: hash checks and the explicit selected D5 night coordinates.
- `temperature.py`: existing eigsim setup reused for the mask response, checked
  against public simulate_path calls and analytic temperature controls.
- `cache.py`: atomic array blocks with content, source/input and parameter checks.
- `reference_pilot.py`: full-resolution nominal trace reproduction and timing.

From the monorepo root, run focused tests with
`uv run python -m pytest horizon_position_d5/tests`.
Run heavy phases on Limbo in the registered private checkout, with nice2,
thread/CPU limits, and the aggregate cbye CPU rule. Inputs, caches and logs must
be private; team directories are read-only. The pilot command is
`python -m horizon_position_d5.reference_pilot --inputs INPUT_DIR --output RUN_DIR`.
An input directory contains INPUT_MANIFEST.json and each named hashed file;
its source paths are provenance, not fallbacks when an input is missing.

## Recent changes

- 2026-10-14: Added independent reference, interpolation and temperature controls
  for issue14; D5 execution and conclusions are recorded separately from tests.
