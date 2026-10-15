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

## Temperature pilot

Run `python -m horizon_position_d5.pilot --inputs DIR --output DIR
--response-only` first, with a detached wrapper supplying nice 2, CPU/thread
limits, runtime bound, log and exit sentinels. This checks the sampled horizon
before building one frequency response block at a time. Each affine response
is tested against public `simulate_path` on independent masks; failed blocks
are not saved. Sequential adjoints bound the intermediate transform memory.

After recording the measured timing gate, repeat without `--response-only`
from the **same clean commit and environment** to reuse checkpoints. The full
pilot stores independent E/N/U retracings at 1 m through 1 mm, candidate
Jacobians, one-sided differences, smooth alpha-chain perturbations, phi
subsampling and angular-resolution controls. The final `RESULTS.json` is a
hash-checked inventory; `RESPONSE_DONE.json` means only that the response phase
finished. A wrapper's successful exit is not completion of the whole study.

The first eight coordinates sample the selected 87.5 m night near a parked
pointing; the last four are synthetic tilts. The 30 m and 91 m curves at these
times are height controls. There is no D5 physical position posterior here.
Run unit checks with `uv run python -m pytest horizon_position_d5/tests`.
