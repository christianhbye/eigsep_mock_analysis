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
Run heavy phases on Limbo in the registered private checkout, with nice 0,
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
--response-only` first, with a detached wrapper supplying nice 0, CPU/thread
limits, runtime bound, log and exit sentinels. On the private Limbo no-AVX
environment, use `--split-orientation` after the recorded fused-compilation
crashes. It preserves the model and transform resolution while removing the
outer orientation/adjoint JIT wrappers. Keep that flag identical on resume.
This checks the sampled horizon
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

## Held-out ensemble

`python -m horizon_position_d5.ensemble --inputs DIR --pilot COMPLETED_PILOT
--output DIR --workers 2 --grid-points 3 5` reads the pilot as a hashed
upstream input, so its producer commit can differ from the ensemble driver.
Freeze the measured `gate.json` before launching; the first timing measurement
rules out a 9^3 grid within the overnight budget. Every training/validation
trace is independently checkpointed. A killed run can resume only at the same
source/input identity. Source changes require a new output directory.

Eighteen held-out positions test forward temperatures; four positions also
receive newly traced 10 mm and 3 mm derivative controls. Six optimizer starts
for each conditional injection expose competing solutions. Local Gaussian
widths are diagnostics per declared independent kelvin noise; the box is a
uniform synthetic prior, not an inferred D5 position distribution. Reference
widths carry a separate derivative-resolution flag. Inspect all failure,
rank, conditioning and bound-contact information before interpreting a fit.

## Detached launch and resume

Use `python -m horizon_position_d5.launch` for later phases. Supply `--phase
response|pilot|ensemble`, private `--inputs`, `--output`, a full `--commit`,
`--cpus 0,1`, `--reserved-other-cpus` (the peak allocation of all other cbye
jobs, not their idle usage), and `--gate gate.json`. The gate records
`producer_commit`, `allowed_phases`, `scope` and `timing_basis`; create it from
measured pilot costs before choosing the ensemble. Add `--pilot DIR` for the
ensemble. Run the same command with `--dry-run` first.

The launcher checks clean source, input hashes, checkpoint integrity, existing
run identity and live output owners. Dry runs write nothing. Actual launches
use an output-scoped lock and systemd unit, nice 0, requested per-unit CPU quota, affinity,
single-thread BLAS, a bounded runtime, attempt logs and `/usr/bin/time -v`.
The CPU reservation sum cannot exceed 50% of the host's logical CPUs;
affinity supplies the bound because user-service quota enforcement is not
assumed on Limbo's cgroup v1 setup. Other sessions still need ongoing monitoring.
The launcher never changes their limits.
A failed attempt retains its log, exit code and invocation record. Resume uses
the same source/environment/output; changing science or package versions
requires a new run. The stage independently validates parameter identities
before reusing each integrity-valid block.

The test helper `python -m horizon_position_d5.test_suite --inputs DIR
--import-mode=importlib eigsim/tests rotis/tests pointing_sim/tests
horizon_position_d5/tests -q` uses the study's hashed IERS tables. Synthetic
controls otherwise use a scoped offline policy, leaving unrelated frame tests'
Earth-orientation policy unchanged.

For a future attempt, `--persistent-jax-cache` records and enables a private
`OUTPUT/jax-cache` directory (created with mode 0700). The launcher refuses a
symlink, a directory owned by another user, or a group/other-writable cache.
This can reuse expensive compiled executables after the study clears its
in-memory caches, and on sequential resumes; numerical checkpoint identity and
all response checks remain unchanged. It cannot change an already running job
or recover a compilation that failed before being cached. Cache use is part of
the attempt record, so timing comparisons must state whether it was enabled.
The launcher explicitly sets the cache enable switch in either mode, overriding
an inherited environment value. The default leaves persistent caching off. See the official
[JAX cache guide](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)
and [`clear_caches` contract](https://docs.jax.dev/en/latest/_autosummary/jax.clear_caches.html);
the installed JAX 0.9.2 implementation was also checked.

The stdlib-only launcher also accepts an absolute script-path invocation through
`uv run --no-sync python /private/path/launch.py`. This permits a separately
committed launcher to run against an older, still-pinned scientific checkout
without invalidating its checkpoints. Record that launcher's commit and the
execution record's file hash. Only its own ancestor wrappers are exempted from
output ownership; competing copies and science runners remain protected.

## Reference-convergence planning

The next numerical-feasibility stage is specified in
[`../docs/specs/2026-10-15-d5-position-feasibility.md`](../docs/specs/2026-10-15-d5-position-feasibility.md).
The implementation provides a strict plan, a read-only task inventory,
pure NumPy convergence diagnostics, and checkpointed execution. The command
below is read-only; execution requires explicit `--execute` and a frozen gate.

```bash
uv run python -m horizon_position_d5.reference_convergence \
  --plan horizon_position_d5/plans/reference-convergence-v1.json \
  --pilot /path/to/verified/pilot \
  --ensemble /path/to/verified/ensemble --dry-run
```

JSON is printed to stdout. The supplied plan pins the completed study's result
manifest hashes and numerical criteria. The planner verifies every exported
block, the pilot/ensemble relationship and coordinate agreement. It derives
the old worst observed 87.5 m derivative position from coordinate labels,
deduplicates it against nominal, and inventories signed E/N/U stencils across
bearing and mask resolutions. Trace tasks are shared by mask evaluations.
Every task is `not_run`; inventory creation supplies neither timing evidence
nor a resource gate. Imported artifacts retain their original producer identity.

`convergence.evaluate_stencil` takes explicit refinement levels, axis/sign
labels, named cell coordinates, presence masks and edge flags. It returns
left/right/central slopes, signed refinement differences and per-cell forward,
one-sided and common-slope statuses. Missing cells cannot pass, and distinct
stable one-sided slopes are preserved. Comparisons use the finest consecutive
triplet; pass prefixes to retain earlier verdicts when extending a study.
`difference_summary` retains signed means, RMS, p95, maxima, finite/nonfinite
counts and the worst cell's coordinates. These are numerical diagnostics on
the tested scales, not a physical uncertainty or continuum error certificate.

Run the lightweight controls with
`uv run python -m pytest horizon_position_d5/tests/test_convergence.py
horizon_position_d5/tests/test_convergence_plan.py -q`.


### Checkpoint execution and probe

Use the existing `horizon_position_d5.launch` entry point on Limbo with phase
`reference-probe` or `reference-convergence`. Both take `--plan`, `--pilot`,
`--ensemble`, `--inputs`, `--output`, `--workers` and the existing producer,
resource and gate arguments. Run the launcher with `--dry-run` first.

The gate additionally pins `plan_sha256` (file bytes), `inventory_hash` (from
this clean producer's planner), `workers`, `allocated_cpus`, `max_seconds`,
`memory_high_bytes` and `memory_max_bytes`. Its existing `producer_commit`,
`allowed_phases`, `scope` and `timing_basis` remain required. Allocate memory
from the observed/probed footprint and current host headroom. For the initial
probe, document the existing-run evidence used for its provisional cap. The
launcher freezes plan/gate copies and requests systemd memory limits alongside
nice 0, CPU affinity/quota and a runtime cap. The runner verifies the frozen
plan/inventory again before writing and requires single-thread BLAS settings.

Probe mode traces only the nominal center at every planned bearing spacing,
then evaluates every mask setting for those horizons. It uses the same run
identity, trace keys and evaluation keys as full execution. Each probe retains
its own timestamped report; `PROBE_REPORT.json` points to the latest report.
It records per-task timings, reuse, worker/parent peak RSS, full planned counts
and the probe scope. A probe writes no complete `RESULTS.json`. Its timing
includes cold compilation where incurred; reused blocks are explicitly marked.
These measurements do not automatically approve a full run or demonstrate
parallel speedup. Include all probe/attempt time in the phase budget, and allow
at least 25 percent headroom when projecting the remaining work.

Full execution reuses only identity- and content-verified checkpoints. Trace
workers run in bounded batches; the parent is the sole checkpoint writer.
Temperature evaluation holds one compiled bearing/mask operator at a time and
processes heights sequentially. Evaluation keys include the actual trace file
hash. Output locks reject concurrent writers. Changed run identities, missing
run ownership records and attempts to overwrite a complete run are rejected.
Failed attempts retain their records and complete checkpoints.

The result manifest hashes every raw trace and temperature block, plus:

- `coordinates`: positions, heights, time/pointing/frequency coordinates and all
  refinement levels, axis labels and displacement signs;
- `stencil_temperatures`: center `(bearing, mask, position, height, sample,
  frequency)` and displaced `(bearing, mask, step, axis, sign, position, height,
  sample, frequency)` temperatures, presence arrays and edge exclusions;
- `convergence`: signed slopes, all numerical comparisons and per-cell statuses.
  Nested diagnostic keys use `__` separators in the NPZ export.

A completed result requires every planned block and every stencil destination.
A numerical convergence failure is a retained result and can have a successful
process exit. Missing/nonfinite computation cannot create a complete result.
Operational completion still requires the exact invocation's exit/journal
and transfer/hash checks, independently of numerical pass/fail statuses.
