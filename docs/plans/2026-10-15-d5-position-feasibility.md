# Implementation sequence: D5 numerical feasibility

Prepared 2026-10-15 from the [spec](../specs/2026-10-15-d5-position-feasibility.md).
CHB chose numerical feasibility first. This planning PR contains no new science
result or computation launch. All heavy stages carry **LIMBO-COMPUTE**.

## PR A: reference-convergence driver and reporting schema

Implement first; base on merged mock #34, following any path migration in #35.

1. Add `convergence.py` with pure, coordinate-aware calculations of left,
   right and central slopes, independent bearing/mask/step comparisons, and
   per-cell outcome flags. Keep numerical targets in a versioned plan object.
2. Add `reference_convergence.py` with `--plan`, `--inputs`, `--pilot`,
   `--ensemble`, `--output`, `--dry-run` and an inventory-only planning mode.
   Resolve the two development positions from hashed upstream artifacts;
   serialize actual coordinates, all settings and a deterministic task graph.
   Proposed command names are not available until this PR is implemented.
3. Reuse the independent tracer and block cache. Include bearing arrays,
   absolute position, height convention, DEM and tracer identity in trace keys;
   include the response identity and mask subsampling in evaluation keys.
   Deduplicate traces independently of response evaluations. Verify imported
   upstream results without pretending they were produced at the new commit.
4. Save every displaced horizon and temperature. Batch/stream mask evaluations
   to a measured memory limit; avoid materializing the complete cross-product
   of dense masks. Preserve split-orientation/no-AVX execution support.
5. Extend `launch.py` for this phase and its frozen plan. Retain dry-run
   non-mutation, resource checks, source/input hashes, output locks, attempt
   records and resume rejection on any identity change.
6. Add meaningful synthetic controls: constant horizon; smooth manufactured
   temperature with known slopes; cusp with different stable one-sided slopes;
   angular aliasing where a central slope looks stable but one side fails;
   corrupt or incomplete block; frame/axis/sign permutation; boundary stencil;
   cached resume and serial/spawn agreement. Tests must demonstrate each report
   outcome, including failures and missing-work status.
7. Run relevant reference, cache, launch, temperature and eigsim tests; run Ruff
   check and format checks. Track the existing repository-wide lint defect in
   #35 rather than disabling it or narrowing the test selection.

Done: reviewed runner and dry-run inventory; tests pass; no scientific pass
claimed. Private PR to christianhbye/eigsep_mock_analysis, CHB merges.

A companion analysis-code PR extends the strict reader and notebook 026 for
these exports. It starts from main after #24 merges. Before a real run, use tiny
manufactured exports to check array coordinates and failure reporting. After
the run, commit executed notebook outputs and independently reconstructed
arithmetic. The generator owns simulation; the notebook owns interpretation.

## Run A0: cost and correctness probe on Limbo

Before any launch, refresh the read-only team survey and check the completed
ensemble's exact invocation status. Inspect all cbye jobs and their reserved
peak CPU allocations, host CPU/memory/I/O pressure, available RAM and disk.
Use a separate registered private checkout/output; never alter the completed
producer, its environment or the shared tree.

Probe one representative complete trace at each proposed bearing spacing and
one full-size mask evaluation at each proposed subsampling level. Also measure
response loading/building and process-spawn overhead. Save wall time, peak RSS,
bytes written, numerical reproduction and serial/worker agreement. Inputs and
code must be pinned; use the existing verified no-AVX environment or a separately
validated replacement. A laptop test does not establish Limbo compatibility.

Derive the exact remaining block count after verified reuse. Estimate total
wall time using measured costs, worker allocation and at least 25% runtime
headroom. State assumptions and warm/cold compilation cache status. Probe work
counts toward the phase budget and is retained as checkpoints when compatible.

Proposed initial budget: at most six hours wall time including the probe, nice
0, up to four single-thread workers if resource sharing permits. Aggregate
reserved cbye allocation must remain <=50% of host CPUs (currently eight CPU
equivalents). Choose memory limits from measured peaks and current host
headroom; affinity bounds CPU use when cgroup quotas are unavailable.

If the complete first matrix does not fit, return a costed reduced-scope or
extended-budget proposal before launch. Do not drop unfavorable cells or a
required refinement comparison to obtain a pass. Freeze `gate.json` once;
resume consumes it unchanged. An observer must be read-only and completion
must be tied to the recorded InvocationID, exit and complete inventory.

## Run A1: first reference matrix

Run the two-position, three-bearing, three-mask, three-step matrix in spec
section 4 with the frozen plan. No dense training grid runs alongside it as a
speculative next step. Stop at the runtime cap and preserve checkpoints.

Collect into a fresh immutable local snapshot with remote-before/local/remote-
after hashes. Reconstruct slopes and comparisons independently in NumPy from
exported temperatures. Check all required cells and edge exclusions; review
forward and one-sided failures separately. Produce a short decision table by
height, observed/synthetic group, frequency and axis.

Possible next actions:

| Evidence | Next action |
|---|---|
| Forward and slope comparisons resolve | Run the breadth checks in section 5; then cost the spatial trial. |
| Forward resolves; slopes do not | Retain forward-only feasibility; investigate the failing angular/step control before claiming gradients. |
| Mask resolution fails | Independent integration oracle, then a separately reviewed eigsim improvement if justified. |
| Bearing resolution fails | Cost a targeted finer-bearing extension with unchanged criteria. |
| Inventory incomplete or runtime exceeded | Record `not_run`/`unresolved`; resume only with compatible source/config and an explicit remaining budget. |

## PR B and Run B: spatial refinement and domain contract

After reference breadth checks, extend the ensemble driver to consume the
accepted reference identity, a distinct query box and traced halo, and frozen
development/assessment sets. Start with existing linear/cubic eigsim evaluators;
implement only missing domain/manifest checks. Add manufactured polynomial,
join, overshoot, exact boundary and invalid-query tests.

Cost actual stored nodes: halo makes 3/5/9 interior nodes per axis become
5/7/11 stored nodes per axis. Do not reuse the old 0.1-degree values under a
new reference identity. Gate each expensive refinement from measured costs.
A spatial pass requires fresh off-node assessment and a further refinement
comparison; training-node reproduction is only an implementation check.

If uniform refinement is too expensive, report the cost and the failing cells.
An adaptive representation is a new design/test decision with fresh assessment,
not an automatic change hidden inside this run.

## PR C: intended likelihood and release preparation

Once numerical feasibility is known, freeze the actual data selection,
covariance, nuisance model and inference tolerances. Test direct versus
emulated likelihoods, recovered parameters, degeneracies and uncertainty
stability. Select gradient-based or gradient-free inference from those results.
No physical posterior or release follows from numerical screening alone.

The conditional release includes horizons, evaluator, regeneration command,
manifest, tested domain/scales and accuracy report. Update M008 by dated
amendment from the executed notebook. Canonical main reporting runs and the
workspace graduation procedure precede a shared memo/product contribution.

## First coding task and review boundary

The next coding task is PR A items 1--3: plan validation, deterministic block
inventory and pure convergence arithmetic, with synthetic failure cases. Then
add execution/export and launcher integration before the Limbo probe. This
planning PR can be reviewed independently of the cleanup issue and the memo.
Numerical thresholds and the six-hour allocation remain proposed until frozen
in the reviewed run manifest; they do not inherit the previous run's approval.

## Implementation checkpoint — 2026-10-15

The first increment implements PR A's plan validation, deterministic inventory
and pure convergence diagnostics. `convergence_plan.py` holds the plan and
upstream selection; `reference_convergence.py` exposes the read-only CLI.
The versioned plan pins the original pilot/ensemble RESULTS hashes. A dry run
on those verified exports derives two development positions, 114 trace tasks
and 342 mask evaluations. These are planned counts, not completed computations.

Manufactured tests exercise smooth and constant values, a cusp, angular
cancellation, step instability, axis/sign permutations, missing/nonfinite/edge
cells, corrupted upstream artifacts, coordinate mismatches and deterministic
selection. Existing cache/reference/launch behavior is regression-tested.

The next increment adds checkpoint execution, full raw exports and the
reference-convergence launcher phase, then the companion reporting reader.
The Limbo cost probe follows those implementation checks. No current CLI path
executes the proposed matrix or interprets its inventory as a scientific pass.


## Execution implementation checkpoint — 2026-10-15

The second increment adds `convergence_runner.py`, explicit CLI execution,
nominal-center probe mode and the launcher's `reference-probe` and
`reference-convergence` phases. It retains raw horizons/temperatures, verifies
checkpoint identities and trace-to-temperature dependencies, reconstructs the
coordinate-rich stencil, and exports the numerical decisions from the first
increment. Probe completion and full result completion have separate markers.

Small manufactured end-to-end tests exercise interruption/resume, corruption,
source/input ownership, concurrent-writer rejection and numerical failure with
operational success. Tiny real-tracer serial/spawn and all-sky/all-ground mask
controls check the actual numerical kernels. These checks do not replace the
native Limbo probe or establish D5 convergence.

The remaining pre-study work is the companion reporting reader, native Limbo
verification, measured cost/resource review and frozen launch gate. Code PR #24
is still the reporting prerequisite at this checkpoint. No D5 feasibility
computation or shared contribution was launched by this implementation.
