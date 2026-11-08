# C56: isolated HFSS clone preparation (MOD129)

This increment prepares the **source gate**, not a converged beam or a loss
prior. Its only executable vendor phase copies the pinned design to a new
project, saves that new project, proves a conservative saved-source signature,
and stops at `CLONE_VALIDATED`. No AEDT operation has been executed for this PR.
There is no cleanup, domain/setup edit, Analyze, batch solve or export entry
point in this increment. Those are explicit follow-up phases after review.

## Why this experiment

Notebook 064 separates native reported radiation efficiency, far-field flux and
direct material-loss integrals. The saved 400 mm ordinary radiation boundary is
one numerical hypothesis for their disagreement; it is not an established cause.
The observed model contains aluminum, FR4 and PEC boundaries. These material
assignments do not establish the loss or geometry of the deployed feed/coax.

MOD129 predeclares two **fresh single-frequency 50 MHz** ordinary radiation
boundary controls, 400 and 1500 mm absolute padding on all six axes. Existing
400 mm results used 33 adaptive frequencies and remain a separate legacy
control. Fresh 400 versus legacy 400 exposes setup/mesh changes; fresh 400
versus fresh 1500 tests padding under the same numerical rule. Two radii at one
frequency cannot establish band-wide convergence or justify adopting a loss prior.
No PML, field rescaling, symmetry, phase fit, SHT or downstream default changes.

## Frozen authority and strict clone check

The packaged `hfss_boundary_source_v0001.json` pins the source project SHA256,
exact design and a reproducible saved-project signature. Source path on acme2:
`/data/cbye/eigsep_hfss/20261106_bowtie_1mhz/project/BowTie_Strip3_Opt8_Balun.aedt`.
The original 23 GB results tree is neither copied wholesale nor cleaned.

`hfss_boundary_pilot.py` is standard-library code compatible with vendor
IronPython 2.7. It parses named AEDT blocks without executing the source. It
retains object/face IDs, nine physical top-level parts, all operand histories,
coordinate systems, design/project variable expressions, project datasets,
complete project definitions/materials and embedded dependency manifest,
source excitation, lumped port/PEC definitions, sphere and mesh settings.
At this **clone-only** stage full ModelSetup, BoundarySetup, AnalysisSetup and
Optimetrics snapshots must also match. No numerical domain/setup difference is
allowed yet. Only `ModTime=` lines are omitted from canonical signatures.
The raw source SHA remains exact. No ID renumbering is accepted silently.

A changed definition, missing material/variable/dependency, unknown object,
wrong units, unbalanced named block or unexpected CopyDesign serialization
stops the phase. This conservative check may reject harmless vendor metadata
changes; preserve the failed draft and inspect the exact diff before proposing
a narrower rule. Never repair from local material libraries or loosen a check
after inspecting physics results. The parser is not a general AEDT file editor;
it preserves embedded data as source signatures and never rewrites project bytes.

## Operational contract

1. Root records a concrete clone-prepare decision and fresh all-user host, RAM,
   pressure, disk and cbye-process check. No active HFSS session may be taken over.
2. Stage the exact committed guard, vendor template and pin in a new private
   output prefix. `plan.py` only renders a vendor script and prints its hash;
   it never starts AEDT or grants a gate. Gate JSON records schema
   `eigsep.hfss-boundary-prepare.v1`, phase `prepare`, source/target/private prefix,
   case padding, source SHA, canonical source-pin SHA, rendered-script SHA,
   guard SHA, `approved_by: root` and a nonempty decision-record pointer.
3. Use a fresh isolated `ansysedt -ng -RunScriptAndExit <rendered.py>` process.
   Before any vendor action, validate exact source/code/contract identity and
   absent target/results paths. All paths must be absolute; realpath checks
   reject symlink escapes, source overlap and broad shared prefixes.
4. Reject any desktop with open projects. Open the source; CopyDesign only the
   target design; NewProject/Paste; **SaveAs(new_private_path, False)**. This is
   the first write, before any possible future cache cleanup. Never Save the
   source. Require the actual active desktop project path/name to identify the
   approved saved clone, then compare the saved signature and source hash.
5. Write `CLONE_RESULT.json` with hashes and `CLONE_VALIDATED` or `FAILED` plus
   source identity after failure. No automatic retry/repair/overwrite. Closing
   the isolated process must not save the original project.

Proposed clone-only external gate: one CPU, nice 0, 8 GiB, 900 s, new private
TMP/cache/output paths, detached job/log/sentinels and before/after project,
solution-index and stored-field identity. This proposal grants no launch.
The script has not yet been tested inside installed AEDT; focused tests exercise
real parsing and a deliberately limited fake desktop API, not a solver substitute.

## Follow-up source and resource gates

After clone validation, separately implement/review cache cleanup **only on the
saved and active clone**, fresh setup and region replacement. Preserve external
materials/variables/dependencies. Recreate AutoOpen1/RadiatingSurface identically
in both controls, six Absolute Offset paddings, ordinary radiation, no PML/FSS/
adaptive IE. Fresh 400 mm must match legacy domain bounds/type, units and physical
source signature; `bounds_check` predeclares 1e-9 mm CAD serialization tolerance
and 1500 mm outward displacement of 1100 mm on every side. That helper is a
contract test only; no actual region has passed it in this PR.

Fresh `C56_50MHz`: single frequency, no sweeps, DeltaS .005, maximum 30/minimum
1/converged 1 passes, 30% refinement, no imported mesh, Direct Solver, first-order
basis, original lambda/material/port/IE rules, SaveAnyFields true and
SaveRadFieldsOnly false. Compare complete saved settings rather than accepting
vendor defaults. Verify no stale meshes/fields for this new setup before solve.

First **400 mm solve** proposal remains four CPUs, nice 0, 48 GiB/6 h hard caps,
at least 64 GiB fresh available RAM and separately budgeted private cache/disk.
Legacy solves used about 34–36 GB; fresh 50 MHz cost is unmeasured. Explicit batch
selection is `<design>:Nominal:C56_50MHz` with `-distributed -machinelist
list=acme2:1:4`, enforced by OS affinity/quota, no global DSO changes. A measured
400 result and another root gate precede any 1500 solve. Stop on failure/caps;
no automatic extension or retry.

A later exporter must preserve raw complex Cartesian rE, native coordinates,
both 50/75-ohm S33 and reference-invariant Z, reported incident/accepted/radiated
powers and direct surface/volume loss arrays. Calculator replicas are one global
integral only after exact scalar/cardinality validation; never sum them as data.
Keep raw scalar/field normalization disagreements visible. Set accuracy criteria
before scoring. Neither a successful clone nor this pair adopts a beam or loss
prior. Existing artifacts, PR44, PR99 and PR101 remain unchanged.

### 2026-11-08 native serialization counterexample

The first actual clone-only unit ended in `FAILED` before `OpenProject`:
IronPython's JSON encoder treated a native embedded-file byte `0xf8` as UTF8.
No clone was created; the project, solution index and three field coefficients
retained their exact hashes. The RPCSS endpoint warning in the engine log does
not establish the cause of this Python exception.

Canonical serialization now recursively decodes only Python2 byte strings as
Latin1, matching the native text reader. Python3 Unicode and its existing
signature `09255b08...fe29` must remain exact; Python3 arbitrary bytes remain
invalid JSON. This preserves embedded data rather than dropping or replacing
non-ASCII values. Regressions check nested non-UTF8 bytes, mutation sensitivity,
native embedded dependencies and unchanged Python3 digests. The standalone
`check_serialization.py` additionally compares the actual source pin in CPython3
and installed IronPython without importing Desktop or invoking a license API.
This amendment requires complete configured tests and a new clone-only gate;
the failed native attempt does not authorize a retry or any solve.
