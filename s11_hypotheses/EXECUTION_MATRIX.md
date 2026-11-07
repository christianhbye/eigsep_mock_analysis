# C54 measured next-execution matrix

Frozen after the original probe completed2026-11-07T21:43:53UTC. Root
copied it to workspace products/_orchestration/c54-s11-hypothesis-tests/
probe-20261107. The verification script/result are in author scratch;
all four scientific keys, recorded result hashes and raw identities pass.
Geometry18.785CPU-s; native46.747CPU-s; fine877.561CPU-s; raw control0.017CPU-s.
Whole probe958.387wall-s/958.269CPU-s, peak1415784KiB,4923767bytes. Original
1200s cap held. No extension/rekey/relabel of the old probe is authorized.
The250MHz common native/fine PO difference is1.55e-18, diffuse3.17e-19;
physical offsets, antenna, rotation, roughness, shell edges and native masks
are identical. Diffuse covariance is PSD within numerical precision.

## First complete physical comparison

Preserve16members, all three epochs(night,13:10,19:17), radius600m,stride4,
incidence1deg and all original scientific defaults. Declare all96 grid tasks:
48native+48fine. `EXECUTION_PLAN.json` lists exact task names and four
nonoverlapping partitions per stage. Stageprimary is48native+3fine(member0
at each epoch), and stagefine is the remaining45fine tasks. No physical
member is dropped or changed. The primary power/AC5 and complex-epoch tests
need all48native and the member0 fine illustration; they need not wait for
all45remaining fine exports. Those remain pending until completed, and no
fine-ensemble uncertainty claim precedes them. Native ensemble carries the
16-member MC uncertainty.600m remains a domain limit, not range closure.

Stage order and provisional hard bounds (root rechecks host resources):

| Stage | Matrix and measured projection | Proposed cap/job |
|---|---|---|
| Geometry | Three600m/stride4 geometries, about57CPU-s plus startup |1CPU,4GB,300s,128MBproducts |
| Controls | Full blinded source/control selection and historical reproduction; original0.017s raw unit is not a benchmark of all SVDs |1CPU,4GB,3600s,512MBproducts; capture actual cost |
| Independent raw score | Full feature/injection/OSL/skirt reports after design.lock; no physical dependency |1CPU,4GB,7200s,2GBproducts; preserve completed result |
| Primary physical |48×46.747+3×877.561=4876.5CPU-s, plus startup; four shards have at most one fine unit each |1CPU,4GB,3600s,512MB/shard |
| Remaining fine |45×877.561=39490.2CPU-s; four shards11/12fine tasks |1CPU,4GB,14400s,512MB/shard |
| HistoricalR0 | Six native first-hit comparisons at ns64/256; not measured by the facet probe |1CPU,4GB,3600s,512MBproducts |
| Verified gather+primary score | Collect unchanged unit bytes, reuse completed independent raw outputs, fit powers/AC5 and report all epochs |1CPU,4GB,3600s,2GBproducts |

The physical total is about12.3CPU-hours from measured native/fine units,
or12.8CPU-hours if the entire probe overhead is repeated per member. Budget
16CPU-hours before further evidence; four single-thread workers project
roughly3.2wall-hours, with a4hour per-shard cap on the longest fine stage.
Do not reserve two CPUs for a single-thread worker merely because the old
probe cap was2CPU. All cbye CPU use combined stays≤8 logical CPUs. While
the existing code full suite consumes up to2CPU, four physical workers and
one raw worker total at most7CPU; use2–3physical workers under RAM/I/O or
other-session pressure, or until a fresh host check supports four. Reserve
4GB/job; measured RSS was1.35GiB, but concurrent DEM copies still need RAM
headroom. Root starts/stops no teammate jobs to make that budget fit.

The new source rebuilds night/member0 native and fine in the primary
partitions. Compare those arrays/physical metadata to the preserved probe
before interpreting the gathered science. It is a fresh new-source replica,
not adoption/rekeying of the old feasibility output.

## Bounded range/angular closure ladder

Do not launch a radius×stride×bin×epoch×member Cartesian product. After the
primary comparison, compare matched night members0–3, native grid only,
one factor at a time against the available600/4/1 primary units:

1.1200m/stride4/bin1: first radius increment; first-case cap900s.
2.600m/stride2/bin1: facet refinement; first-case cap900s.
3.600m/stride4/bin0.5: incidence refinement; first-case cap300s.

Each first case is one1CPU/4GB worker with512MBproduct cap. Root measures
it before authorizing the remaining three matched members. Nominal phase
work for the first two is4×native; a conservative visible-cell bound is
about6.4×the observed case before extra visibility work. The bin refinement
has similar cell/frequency work. A4-member comparison is diagnostic with
explicit MCSE; if close enough to support the5% and0.2spread threshold,
complete the declared16-member comparison before claiming closure.

If those fail or material differences remain, next discriminate with
2400m/stride4/bin1 and600m/stride1/bin1, one native member each,2400s cap
before sizing additional draws. These can be16×nominal phase work(about
25×under the conservative visible-cell bound). Range growth also increases
visibility work, so these are caps, not promised runtimes. Then double
radius only while the exact DEM disc remains valid. ExistingR0long paths
and its last-shell geography must guide that continuation: two quiet
nearby increments do not prove absence of a distant wall. Report truncation
and maximum valid domain if this cannot close. Fine-grid exports for a
new domain follow only after its native prediction is scientifically
material and resource-sized. No numerical/angular alias is mistaken for
frequency-grid folding.

For these bounded tests, set `execution_draw_selection` to explicit
configuration/epoch/member triples, `execution_grids:["native"]`, and
include the full physical configuration in `terrain_configurations`.
The plan retains unselected draws visibly. All scientific defaults remain
unchanged and task keys exclude partition selection. Use the matching
primary reference files in convergence analysis; diffuse expected
power/covariance, deterministicPO complex change and last-shell change are
reported separately, without refitting a new scale to hide convergence.

## Execution and gather contract

Use new root-gated run directories, not the old probe. `geometry` writes
shared immutable geometry once. Each physical shard sets its own `run_dir`
and the same read-only `geometry_root`, with `execution_shards:4`. After
the fresh source-only dry run, gates for physical-shard/gather must also
pin the printed `plan_sha256` and allowed `stages`(primary/fine/all).

```sh
../code-060-s11-hypothesis-tests/.venv/bin/python -m s11_hypotheses.run geometry --config GEOMETRY.json
../code-060-s11-hypothesis-tests/.venv/bin/python -m s11_hypotheses.run controls --config RAW.json
../code-060-s11-hypothesis-tests/.venv/bin/python -m s11_hypotheses.run raw-score --config RAW.json
../code-060-s11-hypothesis-tests/.venv/bin/python -m s11_hypotheses.run physical-shard --config SHARD0.json --stage primary --shard-index 0
../code-060-s11-hypothesis-tests/.venv/bin/python -m s11_hypotheses.run physical-shard --config SHARD0F.json --stage fine --shard-index 0
../code-060-s11-hypothesis-tests/.venv/bin/python -m s11_hypotheses.run r0 --config R0.json
../code-060-s11-hypothesis-tests/.venv/bin/python -m s11_hypotheses.run gather --config COLLECTION.json --stage primary
../code-060-s11-hypothesis-tests/.venv/bin/python -m s11_hypotheses.run score --config COLLECTION.json
```

Repeat the four predeclared indices, each in a distinct immutable root.
`design_lock` may point at RAW/design.lock.json. COLLECTION lists exact
`shard_roots`, `geometry_root`, `raw_science_root`, `r0_source_root`. Gather
reconstructs scientific keys, verifies source/input/plan identities,
coordinates, shared geometry, every unit checksum and complete nonoverlap
coverage, then copies original unit bytes unchanged. `GATHER_primary.json`
records their original roots/hashes and all45pendingfine units. Later gather
all eight primary/fine shards with `--stage all`; use a fresh collection or
the byte-identical existing destination. It never renames old-source output
as new science. Independent raw output reuse similarly verifies its exact
source/input/lock key and copies original component bytes with provenance.

Each shard has immutable RUN_RECORD_physical-shard.json, EXECUTION_PLAN.json
and SHARD_DONE.json. Incomplete shards cannot be gathered. Full suites and
standaloneCI wiring remain final-conformance work. Root launches; this
author phase only verifies the old probe and freezes scheduling machinery.
