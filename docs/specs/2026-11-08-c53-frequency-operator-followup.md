# C53 required frequency and symmetry controls

Status: explicit remaining work after the primary symmetric18-channel/time comparison; no completion or adoption claim. This follows the existing C53 spec, not a new physical model.

The symmetric 18-channel fit is a partial C53 deliverable. The required raw-beam H/N controls and isolated centre-frequency correction remain open. The completed 10/60-second comparison uses the exact native model on both grids; it does not measure a centre-frequency or centre-time approximation.

The raw authorities are available, not missing hardware inputs: OLD v002 `29bbbc19…5750f` (52 × 130 × 258, lmax128) and NEW v003 `1ab98e67…5f68b` (201 × 130 × 258, lmax128). `RAW_ASSET_HEADERS.json` records their exact paths, byte hashes and frequency axes. The OLD cold control reproduces one scalar slice at 70.3125MHz; this is beam reconstruction evidence, not a raw-beam night fit.

1. **Small raw-beam discriminator.** Use the existing D642 producer/API with only `beam.path/role/sha256` changed to `old_raw_control` and `new_raw_control`. Keep channels `[283,603,843]`, the same 37,697 native rows, all362 parents/43 orientations, native masks, GSM/IERS/horizon, consumed034 tilt, pointing812, calibration60a, runtime830 and exact cubic frequency operator. Validate normalization/interpolation/negative-power support before modelling. Two sequential model jobs at the measured six-CPU setting should cost about 2×280s; propose each6CPU/8GiB/900s/512MiB with the existing guards. Reuse completed symmetric3 models. The existing fit CLI accepts both raw roles: same H/N parameters, load pivot/loss prior, training groups and fixed raw metric; radiometer sensitivity controls remain separate. The CLI also emits W when supplied, which can be reported without inventing a new term. This pilot does not discharge the required18-channel controls. Full18 raw expansion is twelve three-frequency model jobs, approximately56minutes sequential from current measured cost, followed by joined-model fits; gate that expansion on actual raw-asset/model cost/QC.

2. **Small isolated centre-frequency discriminator.** The three legacy nominal centres are channels `[288,608,848]` = `[70.3125,148.4375,207.03125]MHz`, while the actual data use six offsets per band. Those centre raw samples are absent from the frozen061 six-offset archives; requesting them from the current thin/materialize CLI would fail. No new raw measurement is needed to test a *model* frequency approximation. Prepare an explicit mock-only model-frequency operator control: preserve all18 data channel IDs/masks/native times, but evaluate sky/beam at the three nominal centres and map each result to its six data channels. Generate only three distinct model frequencies using the same native pointing and all other primary OLD-symmetric inputs. This needs a reviewed small source/API variant and rejection/coordinate tests; it is not implemented now. Budget one model job at measured6CPU/8GiB/900s/512MiB, approximately280s. Compare its three repeated native columns, using each channel's original masks/counts, with the completed exact18 OLD-symmetric columns. This isolates frequency evaluation without changing time averaging, beam, calibration or18-channel parameter freedom. Label it explicitly as the centre-frequency control, not recovered034 byte closure. A raw-OLD version can follow using the same operator once raw18 columns exist.

For both tests, fit only original training groups and score the **same held-out native raw cells with the same frozen training-reference variance**, original seven test blocks and10,000 shared paired block draws. Retain the approved MOD99 block/jackknife margin and95%intervals; do not create a frequency-independence claim or a new post-hoc threshold. A claimed predictive difference must pass the existing declared held-out rule; otherwise report its interval/upper bound. The centre-control template discrepancy should also be reported before fitting. Neither a lower fitted epsilon nor internal chi-square is the pass criterion.

An optional centre-*time* approximation profile is a separate numerical test. It is not currently used by the primary and is not supplied by the matched60-second control.

No control above has been launched, no source/API changed, and no extra model term or adoption is authorized by this proposal.

Primary audit bindings:

- `products/_orchestration/c53-readiness/operator-control-readiness/CONTROLS_INVENTORY.json` — SHA256 `99f324a2acb60d73071a299554de77256e087def3bc7227946cc0a31af3e14fb`.
- `products/_orchestration/c53-readiness/operator-control-readiness/RAW_ASSET_HEADERS.json` — SHA256 `b123243b72c21a3092404476fb2f8917d5629fb97bf744f5910b27a696808188`.
- `products/_orchestration/c53-readiness/operator-control-readiness/CENTRE_COORDINATE_CHECK.json` — SHA256 `203b9624e8d1f77b2074cc9c0ceabaf1f4b019f92a9c6382a59174fc53d32dfd`.

## Source implementation amendment — 2026-11-08

The preceding proposal is preserved as the scope authority. The centre-frequency
source/API is now implemented on the registered private branch; no model run,
fit, predictive comparison or adoption follows from source preparation.

The opt-in model input is `model_frequency_operator`, with exactly these fields:

```json
{
  "version": 1,
  "kind": "c53-nominal-centres",
  "data_channels": [283, 284, 285, 291, 292, 293, 603, 604, 605, 611, 612, 613, 843, 844, 845, 851, 852, 853],
  "evaluation_frequencies_mhz": [70.3125, 148.4375, 207.03125],
  "data_to_evaluation": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
}
```

The exact list must match the top-level `channels`. Null, extra fields, other
versions, nearest-frequency assignments, partial axes and changed mappings are
errors. This first implementation supports only this complete18-channel
control. Existing beam interpolation still rejects out-of-support frequencies;
normalization, symmetry, GSM, pointing, time evaluation and CMB restoration are
unchanged.

`fine_night_d5/model.py` evaluates the existing beam/sky/convolution on the three
unique evaluation frequencies. `FrequencyPlan.expand_and_reduce` expands the
result at each native row to18 columns **before** `streaming.reduce_columns`
uses each channel's actual accepted mask. A repeated native model therefore
need not have the same binned mean across channels with different accepted rows.
No raw channel is averaged, inferred or requested at288/608/848. The18 measured
data columns and their eventual18 per-frequency parameters remain distinct.

The output's existing `channels` and `frequencies_mhz` remain the18 **data**
coordinates. The control adds `model_evaluation_frequencies_mhz` (three values),
`data_to_model_frequency_index` (18 integer indices), and
`model_frequency_operator_version`. `RUN_DONE.json` separately records the
complete operator identity, including both frequency axes and expansion order.
Streaming checkpoints and beam/GSM QC retain their actual three evaluation
frequencies. The original data masks/native identities remain in `columns.npz`.

When the option is absent, the original native-frequency arithmetic, numerical
arrays, column keys and QC structure are retained. New source commits/hashes
still identify the changed producer; existing products are not rekeyed or
claimed to have run this source. The model's existing fresh-output refusal and
full input-manifest/source hashes bind the new declaration. `join.read_piece`
checks declaration/record/payload agreement and retains the new helper hash in
physical context. It rejects using a native payload as a centre control.
Chunk joining of a frequency-mapped product is explicitly unsupported: this
control is already a complete18-channel product, and joining must not erase
its coordinate metadata. The normal historical/native chunk protocol remains.

Focused tests use small independently specified arrays and genuine on-disk
metadata checks; they do not invoke GSM/SHT, native sky evaluation or fits.
They check distinct masks within one band, zero-support cells, repeated native
values with18 separate parameter coordinates, exact default reduction parity,
nonfinite/ambiguous mappings, existing beam support, changed cache/payload
identity and prohibited metadata-erasing joins. Their results are reported in
the draft PR against the frozen source commit.

Before marking that PR ready, run both the complete standalone workflow command
(`uv run --project fine_night_d5 --no-sync python -B -m pytest
fine_night_d5/tests -v`) in its own unchanged standalone lock/runtime, and the
three existing workspace test commands (eigsim+horizon, rotis, pointing) under
the workspace lock/native JAX0.9.2. Package and workspace locks are different;
neither environment substitutes for the other. Freeze source/lock/runtime/test
inventories, use a fresh gated private output, record XML/exits/skip reasons,
and report missing hosted CI/CodeCov honestly. Proposed resources are1CPU,
2GiB,180s for the light standalone suite, then4CPU affinity,16GiB sampled
sum-RSS,3600s for the sequential workspace shards on an isolated registered
private host. These are proposals, not launches; root checks resources/gates.

The later actual centre-model discriminator retains its separate reviewed
six-CPU model resource gate. Before any scientific score, inspect the declared
frequency identities, native mask/count equality, interpolation/negative-power
QC and exact model-array scope against the unchanged18-channel OLD-symmetric
baseline, then use the frozen training/reference metric and shared seven-block
draws above. No lower fitted epsilon or source test pass establishes a science
result.
