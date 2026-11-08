# C53 private combined-refit pilot — concrete source-preparation spec

**Diagnostic proposal; no fit/run/source edit authorized by this file.** Private
implementation can start after root registers notebook/worktrees. Shared release
adoption is pending; it need not block a separately pinned private diagnostic.
No063 notebook, script, worktree or product was found locally or on private Limbo
in the duplicate-work check. Root alone allocates/registers063.

## Concrete first grid

Use phase-C box-air input4 on the frozen034 reference night: the original362
parent bins, original318 fitted-bin mask, train/test memberships, blocks and wind
selection. Split each parent row interval into left-anchored **at-most10s**
subbins, preserving its original parent/raw-bin ID and group labels. A partial
last subbin remains partial. Do not merge across parent boundaries or introduce
a minimum-exposure/RFI cut. Zero-support cells remain explicitly unsupported.

Fit **18 individual244.140625kHz native channels**, not six-channel averages:

- Centre288 (70.3125MHz):283,284,285,291,292,293.
- Centre608 (148.4375MHz):603,604,605,611,612,613.
- Centre848 (207.03125MHz):843,844,845,851,852,853.

These are the existing six offsets[-5,-4,-3,+3,+4,+5] at three baseline bands,
all inside old/new beam support. This is a bounded finer-channel pilot, not a
full-band release or an alias-free spectrum. The periodic/gapped support is
retained and its delay response must be shown; do not silently use all channels,
apply a prefilter, fill samples or claim that the512ns ambiguity disappeared.

## Inputs and work that can start privately now

`INPUT_MANIFEST.json` binds the existing source/contracts and known product pins.
The canonical061 diagnostic is merged/reproduced; shared consumer/product release
adoption remains pending. Its original immutable native run is still usable for
this scoped diagnostic through the accepted specific-reader attestation.

Each existing native unit has axes(row,51 centres,6 offsets); the inspected first
unit has6706 rows and51×6 frequency coordinates. It retains int32 raw powers,
file/original-row/time_best IDs, duration, switch/antenna eligibility, flag codes,
rejection reasons/accepted mask, per-native scale/offset/calibration coverage,
ENBW and thermistor/calibration-donor identities. Its binned companion retains
counts/exposure and historical/window/native calibration variants. The18-channel
pilot can thin these archives directly; it does not need a full-campaign raw
reload or a new dense-channel calibration product first.

**Redo for the new grid:** evaluate historical and exact inclusive-window
coefficients at each new10s bin, using pinned tcalv0004 solutions/APIb459c6c;
recompute all three reductions on identical native rows and channel support.
Saved old60s coefficients are not relabelled10s coefficients. Native scale/offset
and identities can be reused from the attested archives only after exact
frequency/time/source/CRC checks. Resolve time-bin membership before channel
flags; thermistor window bounds come from those actual bin rows, not±60s.
Preserve counts/fallbacks and repeated thermistor/calibration donor IDs.

First implement a private thin/rebin materializer and a calibration-only
existing-resolution bridge. The local corrected producer is a20fbcd; its guard
fix preloads the lazy reader, not a new calibration arithmetic rule. The local
bin-window companion is c3959b4. These sources can materialize a private versioned
snapshot without pushing to the shared repo, replacing a release, or waiting
for CHB's shared merge. Keep their diagnostic label and actual source identity.

## Calibration and resolution ablations

Use three fixed recipes, no new fitted gain/interpolation model:

- H: historical bin-after-average, nearest load reading at the bin time.
- W: bin-after-average, inclusive actual-bin-row thermistor window (MOD119).
- N: calibrate each native integration, then count-weight average; each row's
  own nearest thermistor lookup, no second window smoothing.

First reproduce historical034 at its old60s/six-offset operator and frozen old
beam/design/weights. Existing061 already refits the matched diagnostic with
frozen034 sigma/design; it is not a released full pipeline/noise rebuild.
Compare H/W/N at that fixed old operator before combining beam/resolution changes.
Separately quantify replacing the old centre-frequency approximation by the
actual native-count model average. Do not bundle that correction into a beam win.

For the finer fit, all calibration/beam comparisons use the same18 channels,
same10s subbins and same per-frequency theta/P freedom. A60s×18-channel comparator
keeps that same freedom when isolating time resolution. The historical three
six-offset means are a reproduction/aggregation bridge; do not score a3-frequency
model against18 independent-frequency parameters as a pure resolution test.

## Matched beam operator — MOD24 primary, raw explicit controls

The ready scalar candidate v003-diagnostic has SHA
1ab98e6707f686336d49302fc260357521f11dee97826aeaf559df4ef055f68b,
201 native50–250MHz slices, MWSS lmax128, raw sum|E|²/no complex symmetry.
Its common complex L180 parent is9b405524…672de. This is **not** the same effective
beam as055's complex-body-x-symmetrized comparison merely because the parent agrees.

Primary old/new fields must both use the existing MOD24 projection
`E_sym(n)=[E(n)-P_x E(P_x n)]/2`, `P_x=diag(-1,1,1)`; Cartesian parity(+1,-1,-1).
Use the exact055 `body_x_symmetry` mirror, x only, before squaring. For both:
full native MWSS quadrature normalization to directivity integral4pi; scalar-power
SHT to lmax128 by the same transform; no realized-efficiency/loss insertion or
extra S11 reference transfer. Pin derived old/new **symmetric** artifact hashes;
they are not yet generated/pinned. Derive them privately from old complex native
SHA8c63a213…92924 and new native9b405524…672de. Reuse the symmetry operation, not
055's different cell-area normalization. Existing raw v002 and raw v003 are
explicit controls/legacy bridge; they never silently replace these primaries.

At the18 fractional native frequencies, predeclare the already documented
not-a-knot CubicSpline of normalized scalar-power slices, directly from each
native frequency grid (not via the old interpolated `_1mhz` file). Same operator
for old/new and symmetric/raw. Check withheld slices, normalization and any
negative interpolation power; bound its effect on the sky columns. Do not clip,
swap interpolation or invent a noise floor after seeing the night score.
This is an input/numerical gate, not proof that HFSS matches the built antenna.

Primary fit grid: H/W/N × old/new symmetric beam (six fits; H/N endpoints are
primary2×2 and W attributes thermistor versus ordering). Raw old/new with H/N
are four explicit operator controls. Sky, horizon, pointing/loss priors and
load centring/gauges stay pinned; no C51 correction/coherent term, new gain fit,
new pointing solution or terrain-filter term is silently added.

## Actual estimator, validation and scored observable

Carry measured/excluded/filled identities separately; this primary uses measured
samples with weights0/1 and no fill. Freeze the present v3+explicit consumer EMI
policy and calibration/antenna validity; no unpublishedC9 policy migration is
hidden in this test. Native original masks/reasons and every cell→subbin→parent
mapping remain exported. Use common valid native support across H/W/N/beam
comparisons, and report any variant-only coverage separately.

Model the bin with `sum_i w_ic A(nu_c,time_best_i,pointing_i)/sum_i w_ic`, including
actual channel weights. Do not assume a centre model equals this average.
Use the pinned v5 pointing/raw metadata and original034 mount/IMU convention;
the native calibration archive lacks row pointing, so those inputs must be joined
by exact native identities. Saved034 rotations provide the old-grid closure,
not an invented fine-pointing solution. A centre-time approximation may be
profiled against the exact native operator but needs its own error/cost gate.

Fit theta/P/load pivot on training groups only; keep current loss prior and
switch/free-space plane treatment. Predict held-out **raw powers on identical
native cells**, using each calibration recipe's explicitly recorded scale/offset
forward map. Freeze one training-only reference noise/score metric across
variants so a larger candidate variance cannot buy a better score. Report ideal
count-weight radiometer prediction separately from empirical scatter, fitted
excess and cross-time/frequency covariance; do not treat donor repeats as
independent or adopt a new covariance term without review. Preserve MOD123's
ENBW=channel-spacing/independent-power limitation. Use the existing MOD99 paired
held-out/group uncertainty rule, not independent frequency error bars.

## Bounded implementation and cost gate (not launched)

After source/fixture review, propose a private profiling unit2CPU/4GiB/180s/nice0:
three deterministically selected fitted parent bins (earliest,middle,latest in
UTC order), all their native rows and18 channels. Record cold/warm setup,
source/input hashes, exact model averaging versus10s-centre approximation,
coverage and peakRSS/time. No fullnight fit follows automatically. Beam symmetric
artifact generation has its own source/resource gate; raw v003's measured full
201-slice cost was192.2s outer/1.18GiB RSS on1CPU, not a guarantee for new work.

At most318×6=1908 fine time bins ×18=34,344 reported cells are planned before
zero-support accounting, versus034's13,430 selected cells. **This is not a cost
scaling law:** exact native-time convolution can require far more evaluations.
The native archives already contain the18 raw/calibration channels, so data
materialization is bounded; model averaging and compilation are the measured
cost gate. Atomically checkpoint per stage/channel/time chunk from the start,
with actual QC serialized before figures: do not repeat C51's lost-RAM summaries.

Remaining gates before a production claim: source/conformance and measured cost;
symmetric-beam hashes plus numerical interpolation/normalization checks;
old-grid calibration and model-averaging closure; common-support held-out results;
shared calibration/beam release adoption and material systematics. File-only
private preparation/materialization can proceed now; official releases and
physical-plane uncertainty limit promotion, not implementation. No new ID,
checkout, source edit, calibration release, shared push or fit was made here.

## Root conformance and source-preparation authorization (2026-11-08)

Source preparation is approved; actual materialization/model/fit execution
retains a separate source/input/resource gate. Notebook063 is allocated
(next064). This is a private diagnostic; no shared release is replaced.

- **C53-A1:**10s/18channels is a scoped resolution test at three bands,
  not an alias-free full spectrum. Derive all subbin edges/counts/exposure
  from actual parent/native coordinates;318×6 and34,344 are planning
  estimates, not a hard upper bound or a measured product size.
- **C53-A2:** common MOD24 symmetry, native-MWSS quadrature, lmax128 and
  the declared cubic frequency operator isolate old/new source changes.
  Validate interpolation/normalization and preserve raw controls.
- **C53-A3:** inherited parent held-out groups keep subbins/donors together.
  Frozen training-only raw score prevents candidate variance inflation
  from buying an improvement; shared calibration error remains conditional.
- **C53-A4:** existing061-specific legacy-reader attestation is not blanket
  permission for a new063 reader. Recompute pinned native coefficients or
  supply an explicit063-specific reader audit with raw/coarse closure
  before reusing the original calibrated fields. CRC alone is insufficient.

Reproduce034's exact old calibration/design/sigma operator first. Separate
centre-model→native-weighted averaging changes from beam/time-resolution
changes. Existing-source/model choices stay fixed; no fitted gain or new
loss/noise/pointing term is added silently. MOD128 records these choices.
