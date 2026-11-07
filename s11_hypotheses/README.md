# C54 S11 producer

This package runs the three diagnostic S11 tests; it does not turn raw
repeatability into absolute VNA noise. MOD124/125 were recorded by the root
orchestrator at workspace8c4626b. The reviewed execution design and input
manifest are in code060 `docs/specs/2026-11-07-c54-*`; dispositions are in
`docs/reviews/2026-11-07-c54-physics-dispositions.md`.

The implemented stages are raw/existing-plane feature prediction with
translated controls and source-only uncertainty; original shared-OSL chain
perturbation; explicit-range coherentPO/diffuse4b expected-power/covariance
and historicalR0; single-scale/tau0 positive-power training with signed/epoch
holdouts and an actual600m/AC5 power audit; and empirical causal skirt
prediction with measured masks, influence, upstream delayed-line recovery
and phase-limited conditional terrain comparison. No negative-delay scale
is fitted. R0 angular comparison uses omega_ns/omega64 with unchanged raw
historical export. Fine and native grids are separate and independently
evaluated. Diffuse refinement compares expected power/covariance, not
unrelated complex speckle. Wider range/angular configurations remain an
explicit closure ladder; a600m run cannot claim all-terrain closure.

Use a task-local environment with code060's frozen dependencies and this
adapter installed editable, without dependencies or global modifications:

```sh
# In the assigned code060 worktree:
uv sync --offline --frozen --extra dev
# Install the adapter into that interpreter (use an absolute adapter path).
uv pip install --offline --no-deps --python .venv/bin/python -e ../mock-c54-s11-hypothesis-tests/s11_hypotheses
```

The mock runtime was independently created from the same frozen code060 lock
using UV_PROJECT_ENVIRONMENT; no existing environment was edited. This
adapter does not require installing eigsim/rotis or altering their lock.
Local versions: Python3.12.12, NumPy2.3.5, SciPy1.18.1, h5py3.16.0,
hera_filters `ed4deb46ae0e16488a2f0633eaa792eb731e7fc9`. Ruff0.16.10 is
lint-only. On Limbo use the approved no-AVX dependency/runtime procedure;
record its differences from this local setup before the probe.

From the mock worktree, first inspect the entire dry-run JSON:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 .venv/bin/python -m s11_hypotheses.run probe \
 --config s11_hypotheses/configs/local-probe.json --dry-run
```

It prints every required original path, relocated path, size/hash, missing
input, physical unit, full grid count and phase-product cost per visible
cell. Local example paths are intentionally UNLAUNCHED. Root creates a new
private-Limbo config with explicit path_map prefixes, never a silent input
fallback. Stage the exact listed primary files:40D5 raw captures,28MISTIC raw
captures,013 calibrated planes/donor table, C43 native-standard snapshot,
DEMv0004/horizonv0005/height/pointing/mount files, existing sixR0 trace units,
original HFSS beam, paperS11 magnitude and switch paths. Preserve the frozen
manifest bytes; `path_map` changes locations only. MISTIC raw records and
all original data/workspaces remain read-only. No new1MHz beam is required;
an amended beam needs its own checked intake and lock before use.

Root's resource gate is a JSON file with `allow`, `source_fingerprint`,
`manifest_sha256` and `configuration_fingerprint` exactly matching the
command's dry-run result. Probe scope is one night/member, slant0–600m,
stride4/bin1deg, full native50–250MHz and fine10kHz, plus raw-ant/control,
receiver, shared OSL and one source-only MISTIC raw short. Provisional
reservation2CPU/4GB/1200s/1GB is not measured feasibility. Run detached with
nice0 after the fresh aggregateCPU/RAM/I/O check. Save each completed cells,
native, fine and raw-control subunit; no agent waits alive during the job.
A timeout retains completed units; an incomplete unit refuses silent reuse.

After the measured gate permits subsequent work:

```sh
.venv/bin/python -m s11_hypotheses.run controls --config RUN.json
.venv/bin/python -m s11_hypotheses.run export --config RUN.json --resume
.venv/bin/python -m s11_hypotheses.run score --config RUN.json --resume
.venv/bin/python -m s11_hypotheses.run report --config RUN.json
```

`controls` reproduces historical042 separately, uses first10 canyon captures
for source estimation, last10 for the new positive-control score, tunes
blinded control intervals, then writes `design.lock.json`. D5 target values
never select baseline support/penalty. The lock is required before final
score/export. Injections are upstream of fitting; the fixed linear
operator's homogeneous scale/phase family is tested against explicit
refits and recorded, including zero-support UNOBSERVED cases and source
leave-one-out uncertainty. A sensitivity exclusion still requires real
withheld baseline adequacy and recoverability. No synthetic pass alone
validates a DUT model.

All units record source-only transitive fingerprints, dependency identities,
parameters, exact coordinates/masks/settings, upstream/result hashes,
RNG/physical draws and wall/CPU/RSS/I/O measures. Resume skips only verified
exact matches. `RUN_RECORD_<command>.json` retains host/interpreter/commits,
invocation, run config and gate. No notebook-output commit enters source
cache keys. Existing041 units/runs are consumed read-only and never modified.

Code060 provides the observed fitting/report modules, wrapper
`scripts/060_s11_hypothesis_tests.py`, and completion-gated notebook builder
`scripts/060_build_notebook.py`. Do not build/commit060 as completed before
`units/science/DONE.json`; execute in060's own kernel, retain outputs, inspect
all printed-size figures, then rewrite/build/render/audit M009. Science
results and a readable memo remain the task deliverables after this
run-ready machinery phase.
