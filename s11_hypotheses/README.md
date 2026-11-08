# C54 S11 generator: separate runtime and diagnostic scope

This is the simulation/upstream-injection side of the three completed S11
tests. The observed-data fits and executed060 reader live in the private
analysis repository: [complete code PR102](https://github.com/christianhbye/eigsep_analysis_code/pull/102),
executed output64f601c, and [M009 PR11](https://github.com/christianhbye/eigsep_analysis_manuscript/pull/11).
Results remain diagnostic: no unconditional D5 narrow-feature absence,
numerically converged terrain amplitude, or general VNA-noise floor is
identified. The actual source/positive-control captures remain disjoint.

The original physical/synthetic producer is mockb9d2a89 with analysis2ef1984.
Its source, environment, models and checkpoint keys remain unchanged.
The original instructions are preserved in `docs/B9_PRODUCER_HISTORY.md`.
This delivery uses an installed src layout and a new source schema; it
never renames/rekeys an old scientific product as newly generated science.
All feature/leakage/terrain function ASTs and frozen choices are checked
against a source-only original-b9 manifest. No physical kernel is duplicated.

## Actual dependency and install contract

This standalone project requires NumPy, h5py, pandas, SciPy and the private
`eigsep_analysis` companion at exact64f601cdcc67e12791ff71078813fd3d737a37d8
(distribution version0.0.1). Its immutable source policy is shipped in
`src/s11_hypotheses/runtime_contract.json`.

The companion's frozen runtime uses JAX/jaxlib0.10.2; the ordinary mock
workspace is locked to0.9.2. This package is explicitly excluded from that
workspace. Its supported bootstrap installs the companion's frozen lock
into a NEW environment owned by this worktree, then installs this src package
without resolving another dependency graph. The private companion contains
editable checkout sources: installing it as an arbitrary registry package
or pretending a bare `pip install` is portable would hide real prerequisites.

First prepare registered, owned snapshots of the clean companion checkout
and every local source dependency, with the lock's relative directory layout.
Both their source directories and writable Git metadata must be inside the
owned prefix; symlinks back to protected sources are refused. The analysis
lock must have SHA256
`74bebe6da6a112ca91fe0268937e0c9f8f3eb4d6998205671d44fd41aff13f0b`.
Then, from this mock worktree:

```sh
python3 s11_hypotheses/scripts/bootstrap_runtime.py \
  --analysis-root .runtime-s11/research/papers/eigsep_analysis/code \
  --owned-source-root .runtime-s11/research \
  --retained-healjax-from /OWN/PREPARED/ENV \
  --environment .venv-s11-new
```

The bootstrap is offline by default, refuses an existing environment, and
uses `UV_PROJECT_ENVIRONMENT` to protect the companion's own environment.
It records exact install commands. Ruff0.16.10 is an additional pinned
lint-only tool; scientific dependencies come from the companion lock.
Missing cached packages or editable checkouts are installation failures,
not permission to use different dependencies or fake core modules.

Companion source builds use only the validated owned snapshots. The
non-editable healjax distribution is retained from an explicitly supplied
prepared environment: actual installed files and original metadata/RECORD
are copied byte for byte and hash recorded, without inferring a Git revision.
The sync excludes healjax so it cannot silently rebuild a different snapshot.
Git source checks disable optional index refresh writes.
The actual initial bootstrap journal showed cached installation of184
companion packages; only this owned new adapter was built. A read-only
post-run audit found all protected tracked files clean and all protected
egg-info mtimes predating bootstrap; no pre-run full-filesystem snapshot
was taken, so that is the recorded evidence scope. A later read-only
`--check --no-build` attempt refused cached editable analysis, so no-build
installation is not offered as a working setup path. The registered owned
snapshots now remove the protected-source build ambiguity.

Runtime verification checks the actual import binding, immutable companion
commit, clean tracked files, exact lock and installed versions. The actual
executable-payload identities remain the producer's transitive fingerprint
(including its supported noAVX conda convention, without fabricated RECORDs).
The src files, implementation of fingerprinting, package metadata and runtime
policy are included in the new source key. Changing the policy changes that
key; old scientific checkpoints cannot silently resume under this layout.

## Tests and CI coverage

The configured standalone suite requires this installed src package and the
real companion. It includes original equation/operator/sharding tests plus
installed-layout/CLI, dirty-tracked-companion, source-policy mutation,
original-mathematics and actual-required-input tests. The known historical042
summary input is now explicitly listed by producer staging preflight.

After committing source and receiving the resource gate, the complete command is:

```sh
.venv-s11/bin/python -B s11_hypotheses/scripts/test_contract.py \
  --analysis-root .runtime-s11/research/papers/eigsep_analysis/code \
  --output /OWN/RUN/standalone
```

It runs every configured `s11_hypotheses/tests` case and records source,
runtime, inventory, argv, JUnit counts/hashes, wall/CPU/RSS and DONE/FAILED.
The existing configured mock suites must be run separately in their own
`.venv-workspace` from the unchanged mock `uv.lock` (SHA256
`8ca1138dbbe87b9d80e5a060bd19e81bac43073d30bc64a01bde6b1e275c7038`).
Their default testpaths cover eigsim, rotis, pointing_sim and horizon_position_d5;
they do not cover this standalone package. Neither suite substitutes for the other.
Full configured LOCAL runs remain pending the orchestrator's resource gate;
focused checks are reported separately, never as a full passing run.

The dedicated manual workflow fails closed unless an explicitly prepared
private runtime is available. It invokes the same standalone command and
does not fetch a companion/shared repository, assume credentials, upload
artifacts, or provision a runner. No hosted run has been dispatched. CI
private-runtime provisioning/access is a concrete optional owner follow-up;
LOCAL configured coverage is the present completion criterion.

Automatic approval review rejected the earlier proposed hosted cross-repository
fetch/artifact-upload patch because private/shared access, an unconfigured
credential and egress were not specifically authorized. That patch, clone
helper and upload action were not applied. The safer manual workflow adds
none of those actions. Missing CI access is not a passing test result.

## Scientific limits and future producers

The original600m model gives a corrected alias-power/adjacent-epoch-difference
proxy ratio0.0500791. The proxy is not VNA thermal noise. Coherent folded power
changes44.18% under the final1m→0.5m native facet refinement; expected diffuse
folded power changes8.14%. A broader DEM is required beyond the1343.634m
full-disc limit, and coherent within-cell quadrature/spatial-alias treatment
needs a separate test. More Monte Carlo members cannot remove that deterministic
nonconvergence. Raw|E|²/clamped old beam, conditional network phase/plane,
shared-donor/chain covariance and one-member fine-grid limits stay explicit.

Both donor-excluded calibrated antenna heldouts fail their narrow native-frequency
criterion (3/12 and1/12 blocks); the latest raw antenna pass is conditional.
The twelve predeclared two-MHz blocks contain96 native samples in14–120MHz,
not continuous or full1–250MHz coverage. Original records and the corrected
observation consumer are retained by060; no simulation/export is rerun for
this packaging task. Any future producer needs a fresh full-input/source
dry-run and explicit time/RAM/product gate. Historical configs are not
authorization to launch or reuse original keys.
