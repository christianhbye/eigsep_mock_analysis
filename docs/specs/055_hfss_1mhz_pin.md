# Explicit dense-HFSS authority and opt-in v003 power candidate

**Status: diagnostic source preparation; generation, review and adoption pending.**
CHB asked that the established 1 MHz MWSS beam be pinned in both analysis and
mock_analysis. Analysis PR 99 supplies the common complex-field authority;
this change copies its exact manifest and prepares a separately versioned
eigsim scalar-power product. It leaves every old file/config/default intact.

## Inputs and schemas

`eigsim/src/eigsim/pins/hfss_1mhz_mwss_v0001.json` is byte-identical to analysis 055's
`docs/specs/055_beam_artifact.json`. Native artifact: 299,938,679 bytes, SHA256
`9b40552469364304305f0a65858663f52d60621148088c9f5cc7752b458672de`.
It pins raw export SHA `827c11c0...442389`, raw provenance and source-manifest
hashes. Version `v0001-diagnostic`, complex Cartesian rE in mV, shape
(201, 3, 181, 360), MWSS L=180, integer 50–250 MHz, theta 0–180/phi 0–359 deg.
Frame is the original global/body Cartesian frame, port 3:1, 1 W/0 deg.
The native S33 reference is 75 ohm; both 50/75 exports are retained. Physical
port location remains Q-BK-06 and absolute normalization is not independently
validated. No balun/coax transfer is inferred.

`eigsim.hfss_pin.load_native_fields(path)` requires that exact artifact/hash,
axes, complex dtype/shape and source/frame/version metadata. It adds no field
rescaling, symmetry or SHT and has no old-file fallback. Native arrays are not
the scalar-power schema that `data.load_beam` consumes.

## Derived recipe and assumptions

The new `make_bowtie_v003.py` reuses `make_bowtie_v002.native_to_mwss` verbatim;
the old generator is unchanged. Recipe version:
`raw-sumE2-directivity-mwss-lmax128-v1`.

- A1: follow existing v002 raw sum |Ex|²+|Ey|²+|Ez|², with no complex body-x
  symmetrization. Normalize each native-frequency power pattern by its native
  MWSS quadrature integral to directivity (integral 4pi).
- A2: use the existing exact MWSS spin-0 real forward transform at L=180,
  truncate coefficients to lmax=128, and invert onto MWSS L=129. No HEALPix,
  frequency interpolation, clipping or post-truncation renormalization.
- A3: retain the existing original-excitation efficiency calculation as a
  **diagnostic** array. It is neither a measured efficiency nor an independently
  validated absolute calibration. Store its interpretation explicitly.
- A4: retain the separate analysis operator: 055 symmetrizes complex Cartesian
  fields before power normalization. The same parent artifact does not make
  this raw-power eigsim variant the same effective beam. A same-beam comparison
  needs explicitly matched symmetry/normalization/truncation variants.

The candidate NPZ contains `freqs` in Hz, scalar `bm` with shape (201, 130, 258),
`lmax=128`, diagnostic efficiency/dropped-power arrays, recipe metadata and
generator/runtime/source provenance. Its separate candidate JSON records
the actual output SHA/bytes, parent SHA, reused transform-file SHA, generator
commit, dependency versions, lock hash, normalization/drop/minimum checks,
per-frequency timings and peak RSS. Fresh output paths prevent overwrite.

## Opt-in loading and pending release

`load_config("eigsep_v003_1mhz")` names the separate native 50–250 MHz variant.
Its `beam.pin_manifest` makes `data.load_beam(config=...)` verify the derived
artifact checksum/schema/recipe metadata and exact config-frequency agreement.
Explicit-file users should call `hfss_pin.load_pinned_power(path, manifest)`;
legacy `load_beam(path=...)` retains its generic, unpinned semantics.

The packaged power pin intentionally has null SHA/bytes until actual generation
and review. The new opt-in config therefore fails with an explicit pending-pin
error; it never substitutes v002 or an unpinned file. A three-frequency probe
has a different output filename and cannot silently replace the 201-frequency
product. Review must pin the actual full derivative hash before using this config.
All v002 / spline `_1mhz` names and defaults remain unchanged.

## Validation and resource gate

Focused checks exercise changed-byte rejection, rehashed wrong axes/dtype/frame/
source/finite support, derived schema/parent/config frequencies, pending-pin
failure, and the unchanged transform on a known constant harmonic at L=4→lmax=2.
No actual L=180 transform or 201-frequency generation is part of source preparation.
The local focused command passed 22 tests with 8 skipped old-data tests in 4.03 s;
process wall 5.32 s, peak RSS 395,196 KiB. Lint passes. These are not full-suite
or real-derivative validation claims.

Proposed next gate: a clean committed source and separate private frozen mock
runtime, with compatible no-AVX JAX/JAXLIB recorded; verify the exact native
artifact pin, then root's fresh aggregate CPU/memory/I/O check. Suggested probe
ceiling: 1 CPU, nice 0, 4–6 GiB, 1200 s, only 50/150/250 MHz. Root chooses/approves
the actual limits before launch. Capture first-compilation and subsequent
frequency costs, peak RSS and all scalar QC before proposing the full 201 run.

```sh
uv run --no-sync python eigsim/scripts/make_bowtie_v003.py \
  --native-source /explicit/private/bowtie_1MHz_mwss_v0001.npz \
  --output-dir /explicit/private/fresh-probe --probe
```

After a measured full-run gate, omit `--probe` with a separate fresh directory.
Compare the actual derivative against the same recipe/source and inspect
patterns, normalization, harmonic loss and effective-beam differences. Record
the full derivative SHA in the reviewed packaged pin, stage the binary as
gitignored model data, and run the configured eigsim suite. Keep the PR draft
until those checks and the real manifest are verified. Defaults and downstream
analysis/fit adoption require separate explicit review under the existing gates.
