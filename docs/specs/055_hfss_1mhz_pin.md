# Explicit dense-HFSS authority and opt-in v003 power candidate

**Status: full 201-frequency diagnostic candidate generated and pinned; full-suite and
scientific adoption review remain pending.**
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

The packaged power pin now records the generated full201 artifact:
SHA256 `1ab98e6707f686336d49302fc260357521f11dee97826aeaf559df4ef055f68b`,
54,002,762 bytes. The binary is gitignored model data, staged only in this
private opt-in checkout; the exact same producer pin is in the analysis
contract. Missing/changed files or pending/null pins fail explicitly and never
substitute v002. The three-frequency probe has a different filename and cannot
replace this 201-frequency product. This is a diagnostic pin, not a default
or scientific adoption.
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

The root-approved full201 run used exact source e6e367c on Limbo, unit
`limbo-c055-mock-full-20261108T005303Z`, 1 CPU / nice0 / 4 GiB / 1200 s,
256 MiB product cap, frozen JAX/JAXLIB0.9.2, explicit x64 and one math thread.
It finished in 192.19 s outer wall / 183.90 s generator wall, peak RSS
1,234,608 KiB. Independent checks load the exact pin, verify all201 native
frequencies/finite float64 arrays, recompute every normalization integral,
and compare the50/150/250 slices bit-exactly with the earlier separate probe.
Max normalization error is3.494e-12, dropped harmonic fraction1.763e-12;
minimum power4.608e-5. The saved drop array matches the producer report;
this is not an independent HFSS convergence test.

`055_effective_operator_contract.json` records the actual generation/runtime
identity and intentional operator variants. The common raw authority is exact;
the existing055 complex-symmetrized/cell-area-normalized diagnostic differs
from raw-power/native-MWSS v003 before truncation by relative RMS2.21e-5,
9.92e-5 and3.88e-4 at50/150/250 MHz. Neither is silently substituted for the
other. Matching a consumer comparison requires an explicit matched variant.
Run-specific QC stays outside the generator's producer-pin template.

The full configured eigsim/horizon test suite must run at the final source
head and its outcome be recorded in the PR; the PR stays draft until that
gate is handled. Hosted CI is not used while runner minutes are exhausted.
Defaults and downstream analysis/fit adoption require separate explicit
review under the existing gates.
