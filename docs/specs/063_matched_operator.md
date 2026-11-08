# C53/063 opt-in model operators

Diagnostic source only. Workspace registration `b8e699e`, MOD128 and reviewed
`docs/specs/2026-11-08-c53-fine-night-pilot.md` govern execution. No source
generation, SHT, materialization or fit is authorized by this file.

The standalone `fine_night_d5` component isolates lightweight NumPy/SciPy
operators and tests from eigsim's existing heavy import chain. It changes no
eigsim default or existing beam artifact. Its own configured workflow covers
its entire test directory. Real SHT calls import the existing pinned s2fft/JAX
runtime lazily; the builder requires x64 and does not change JIT or flags.

Primary old/new fields both apply055's complex Cartesian body-x projection
`0.5*(E+diag(1,-1,-1)*E(phi_mirror))` with
`phi_mirror=(nphi/2-index)%nphi`, before scalar power. Normalize with native
MWSS quadrature to integral4pi, then exact forward scalar MWSS/crop lmax128/
inverse MWSS as in make_bowtie_v002.py. Do not use055's cell-area normalization.
Only explicit `body-x` or `raw` controls are accepted. Unknown physical port
plane and loss/S11-transfer uncertainty survive; no realized efficiency is
inserted into directivity. Each frequency checkpoint and QC is saved before
assembly so a timeout cannot erase every completed slice's evidence.

The old complex source SHA is
`8c63a213cbdd77c1a7eed51b592359ee63bf4495228de07fe397510369992924`.
The new complex source SHA is
`9b40552469364304305f0a65858663f52d60621148088c9f5cc7752b458672de`.
Derived symmetric artifact hashes are **pending** until an approved real run.
Raw v002/v003 stay labelled controls; sharing a raw complex parent does not
make two effective operators identical.

Use direct not-a-knot CubicSpline from each normalized native-frequency scalar
grid to the18 fractional native frequencies. No chained1MHz interpolation,
extrapolation, clipping, renormalization after interpolation or score-driven
method selection. Report negative pixels, full-sphere integrals and withheld
slice errors. The source provides these numerical QC arrays; actual acceptable
sky-column errors and their existing uncertainty comparison still require the
run review. Toy cubic recovery is not real beam interpolation validation.

`average_native` evaluates explicit actual native times/rotations, then uses
the exact per-channel data mask/counts. It returns model means and counts;
it never invents exposure weights or independent observations. The existing
sky/horizon/loss inputs are supplied explicitly to the concrete EigsimColumns
adapter, which calls existing simulate_path(Tgnd=0) and compute_fgnd exactly
as034, at actual native times. The paired code bridge uses the shared v5
provider,034 mount rounding/offsets and035 night tilt. Actual model assets,
native pointing coverage and real adapter validation remain run gates.
`join_pointing` rejects missing/duplicate rows, time mismatches and improper
rotations. No nearest parent or inferred hardware state is inserted.

Run the real builder only in an approved bounded detached unit, for example
`python -m fine_night_d5.build --native PATH --sha256 SHA --symmetry body-x
--out NEW_STAGE`. The driver must bind source commits, full runtime/transform
pins and resource/product caps and write killed/timeout disposition. The
builder's in-process FAILED sentinel does not catch a kernel kill. No source
push, release, phase2 compact-source correction or night fit is implied.
