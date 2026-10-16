# S4 spec-revision injection pilot

`s4_revision_pilot.py` produces a bounded, conditional preview for the D5 S4
night-trend spec. It is not the formal T6 battery and cannot identify a physical
cause. The spec and notebook029 live in `christianhbye/eigsep_analysis_code`,
`docs/specs/2026-10-15-s4-night-trend-hunt.md`; notebook029 is not executed by
this script.

The pilot was first executed from historical code commit
`ae3927783607535e84fa692bdfc936ee993b8e5a`,
`docs/specs/s4_revision_pilot.py`, SHA256
`ae7305de80c5611d931d8947181178de254a708faee231f6aaa2e5e062d0fcd6`.
It is moved here because synthetic-data producers belong in mock_analysis.
Formatting and the misleading output-field label change; the relocation
manifest checks the complete Python AST after normalizing only independent
top-level import order and that label. Historical frozen run sources remain immutable.

Use Python3.12 with NumPy/SciPy and a compatible no-AVX JAX runtime on Limbo.
Pass explicit paths; do not mutate another run's environment or checkout:

```bash
python scripts/s4_revision_pilot.py \
  --s1 /private/frozen/canonical023 \
  --team /private/frozen/marjum-2026-07 \
  --skyground /private/frozen/eigsep_analysis/skyground.py \
  --out /private/new-run/pilot --n 8 --cal-n 8 --budget-seconds 1200
```

Supply the complete exact `src/eigsep_analysis/*.py` tree containing the pinned
S1 solver as the parent of `--skyground`. The loader uses a private namespace;
it does not substitute a different installed solver or execute the package
initializer. Input and source hashes are recorded. Required S1 files are
`columns/columns_eigsim.npz`, `fits.npz`, `noise.npz`; required team files are
`derived/ground_sky/binned_v0001/binned.npz`,
`derived/ground_sky/fit_v0009/fit.npz`, and
`derived/tcal/v0003/solutions.npz`.

First run `--probe` in a detached bounded unit: canonical coefficients and the
profiled solver must agree with independent full joint least squares. The
pilot then measures one battery and freezes balanced counts4–8 per truth with
25% projected timing headroom. If4+4 cannot fit, it returns a gate-only report.
A budget stop saves incomplete draws and supplies no power table. The launcher
must enforce aggregate CPU, nice0, wall-time and memory limits; this script does
not launch jobs or enforce host sharing. Treat output as complete only with
exact-invocation terminal success, exit0 and a complete summary.

Calibration/evaluation signs are separate conditional draws of one fixed
residual field, not new observational noise realizations. Wilson intervals
quantify Monte Carlo variation only. The joint statistic is a partial-rule preview: it uses only the stricter sign
arm of criterion1 while omitting equal-freedom competition, driver/physical
gates, high-band selection and H0 variants. These gate sets are not nested, so
the preview rate has no guaranteed upper/lower ordering relative to formal power.
Historical executed JSON names this field `joint_class_upper_bound`; that name
is misleading and supplies no mathematical bound. Future output uses
`joint_partial_preview` for the identical Boolean calculation. The small pilot cannot certify a5% tail. Full formal
T6 must use the complete approved verdict function before any scientific claim.
