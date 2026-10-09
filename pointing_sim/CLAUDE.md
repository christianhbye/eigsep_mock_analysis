# Pointing-only simulations

Follow the root CLAUDE.md. Use uv, ruff and focused pytest under
`pointing_sim/tests`. This member is plain NumPy: no receiver, sky,
eigsep_cal, eigsim or eigsep_analysis imports, and no file I/O in src.
The caller owns products, provenance and data selection.

Keep continuous drive moments, microbin means and instantaneous samples
distinct in APIs/tests. Sparse telemetry-update averages do not acquire
continuous-integration semantics from a passing idealized injection.
