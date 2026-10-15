# D5 derivative implementation plan

1. Reference and tests: `horizon_position_d5/reference.py`, `horizon_position_d5/tests/test_reference.py`.
   Implement the exact cell-boundary/stationary-point reference and a sampled
   control with a fixed inner cutoff. Test synthetic plane, oblique crease,
   interior extrema, near ties, cardinal rays, finite raster and E/N/U response.
   Keep study functions NumPy-only, independently comparable to dense sampling.
2. Reusable interpolation: `eigsim/src/eigsim/position.py`,
   `eigsim/tests/test_position.py`, export through `eigsim/__init__.py` and
   document in `eigsim/README.md`. Tests first for node/affine reproduction,
   independent gradients, C1 joins and domain guards. No terrain-file I/O here.
3. Study driver: `horizon_position_d5/run.py`, `inputs.py`, `temperature.py`,
   `cache.py`, README and focused tests. Validate inputs/hashes; use the pinned
   S1 sky/pointing conventions. Verify W-affine acceleration against full
   simulate_path. Record reference/candidate baselines separately so the
   product-march approximation is not conflated with derivative error.
4. Run small synthetic tests locally. Commit/push the clean source before
   the first private Limbo pilot. Verify JAX import/JVP, nice2 and affinity.
   A detached job writes a timing gate and first result before being reported
   as running. Source/lock/input fingerprints key every cached block.
5. Use the measured gate to run displacement, resolution and held-out grid
   comparisons on Limbo. Preserve failed/unresolved cases. Run the conditional
   synthetic inference diagnostic only with identifiable response columns.
6. `code` notebook026 and a strict result reader in `src/eigsep_analysis/`:
   configure paths through paths.py; document the cross-repo spec commit;
   reject inconsistent input provenance; execute with the worktree's kernel;
   plot every stage and retain outputs. Register the diagnostic notebook.
7. Focused tests and formatting for both changes; full mock suite and configured
   CI as appropriate. Fresh conformance reviewer checks equations, tests,
   failure cases and notebook/claim provenance; fix and re-check once.
8. Open two private PRs for CHB; include spec/plan commits, known gaps and exact
   run scope. Return tracker updates in STUDY_STATUS.md for Claude. Keep shared
   contribution edits for tomorrow's approved packet/session.

No new tracker IDs allocated. Code notebook026 and both worktrees were registered
before creation in workspace commit4a5fb0c at CHB's explicit request.
