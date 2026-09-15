# Horizon and Levelling Sensitivity for the D5 Forward Model — Design

**Date:** 2026-09-14
**Status:** Draft (awaiting review); phase 1 executed, see the note below

## Superseded during execution (2026-09-14)

Three things in the body below did not survive phase 1. The body is left as
written — it is the record of what was designed — but plan **phase 2 from the
corrected story**, not from these:

1. **"Closed-form calculus on one point per azimuth"** (§ *1. `mock_analysis`
   — the horizon generator*) is not the whole derivative. A horizontal move
   also shifts *at which azimuth* a fixed terrain point is seen, adding a
   second, non-local term `alpha_h'(az) * d(az_p)/d(e0, n0)` — azimuthal
   parallax. `dalpha_dE`/`dalpha_dN` in `horizons_position.npz` are therefore
   **totals**: the pixel partial (kept as `dalpha_d{E,N}_pixel`) minus the
   parallax term (`daz_d{E,N}`). Dropping it is off by 57–99 % on the
   solid-angle-weighted open-sky fraction, against < 3 % for the total.
   `dalpha_dU` is unaffected (`d az / d u0 = 0`). Corrected story:
   `horizon_position/make_horizons.py`'s module docstring.

2. **The pointwise `alpha_h + J·delta` validation** (§ *Validation*) is not
   meaningful as prescribed. `alpha_h(az)` is piecewise constant — one
   `hor_ang` per winning DEM pixel, copied across the bins its footprint
   covers — so a horizontal move translates the curve's step edges, which a
   pointwise comparison reads as an O(1) discontinuity however small the step
   is. The pixel partials are validated pointwise only on the population where
   the same pixel wins at both positions; the *total* tangent is validated in
   W-space instead, with `jax.jvp` through `eigsim.open_sky_weight`. Corrected
   story: `horizon_position/test_jacobian.py`'s module docstring.

3. **`margin` was dropped** (referenced at the generator section, the data-flow
   diagram and the memo outline). Computing it exactly needs the recursive
   pruning inside `DEM.calc_horizon`, which this spec forbids touching; the
   measured switch fraction from comparing `crds` across the 19 positions
   serves the same purpose in M004. Corrected story: the plan's *Deviations
   from the spec* § 1
   (`docs/superpowers/plans/2026-09-14-horizon-tilt-sensitivity-phase1.md`).

## Goal

Fold the horizon-position study into the Deployment 5 forward model, so that
the model carries **derivatives** of the antenna temperature with respect to
antenna position and instrument levelling, not just a single prediction at the
nominal geometry.

This requires promoting the fractional (anti-aliased) open-sky mask from
`horizon_position/masks.py` into `eigsim`, where the D5 forward model can reach
it, and adding the misalignment rotation that the EIGSEP drive cannot produce.
The deliverable is memo **M004** in the `eigsep_analysis` repository.

This design extends `2026-06-13-horizon-position-sensitivity-design.md`, whose
scope guard was a zenith-pointing antenna with no drive rotation. Misalignment
is only identifiable across drive orientations, so that guard is lifted here.

## Why the boolean mask cannot serve

`eigsim` currently reduces its packaged horizon to a hard boolean mask —
`run_canonical_sim.py:49` does `horizon = np.isnan(horizon)`. Measured on the
nominal horizon against the MWSS grid at `lmax = 128` (`ntheta = 130`,
`nphi = 258`, `dtheta = 1.395 deg`):

| mask | open-sky solid-angle fraction | vs theta-fractional | per-cell RMS |
|---|---|---|---|
| boolean (`isnan`) | 0.3471014 | +82 ppm | 0.500 |
| theta-fractional (the paper's) | 0.3470193 | — | — |
| phi-fractional (issue #10) | 0.3469965 | −23 ppm | 0.168 |

The 82 ppm static bias is not the reason to change. The reason is that
`dW/d alpha_h` is `1/dtheta` in the boundary cell for the fractional weight and
**zero almost everywhere for a boolean mask**: the chain rule terminates at
zero, so a boolean mask cannot furnish a derivative at all.

That matters because the perturbations of interest are all sub-cell. Measured
from `horizons_position.npz` against the 1.395 deg cell, a position shift moves
the horizon by roughly 1 per cent of a cell at 0.1 m and 5–10 per cent at 1 m;
only the 10 m cases reach cell scale. Up — the binding axis in the instrument
paper — moves the horizon by 0.139 deg RMS at 1 m, with a maximum of 0.34 deg
anywhere on the sky.

## Parameterization

Three groups, entering the forward model in three places.

**Antenna position `(e, n, u)` → the horizon.** Changes `alpha_h(az)` through
the DEM. The beam does not move.

**Outer misalignment → the beam orientation.** A small rotation composed
*outside* the commanded drive, `R = R_mis @ Rx(el) @ Rz(az)`: the instrument as
a whole is not level, so the tilt is static in the topocentric frame and does
not rotate with the drive. The horizon does not move. Inner (antenna-to-mount)
misalignment is assumed not to be an issue and is out of scope.

**Everything else** — beam model, balun, coax, sky — is out of scope, on
`eigsep_cal`'s side of `eigsep_cal/docs/interface.md`.

### Which misalignment components are free

Rotations about a shared axis commute and add, which fixes the identifiability:

- `Rx(eps) @ Rx(el) = Rx(eps + el)` — an outer misalignment about X is
  *exactly* an elevation encoder offset, for every commanded orientation. It is
  therefore **not** a free parameter here: it belongs in an elevation encoder
  offset, wherever the instrument model keeps one.
- `Rz(az) @ Rz(eps) = Rz(az + eps)` — an *inner* misalignment about Z is
  exactly an azimuth encoder offset. Out of scope by the assumption above, and
  degenerate in any case.
- An outer misalignment about Z rotates the tilted boresight about zenith and
  does **not** commute through `Rx(el)`, so it is identifiable. Physically it is
  the error in the azimuth reference against true North.
- A misalignment about **Y** (roughly North–South) is degenerate with nothing,
  in either position in the chain, because the drive contains no Y rotation at
  all. It is the levelling component the data can genuinely constrain.

So the free misalignment parameters are **`eps_y`** (levelling tilt) and
**`eps_z`** (North-reference error). `eigsim/src/eigsim/rotations.py` defines
`rotation_matrix_x` and `rotation_matrix_z` and no `rotation_matrix_y`;
adding one is part of this work.

## Architecture

`eigsim`'s simulation path is already JAX end to end — `_orient` and `_run` are
jitted and the mask enters as `pixel * horizon[None]`. If the mask is *built*
in JAX from `alpha_h` rather than passed in as a frozen array, then everything
from `alpha_h` to `t_ant_k` is one differentiable graph and the only derivative
that must be derived by hand is `d alpha_h/d(e, n, u)`.

Three components, in two repositories. `eigsep_terrain` is not modified and
remains a read-only upstream; the 192 MB DEM never comes near `eigsim`.

### 1. `mock_analysis` — the horizon generator

Extends `horizon_position/make_horizons.py`, which already runs in the
`eigsep_terrain` environment. `DEM.calc_horizon` returns `(hangles, crds)`,
where `crds` identifies the DEM pixel that sets the horizon in each azimuth
bin; line 77 currently discards it (`hangles, _ = ...`). Keeping it makes the
Jacobian closed-form calculus on one point per azimuth, from
`alpha_h = arctan2(U - u0, r_min)`, using the DEM the script has already
loaded.

New output. `alpha_h` and `crds` are emitted for all 19 positions, as
`alpha_h` already is today — the switch diagnostic needs `crds` at the
perturbed positions, not only at nominal. The Jacobian and `margin` are
nominal-only:

```
az_grid, alpha_h (19, n_az), crds (19, 2, n_az),
dalpha_dE, dalpha_dN, dalpha_dU (n_az,), margin (n_az,), provenance
```

`margin` is, per azimuth, the gap in horizon angle between the winning DEM
pixel and the highest pixel that is not it — the local distance to a switch.

### 2. `eigsim` — the fractional mask, in JAX

A new `eigsim.horizon` module exposing `open_sky_weight(alpha_h, az_grid,
lmax)`, returning `W` in `[0, 1]`, **phi-integrated from the start**: map
`phi = pi/2 - az`, interpolate the curve to `n_phi x sub` fine azimuths, clip
the theta-cell fraction, and average into phi cells. Built in `jnp`, so
`dW/d alpha_h` is autodiff rather than hand-derived.

Because the cell integral *is* the band-limiting, `reduce_azimuth` has no
counterpart on this side and must not be reproduced. `sub` defaults to 180,
fixed by the convergence test below rather than chosen; at that value the
integration samples ~46440 fine azimuths against the native curve's 46080, so
it evaluates the curve at essentially its own resolution. Note `n_phi = 258`
does not divide `N_AZ = 46080`, so this is sub-sampling or weighted quadrature,
never a reshape.

### 3. `eigsim` — `rotation_matrix_y` and misalignment

`drive_rotation_matrix(elevation_deg, azimuth_deg, misalignment=None)`
composing `R_mis @ Rx(el) @ Rz(az)`, in `jnp` so that `eps_y` and `eps_z`
derivatives come from the same autodiff. `misalignment=None` reproduces today's
behaviour exactly.

### Data flow

```
DEM ──generator──> horizon_nominal.npz {alpha_h, dalpha_d(e,n,u), crds, margin}
                          │
                          │  tangent dalpha_h/d(e,n,u)   (hand-derived, 1 column per axis)
                          ▼
   alpha_h ──> W ──> beam*W ──> convolve ──> t_ant_k      (all JAX, one graph)
                          ▲
   (eps_y, eps_z) ──> R_mis ──> drive rotation ───────────┘
```

`dT/d(position)` is three `jax.jvp` calls pushing the supplied tangents through
the existing simulation — no new simulation runs and no finite differences.
`dT/d(eps_y, eps_z)` is plain autodiff with no tangent needed.

## Compatibility

Unchanged: the signatures of `simulate()` and `simulate_path()`, the packaged
`horizon_mwss.npz`, the boolean horizon path, and everything `horizon_position`
does today. The fractional mask arrives as a new, opt-in input. The RASTI
paper's inputs stay reproducible from tag `rasti-round2-figs` (`bbde8af`).

## Failure modes

**Argmax switching is the one real failure mode, and it is measured rather than
assumed away.** The analytic derivative is exact only while the same DEM pixel
continues to set the horizon at a given azimuth. `make_horizons.py` records 108
of 720 bins flipping for a 0.1 m East shift at coarse binning, converging away
by `N_AZ = 46080`. Since `crds` is retained anyway, the diagnostic is free:
compare `crds` at nominal against `crds` at each perturbed position and report
the fraction of azimuths that switched, per axis and per step size. Storing
`margin`, the runner-up horizon angle, gives the local predictor of where the
derivative is fragile before any perturbation is run. The memo quotes both.

**No checkpoint/resume.** `run_sims.py` today checkpoints one
`pos<tag>_batch_NN.npz` per position and skips batches already on disk, guarded
by `pos_sha` (`run_sims.py:106-116`). That guard hashes the positions only, so
it accepts stale batches after a DEM, `N_AZ` or mask change — exactly phase 1's
case, where a pass would merge old-mask and new-mask positions into one
`position_sims.npz` that is then finite-differenced to validate the Jacobian.

Decided: **delete checkpoint/resume rather than extend the guard.** The full run
is ~20 minutes, cheap enough that resumability does not earn its complexity, and
removing the mechanism removes the failure mode instead of patching it. An
interrupted run restarts from scratch; results accumulate in memory and
`position_sims.npz` is written once.

`pos_sha` stays in the *output* npz as a content identifier — the paper's
`horizon_shift.npz` and `horizon_perturbations.npz` cite it to confirm they
share the 19-position configuration, and M004's `PROVENANCE.json` will do the
same. Only the resume path goes.

**Azimuth convention.** `alpha_h` is defined on `az = atan2(E, N)` and the mask
maps `phi = pi/2 - az`. `open_sky_weight` validates the grid's range and
monotonicity rather than silently accepting a curve on the wrong convention.

## Validation

**The validation set already exists.** `horizons_position.npz` holds `alpha_h`
at nominal ±0.1/1/10 m along each of E, N and Up. Predict
`alpha_h(nominal) + dalpha/dpos · delta` and compare against the stored curve:
agreement should be excellent at 0.1 m, degrade at 1 m, and visibly break at
10 m. A test that does not break at 10 m is testing the wrong thing.

**End-to-end check needs one re-run.** `position_sims.npz` was produced with
the theta-only mask on pre-frame-fix `eigsim`, so differencing it against the
new chain would confound the mask change, the croissant frame fix (`754627c`)
and the croissant bump (`1384c8b`). Re-run the 19 positions on current `main`
with the phi-integrated mask (~20 min) and finite-difference that instead. The
re-run is consequence-free now that the paper is pinned, and it is exactly the
measurement issue #10 deferred — so **issue #10 is closed as a by-product of
this work**, not as a separate task.

## Tests

Ported from `horizon_position/test_masks.py` into `eigsim`:

- flat horizon is half open; all-blocked and all-open limits
- monotonic in theta
- a sub-pixel shift registers (no floor to zero)
- the frame mapping `phi = pi/2 - az` blocks East, not North

New:

- convergence in `sub`
- a spike one native azimuth bin wide does not alias **without** any
  `reduce_azimuth` step — issue #10's property, stated as a test
- `jax.test_util.check_grads` on the mask
- a JVP-versus-finite-difference check on `alpha_h`, to catch chain-rule sign
  errors
- `drive_rotation_matrix(misalignment=None)` reproduces the current matrix
  bit-for-bit

`eigsim/tests/test_regression.py` continues to guard the boolean path.

## Deliverable

Memo **M004**, *Horizon and levelling sensitivity of the D5 forward model*, in
`~/Documents/research/papers/eigsep_analysis`:

- `dT_ant/d(position)` in K/m and `dT_ant/d(eps_y, eps_z)` in K/deg, across
  frequency and LST
- the validity range of the linearization, quantified by the argmax-switch
  fraction and the margin distribution
- the resulting requirement: how well position and level must be known before
  the horizon dominates the D5 error budget

Per that repository's `CLAUDE.md`: notebook `007` (registered in
`code/notebooks/README.md`), derived data in
`products/007_horizon_tilt_sensitivity/` with a `PROVENANCE.json` written by
`provenance.write`, the memo in `manuscript/memos/M004_horizon_tilt_sensitivity/`
in AASTeX ending with a provenance table, and entries added to
`docs/logbook.md`, `docs/roadmap.md` and `manuscript/memos/README.md`. Any
question for Christian gets an ID in `docs/open_questions.md`.

## Implementation phases

This is two implementation plans, not one. They are sequential and land in
different repositories under different conventions:

**Phase 1 — the code**, in `mock_analysis`: the generator Jacobian, the
`eigsim.horizon` mask, `rotation_matrix_y` and misalignment, the tests, and the
19-position re-run that validates the chain. Closes issue #10.

**Phase 2 — the memo**, in `eigsep_analysis`: notebook `007`, product `007`,
memo M004 and the register updates. Depends on Phase 1's re-run for every
number it quotes.

## Out of scope

**Full autodiff through the terrain.** Scoped and deferred. `calc_horizon` is a
recursive max-pool pyramid with `np.argsort`, integer bin slicing and
data-dependent Python recursion (`eigsep_terrain/dem.py:246`), so
differentiating it is a rewrite. `eigsep_terrain` does have a JAX ray marcher
(`ray_jax.py`) and a `PositionSolver` that fits antenna position against
horizon photographs, which is the path a full-autodiff version would take — but
a marcher's threshold-crossing gradient needs implicit differentiation to be
correct, which reduces to the same analytic derivative designed here. Revisit
only if joint MCMC sampling over antenna position becomes a requirement.

**Inner (antenna-to-mount) misalignment**, by the assumption recorded above.

**The receiver-temperature placeholder** in `simulate()`, which
`eigsep_cal`'s `ReceiverModel` supersedes on its own schedule.
