# Horizon Position Sensitivity — Design

**Date:** 2026-06-13
**Status:** Draft (awaiting review)

## Goal

EIGSEP is a *suspended* antenna over complex quarry terrain, so a small
translational error in the antenna position changes the horizon profile
and therefore the antenna temperature. To forward-model the instrument
we need a **spec on how accurately the antenna position must be known**.

This project quantifies the change in antenna temperature vs.
frequency and LST caused by displacing the antenna by 0.1, 1, and 10 m
along each axis, relative to the nominal (unperturbed) position. It is a
self-contained analysis project that *uses* `eigsim` and
`eigsep_terrain` but ships with neither — a sibling of
`horizon_chromaticity/`.

Scope guard: **zenith-pointing antenna only, no drive rotation**
(`N_ori = 1`).

## Background: how a horizon profile is made

The nominal antenna is at ENU `(E=1648, N=2024, U=1796) m`
(`eigsep_terrain` site `1P`, 114 m above the quarry floor). A horizon
profile is produced by ray-casting the Marjum DEM from the antenna
position:

- `eigsep_terrain.MarjumDEM.calc_horizon(e, n, u, n_az)` returns the
  **horizon elevation angle vs. azimuth** `α_h(φ)` — the elevation
  above which sky is visible along each azimuth. Because terrain is a
  height field, "open sky ⇔ elevation > `α_h(φ)`" fully describes the
  mask along a single azimuth.
- `α_h(φ)` is a **continuous function of the (continuous) antenna
  position**, so sub-meter displacements move it smoothly — there is no
  pixel-grid discretization floor.

The existing `horizon_chromaticity` `eigsep` case instead ray-traces a
HEALPix mask (`nside=64`) and resamples it to the MWSS grid
(`lmax=128`, ~1.4°/pixel). At that resolution a 0.1 m shift moves the
horizon ~0.05° (≈ 1/25 of a pixel) and a 1 m shift ~0.4° (≈ 1/3 of a
pixel), so a pixel-mask pipeline would report ΔT ≈ 0 for the small
shifts — a discretization artifact, not physics. This project therefore
works directly with `α_h(φ)`.

## Antenna positions

Nominal `p0 = (1648, 2024, 1796) m`. One axis perturbed at a time:

- **Axes:** x = East (E), y = North (N), z = Up (U).
- **Magnitudes:** {0.1, 1, 10} m.
- **Signs:** both (+ and −) — terrain is asymmetric, so sensitivity is
  direction-dependent.
- **Total:** nominal + 6 directions × 3 magnitudes = **19 positions**.

## Simulation parameters (shared with the nominal waterfall)

- **Pointing:** zenith only, no rotation (`N_ori = 1`).
- **Times:** one sidereal day, 1436 samples (~1 min cadence), start
  2026-07-01 06:00:00 UTC (matching `horizon_chromaticity`).
- **Frequencies:** 50–250 MHz, 1 MHz steps (201 channels, from config).
- **Sky:** GSM16 (TRJ, lo resolution, CMB included, from config).
- **Beam:** EIGSEP bowtie, band-limited at `lmax=128` (native
  `nside=64`), from `eigsim.load_beam()`.
- **Ground / receiver:** `T_gnd = 300 K`, `T_rcvr = 50 K` (from config).
- **Noise:** none. Scripts save raw noiseless quantities; metrics live
  in the notebook.

## Method

### Definitions

- `B(α, φ; ν)` — band-limited beam power, zenith-pointing.
  `Ω_B(ν) = ∮∮ B cosα dα dφ` (full-sphere beam integral).
- `α_h(φ; p)` — horizon elevation vs. azimuth for position `p`.
- `T_sky(n; ν, t)` — GSM16 brightness rotated to topocentric at time
  `t`.
- System temperature:
  `t_sys = T_ant_sky + fgnd·T_gnd + T_rcvr`, where
  `T_ant_sky = (1/Ω_B) ∮∮_{open} B·T_sky cosα dα dφ` and
  `fgnd = (1/Ω_B) ∮∮_{blocked} B cosα dα dφ`.

### Core idea: anti-aliased horizon masks through `eigsim.simulate`

The discretization floor that makes a 0.1 m shift vanish comes from
representing the horizon as a **boolean** mask on the beam grid — a
sub-pixel edge motion flips no pixels. The fix is to keep the mask
**fractional**: each grid cell carries its **open-sky fraction**
`W(θ, φ) ∈ [0, 1]` (1 = fully open, 0 = fully blocked, a partial value
for the cell the horizon edge crosses). A sub-pixel horizon move then
changes the boundary cell's fraction **continuously**, so the floor
disappears with no change to the grid resolution.

This is exactly the mask that `eigsim.simulate()` already consumes:
internally it forms `pixel * horizon` (a plain multiply) and computes
`fgnd = 1 − Σ(beam·horizon·w)/Ω_B`, normalizing by the full-sphere beam
integral `Ω_B`. croissant stores the passed mask as-is
(`self.horizon = jnp.asarray(horizon)`) — it does **not** booleanize —
so a float `W` flows through untouched. Therefore:

```
t_sys^p(t, ν) = eigsim.simulate(beam, ν, sky, t, elev=0, az=0,
                                horizon = W_p)          # zenith only
fgnd^p(ν)     = eigsim.compute_fgnd(beam, ν, elev=0, az=0, horizon = W_p)
```

We run this once per position (19 cheap zenith-only waterfalls), reusing
the precomputed sky ALM across positions, and difference the results.
The whole custom-numerics surface is a single function: build `W_p` from
the continuous horizon curve `α_h(φ; p)`.

**Why this still resolves 0.1/1/10 m.** The difference `t_sys^p − t_sys^0`
is driven by `B·(W_p − W_0)`, which is nonzero only in the **boundary
cells whose open-fraction changed** — the flip band of the
one-waterfall-plus-sliver picture, now realized as fractional cell
weights rather than a hand-rolled line integral. A 0.1 m shift changes
the boundary fraction by `≈ Δα_h / Δα_cell ≈ 0.05°/1.4° ≈ 4%` per ring
cell: small, smooth, and non-zero. The fractional weight is a
first-order-accurate stand-in for the sub-cell edge position, which is
the regime that matters for the small shifts; for 10 m whole cells flip
(exact) plus a fractional boundary.

**Building `W_p`.** `α_h(φ)` comes from `calc_horizon` on a fine azimuth
grid. For each beam-grid azimuth column `φ_j`, `θ_h = π/2 − α_h(φ_j)`,
and the cell open-fraction is the portion of the cell's polar extent
above the horizon, clipped to `[0, 1]`. **Frame mapping:** `calc_horizon`
azimuth is `atan2(E, N)` (from North toward East), while croissant's
beam/grid azimuth is measured from East (`beam_rot=0` ⇒ `φ=0` along ENU
East), so `φ = π/2 − az`, i.e. evaluate `α_h` at `az = π/2 − φ_j`. This
single mapping is verified against the existing nominal
`horizon_mwss.npz` (same site, croissant frame) before any sims run.

**Resolution knob.** The beam's native grid is `lmax = 128`
(~1.4°/cell); anti-aliasing removes the floor at that resolution. If
edge aliasing in the masked-beam SHT proves non-negligible (checked by
the 10 m anti-aliased-vs-boolean cross-check and the convergence test),
the beam is resampled to a finer working grid `L_work` (lossless inverse
SHT of the `lmax=128` beam ALM) and masks are rebuilt there. Default is
the native grid.

## Analysis modes (ground-loss handling)

`GLC(t_sys, fgnd) = (t_sys − T_rcvr − fgnd·T_gnd) / (1 − fgnd)`.

Three modes, selectable in the analysis (default = uncorrected). All
three are post-processing of the same stored quantities — no extra
simulation.

1. **Uncorrected** (default): `D1^p = Δt_sys^p = t_sys^p − t_sys^0`.
   Raw measured-temperature change.
2. **Oracle-corrected**: `D2^p = GLC(t_sys^p, fgnd^p) − GLC(t_sys^0,
   fgnd^0)`. The antenna is *known* to have moved and is corrected with
   its true `fgnd^p`. Non-zero residual = the irreducible change from
   the moved horizon sampling a different sky patch.
3. **Mis-corrected** (position error unknown): `D3^p = GLC(t_sys^p,
   fgnd^0) − GLC(t_sys^0, fgnd^0)`. The antenna moved but is corrected
   with the **nominal** `fgnd^0`. Because `fgnd^0` is held fixed and GLC
   is affine in `t_sys`, this collapses to

   ```
   D3^p = Δt_sys^p / (1 − fgnd^0)
   ```

   i.e. mis-correction **amplifies the raw error by the ground-loss
   gain** `1/(1−fgnd^0) > 1`. This is the error incurred by forward-
   modeling at the nominal position when the antenna has actually
   moved — the mode most directly tied to the spec's question.

Mode 3 vs mode 1 isolates "error from not knowing the position"; mode 2
isolates "error that is physically unavoidable even with perfect
knowledge."

## The spec / deliverable

Per position, the analysis produces **ΔT vs. (LST, frequency)** for the
selected mode (a difference waterfall). From each waterfall we reduce to
summary statistics.

**Scalar summary (one number per position).** The headline statistic is
the RMS of the difference waterfall over the whole (LST, frequency)
plane:

```
S(p) = sqrt( mean_{LST, ν} [ ΔT^p(LST, ν)² ] )
```

with the worst-case `max_{LST,ν} |ΔT^p|` reported alongside it. `S(p)`
is the "typical" temperature error injected by a position error `p`.
(Both are computed per mode.)

**The spec curve.** Plot `S` (and `max|ΔT|`) **vs. shift magnitude**
`δ ∈ {0.1, 1, 10} m`, one curve per axis (x/y/z), sign, and mode
(log–log). This is the deliverable the paper quotes: read off "antenna
position must be known to within X m to hold the typical/worst-case ΔT
below the error budget." Expected shape (see *Physical expectations*):
roughly **linear in δ** (slope ≈ 1 on log–log) for the small shifts,
possibly departing at 10 m where the sliver is no longer thin.

**Resolved reductions (diagnostics).** To expose *where* the error
lives and test the physical predictions:

- `RMS_LST |ΔT^p|(ν)` — a spectrum: the chromaticity of the error.
- `RMS_ν  |ΔT^p|(LST)` — a time series: the LST structure. The
  prediction is that x/y errors are strongly LST-modulated (sky
  azimuthal contrast) while z errors are flatter/broadband (ground-
  fraction offset).
- A few representative full ΔT waterfalls.

## Pipeline & files (`horizon_position/`)

```
positions.py       -> the 19 ENU positions (pure module, imported)
masks.py           -> anti-aliased open-sky weight W from α_h(φ) (pure module, imported)
make_horizons.py   -> output/horizons_position.npz   (positions + α_h(φ) per position; minutes)
run_sims.py        -> output/position_sims.npz        (t_sys^p(t,ν), fgnd^p(ν) for all 19; zenith-only; tens of minutes)
analysis.py        -> the three modes + summary statistics (pure module, imported + unit-tested)
notebooks/horizon_position.ipynb                      (modes + spec curves; loads npz + analysis.py only)
```

- `positions.py`: `NOMINAL_ENU` and `build_positions()` → the ordered 19
  `(name, enu)` pairs (nominal + ±x/±y/±z × {0.1, 1, 10} m).
- `masks.py`: `open_sky_weight(alpha_h, az_grid, thetas, phis)` → the
  fractional `W` on the beam grid (with the `φ = π/2 − az` frame
  mapping), plus a `boolean_weight` variant for the cross-check.
- `make_horizons.py`: builds the DEM via `MarjumDEM` (rebuilds from the
  TIFs already in `eigsep_terrain/data` if no cache), enumerates the 19
  positions, computes `α_h(φ)` for each via `calc_horizon` at high
  `n_az`, and records a hash of the position+horizon set so `run_sims.py`
  can guard against staleness (as `horizon_chromaticity/run_sims.py`
  does with `mask_sha`).
- `run_sims.py`: for each position, builds `W_p` (via `masks.py`), runs
  `eigsim.simulate` (zenith-only, `N_ori = 1`) and `eigsim.compute_fgnd`,
  reusing one precomputed sky ALM across positions. Checkpoints per
  position (resumable, like `horizon_chromaticity/run_sims.py`). Stores
  `t_sys` `(19, N_times, N_freqs)` and `fgnd` `(19, N_freqs)` plus axes
  and metadata. Storing absolute `t_sys`/`fgnd` (not differences) keeps
  all three analysis modes available as post-processing.
- `analysis.py`: pure functions for the three modes (uncorrected /
  oracle / mis-corrected) and the summary statistics `S(p)`, `max|ΔT|`,
  and the resolved reductions — imported by the notebook and unit-tested.
- The notebook loads npz from `output/` and imports `analysis.py`; it
  never imports the heavy scripts (same convention as
  `horizon_chromaticity`). The ground-loss mode is a notebook/analysis
  argument, default uncorrected.
- `output/` is gitignored. Own `CLAUDE.md` and `README.md`.

## Conventions (inherited)

- `os.environ.setdefault("JAX_ENABLE_X64", "1")` **before** any
  jax/s2fft/croissant/eigsim/eigsep_terrain import.
- Run everything from the monorepo root via `uv run`.
- Solid-angle / quadrature weights consistent with
  `eigsim.simulate()`.
- Receiver temperature and `T_gnd` from the eigsim config.

## Physical expectations (idealized limits)

These analytic limits set expectations the simulation should reproduce,
and explain *why* the real horizon is the interesting case. They are the
qualitative backbone of the validation below.

**Flat ground (θ = 90°).** A flat infinite plane is translation-
invariant: the horizon sits at elevation 0° in every azimuth regardless
of horizontal position, and (ignoring Earth curvature) regardless of
height. So **every** shift — x, y, *and* z — gives `ΔT = 0` exactly.
This is the null case.

**Symmetric quarry (constant θ_c).** Physically a circular rim of height
`Δh` at radius `R`, antenna at the center, `α_h = arctan(Δh/R)`. A
horizontal shift `δ` makes the near rim closer (`R−δ`, horizon **rises**)
and the far rim farther (`R+δ`, horizon **drops**): the horizon ring
*tilts* by `~δ/R`. But the antenna-temperature change is **suppressed by
near/far cancellation** — the ground gained on the near side cancels the
ground lost on the far side to first order, leaving

```
Δt_sys ∝ (δ/R) · (sky azimuthal contrast across the horizon ring),
```

for an azimuthally symmetric beam. Consequences:

- Horizontal sensitivity scales as `δ/R`: **bigger quarry → smaller
  effect, tighter quarry → larger**.
- The residual is driven by the *sky's* azimuthal structure, so it is
  **strongly LST-modulated** and vanishes for a uniform sky.
- **Vertical shifts do not cancel**: raising the antenna lowers the rim
  elevation uniformly in azimuth, shrinking the blocked region
  everywhere and changing the *total* ground fraction. So in a quarry, z
  produces a net, broadband (ground-fraction) ΔT even for a uniform sky
  — the opposite of the flat case.

**Real EIGSEP horizon (asymmetric).** Terrain distances/heights differ
across azimuth, so the near/far slivers no longer pair up and the
cancellation is **incomplete**. This is exactly why the realistic
horizon is more sensitive than the symmetric quarry, and why we simulate
it. Concrete predictions to check against:

1. **x/y errors** appear mainly through **LST modulation** (sky contrast
   across the moved horizon edge): `RMS_ν|ΔT|(LST)` should be strongly
   structured in LST.
2. **z errors** look more like a **broadband ground-fraction offset**:
   flatter in LST, set by how much open-sky solid angle the move adds or
   removes.
3. **Scaling**: for thin slivers (0.1, 1 m) `ΔT ∝ δ`, so `S(δ)` is
   ~linear (slope ≈ 1 on log–log); the 10 m sliver (~5°) is thick enough
   that beam/sky curvature across it can bend the curve.
4. With nearest terrain `~100 m`, `δ/R ~ 10⁻³` per 0.1 m sets the rough
   amplitude scale.

## Validation

- **Idealized limits:** the predictions above (flat → 0 in all axes;
  x/y LST-modulated; z broadband; `S ∝ δ` for small shifts) are
  reproduced. A symmetric-quarry analytic check can be run as an
  optional unit comparison.
- **Convergence to zero:** `Δt_sys^p → 0` smoothly as the shift → 0
  (0.1 m is the smallest, and should be the smallest ΔT).
- **Sign:** at low frequency `T_sky ≫ T_gnd`, so a rising horizon
  (sky → ground) must give negative `Δt_sys`.
- **Anti-aliasing cross-check at 10 m:** for a 10 m shift the horizon
  moves several grid cells, so a **boolean** mask already resolves it.
  The anti-aliased `W` and the boolean mask must give the same `t_sys`
  to within grid error at 10 m — confirming the fractional weighting
  does not bias large shifts. (Smaller shifts cannot be cross-checked
  this way; that is the whole point of anti-aliasing.)
- **Frame mapping:** the nominal `α_h(φ)` mapped to the beam grid (via
  `φ = π/2 − az`) must reproduce the existing nominal `horizon_mwss.npz`
  (boundary elevation per azimuth) within method differences.
- **Mode 3 identity:** `D3^p` computed via GLC must equal
  `Δt_sys^p / (1 − fgnd^0)` to numerical precision.

## Limitations / caveats

- **DEM resolution.** The Marjum DEM is sampled at 0.5 m. The 0.1 m
  result reflects the continuous *geometric* response of the horizon to
  a sub-pixel antenna move (distances are continuous); it does not add
  terrain structure finer than 0.5 m. This is the best available and is
  stated honestly in the writeup.
- **Horizon feature switches.** `α_h(φ)` is mostly smooth in position
  but can jump where the azimuth's tallest blocking feature changes;
  such jumps are physical and handled by integrating the actual sliver.
- **Edge aliasing.** Masking the band-limited beam with a sharp horizon
  puts power above `lmax` that aliases into the retained harmonics. The
  anti-aliased ramp band-limits the edge to ~1 cell, keeping this small;
  if the cross-checks flag it, the `L_work` knob (resample the beam to a
  finer grid) reduces it.

## Out of scope

- Non-zenith pointings / drive rotations.
- Combined multi-axis (diagonal) displacements — one axis at a time.
- Antenna height *survey* (the coarse `height` sweep in
  `horizon_models_v000.npz`); here z is treated identically to x and y
  at the 0.1/1/10 m scale.
- Terrain scattering / non-specular ground; the ground is the usual
  isothermal `T_gnd`.
