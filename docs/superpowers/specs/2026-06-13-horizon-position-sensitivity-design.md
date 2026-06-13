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

### Core idea: one waterfall + flip-band corrections

Compute the expensive sky×beam convolution **once**, for the nominal
position: the full waterfall `t_sys^0(t, ν)` and `fgnd^0(ν)`. Every
perturbed position is then a cheap correction integrated **only over
the directions whose horizon status flips** between nominal and shifted:

```
t_sys^p(t, ν) = t_sys^0(t, ν) + Δt_sys^p(t, ν)

Δt_sys^p(t, ν) = (1/Ω_B) ∮dφ ∫_{α_h^p(φ)}^{α_h^0(φ)}
                     B(α, φ; ν) · [T_sky(α, φ; t, ν) − T_gnd] · cosα dα
```

The inner integral runs over the **elevation sliver** between the
shifted and nominal horizon curves (note the limits run `α_h^p → α_h^0`,
so a rising horizon — `α_h^p > α_h^0` — inverts them and yields a
negative contribution):

- Where the horizon **rose** (sky → ground): the sliver contributes
  `−(T_sky − T_gnd)` — sky lost, ground gained.
- Where the horizon **dropped** (ground → sky): the reverse.
- Unflipped directions, the receiver term, and the entire unchanged
  sky/ground cancel exactly and never enter.

Because only the horizon ring is ever evaluated, this is a 1-D
azimuthal integral that resolves 0.1/1/10 m uniformly. For thin slivers
(sub-degree shifts) it is the first-order limit `B·(T_sky−T_gnd)·cosα·
(α_h^0 − α_h^p)` per azimuth; for the ~5° sliver of a 10 m shift the sliver is
sub-sampled in elevation so the beam/sky variation across it is
captured.

The LST dependence enters because `T_sky` in the flipped directions
rotates through the sidereal day, so `Δt_sys^p` is itself a (smaller)
waterfall, not a scalar.

### Absolute ground fractions

For the ground-loss modes we also need `fgnd^p(ν)` per position. This is
a beam integral over the blocked region with the horizon as the
elevation limit:

```
fgnd^p(ν) = (1/Ω_B) ∮dφ ∫_{−π/2}^{α_h^p(φ)} B(α, φ; ν) cosα dα
```

computed by direct quadrature from `α_h(φ; p)` (no LST, cheap, and
consistent across all positions because it uses the same `α_h`
representation).

### What is evaluated where

- **Nominal waterfall** `t_sys^0(t, ν)` and `fgnd^0(ν)`: the standard,
  well-tested `eigsim.simulate()` / `eigsim.compute_fgnd()` path
  (`lmax=128`), run once for the nominal horizon. The absolute baseline
  is smooth, so `lmax=128` is fine here — we are *not* differencing at
  that resolution.
- **Beam along the horizon ring**: synthesized from the beam `alm`
  (`lmax=128`) at the ring directions.
- **Sky along the horizon ring**: rotated to topocentric per time using
  the same machinery `eigsim` uses for the sky `alm`. Each fixed
  alt/az direction maps to a constant declination, so the LST axis can
  be reduced to a right-ascension shift along precomputed declination
  rings (optimization, not required for correctness).

### Why this split

The primary metric (uncorrected ΔT) needs **only** the sliver integral,
which is fully floor-free and method-pure. The absolute nominal
baseline (used only by the ground-loss modes) is the one quantity taken
at `lmax=128`; the resulting seam between the `lmax=128` baseline and
the high-res differences is negligible for the metric, which is a
difference. (Alternative considered: compute every absolute on one
consistent fine `(α, φ)` grid. Rejected for now — more custom
integration code for a baseline that cancels in the metric. Easy to add
later if wanted.)

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
selected mode (waterfall). Summary products:

- For each axis / direction / magnitude: `max|ΔT|` and `RMS|ΔT|` over
  the (LST, frequency) plane.
- Summary curves: `|ΔT|` vs. shift magnitude, per axis and per mode,
  reading off "antenna position must be known to within X m to hold ΔT
  below the error budget."
- Representative ΔT waterfalls.

## Pipeline & files (`horizon_position/`)

```
make_horizons.py   -> output/horizons_position.npz   (positions + α_h(φ) per position; minutes)
run_reference.py   -> output/reference_nominal.npz    (t_sys^0(t,ν), fgnd^0(ν); zenith-only; minutes)
compute_deltas.py  -> output/deltas.npz               (Δt_sys^p(t,ν), fgnd^p(ν); reconstructs t_sys^p; minutes)
notebooks/horizon_position.ipynb                      (the three modes + spec; loads npz only)
```

- `make_horizons.py`: builds the DEM via `MarjumDEM` (rebuilds from the
  TIFs already in `eigsep_terrain/data` if no cache), enumerates the 19
  ENU positions, computes `α_h(φ)` for each via `calc_horizon` at high
  `n_az`, and also writes the nominal MWSS mask (for `run_reference`).
  Records a hash of the horizon set so downstream files can guard
  against staleness (as `run_sims.py` does with `mask_sha`).
- `run_reference.py`: nominal waterfall + ground fraction via `eigsim`
  (`--zenith-only` semantics; `N_ori = 1`).
- `compute_deltas.py`: the sliver and blocked-region integrals; stores
  `Δt_sys^p`, `fgnd^p`, and reconstructed `t_sys^p`. Stores enough to
  let the notebook build all three modes; a `--mode` / ground-loss flag
  exists but, since the modes are cheap post-processing, the default is
  to store the ingredients and select the mode in the notebook.
- The notebook only loads npz from `output/`; it never imports the
  scripts (same convention as `horizon_chromaticity`).
- `output/` is gitignored. Own `CLAUDE.md` and `README.md`.

## Conventions (inherited)

- `os.environ.setdefault("JAX_ENABLE_X64", "1")` **before** any
  jax/s2fft/croissant/eigsim/eigsep_terrain import.
- Run everything from the monorepo root via `uv run`.
- Solid-angle / quadrature weights consistent with
  `eigsim.simulate()`.
- Receiver temperature and `T_gnd` from the eigsim config.

## Validation

- **Convergence to zero:** `Δt_sys^p → 0` smoothly as the shift → 0
  (0.1 m is the smallest, and should be the smallest ΔT).
- **Sign:** at low frequency `T_sky ≫ T_gnd`, so a rising horizon
  (sky → ground) must give negative `Δt_sys`.
- **Cross-check at 10 m:** the 10 m shift is several `lmax=128` pixels,
  so it is resolvable by a brute-force `nside=64` mask run through
  `eigsim.simulate()`. The sliver result must agree with that coarse
  run within grid error. (Smaller shifts cannot be cross-checked this
  way — that is the whole reason for the sliver method.)
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
- **Baseline seam.** The `lmax=128` nominal baseline vs. high-res
  differences (see "Why this split"); negligible for the difference
  metric.

## Out of scope

- Non-zenith pointings / drive rotations.
- Combined multi-axis (diagonal) displacements — one axis at a time.
- Antenna height *survey* (the coarse `height` sweep in
  `horizon_models_v000.npz`); here z is treated identically to x and y
  at the 0.1/1/10 m scale.
- Terrain scattering / non-specular ground; the ground is the usual
  isothermal `T_gnd`.
