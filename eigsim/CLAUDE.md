# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync --dev                          # Install package + dev deps
uv run pytest                          # Run all tests
uv run pytest tests/test_rotations.py  # Run a single test file
uv run pytest -k "test_name"           # Run a single test by name
uv run ruff check .                    # Lint
uv run ruff format .                   # Format
uv run python scripts/hp2mwss.py       # Convert HEALPix data to MWSS sampling
```

Always use `uv run` to invoke Python tools, never bare `python` or `pytest`.

## Architecture

eigsim is a thin simulation wrapper around [croissant-sim](https://github.com/christianhbye/croissant) for the EIGSEP lunar radio experiment. The core workflow is:

**config -> load data -> rotate beam per orientation -> simulate visibilities**

### Module dependency flow

```
config.py          data.py          horizon.py
   |                  |                 |
   v                  v                 v
simulate.py <---- rotations.py   (open-sky weight W,
   |                  |            passed in as beam_kw={"horizon": W})
   v                  v
croissant-sim     s2fft (JAX)
```

- **`config.py`** — Loads YAML experiment config (defaults in `src/eigsim/configs/eigsep.yaml`). Returns a dict with location, frequencies, ground temperature, beam/horizon file paths.
- **`data.py`** — Loads pre-computed `.npz` beam patterns and horizon masks from `data/`. Default files use MWSS sampling.
- **`horizon.py`** — `open_sky_weight(alpha_h, az_grid, lmax, sub=180)` turns a continuous horizon elevation curve `alpha_h(az)` (with `az = atan2(E, N)`, North->East) into the fractional open-sky weight `W(theta, phi)` in `[0, 1]` on the beam's MWSS grid — 1 = open sky, 0 = blocked. Frame map: the grid's `phi` runs from ENU East (croissant `beam_rot=0`), so `phi = pi/2 - az`. `W` is the fraction of each grid **cell** above the horizon, integrated over theta *and* over phi (`sub` sub-samples per phi cell). **That phi-cell integral is the band-limiting**, so there is deliberately no `reduce_azimuth` counterpart and callers must not pre-reduce the curve — doing so applies the averaging twice (`horizon_position/make_horizons.py`, "WHO REDUCES, AND WHY"). Everything is built in `jnp` so `dW/d alpha_h` comes from autodiff, which is the point: a boolean mask has zero gradient almost everywhere and cannot carry a horizon derivative at all (`horizon_position/make_sensitivity.py` `jax.jvp`s through it). It is an *alternative* to, not a replacement for, `data.load_horizon`, which still loads the packaged `.npz` mask (NaN = open sky) that the pinned-paper scripts use. `mwss_grid(lmax)` returns the grid's `(thetas, phis)`.
- **`rotations.py`** — Models the EIGSEP mechanical drive (elevation via Rx, azimuth via Rz). `rotate_beam_data()` does forward SHT -> Wigner-D rotation -> inverse SHT using s2fft+JAX (`jax.vmap` over frequencies).
- **`simulate.py`** — `simulate()` orchestrates multi-orientation runs: for each (elevation, azimuth) pair, rotates the beam, convolves it with the sky (orientation graph JIT-compiled once per call; croissant's convolution outside it), and stacks results into `(N_orientations, N_times, N_freqs)`, adding the receiver temperature. `simulate_path()` is D5 path mode: one orientation per time sample, grouped by unique orientation, returning `(N_times, N_freqs)` with **no** receiver term (`SkyTemperature.t_ant_k`, `eigsep_cal/docs/api.md` § 5.1). Both share `_setup()`, so the beam transform and the orientation graph are compiled once per call; only croissant's sky convolution specialises on the number of times, which is cheap.

### Drive rotation convention

- Composition order: `R = Rx(elevation) @ Rz(azimuth)`
- Elevation 0 = zenith; positive tilts toward South (right-hand rule about East axis)
- Azimuth positive = counterclockwise from above
- Optional outer mount-to-ground misalignment: `R = R_mis @ Rx(elevation) @ Rz(azimuth)`, static in the topocentric frame. X misalignment is omitted because it is exactly an elevation encoder offset (`Rx(eps) @ Rx(el) == Rx(eps + el)`); only Y (levelling) and Z (azimuth-reference) tilts are identifiable. `simulate`, `simulate_path` and `compute_fgnd` take `misalignment=`; `rotate_beam_data`/`rotate_alm_to_beam` do **not** — they model the commanded drive only.
- Misalignment sign convention, right-handed about each ENU axis and load-bearing for the sign of `dT/d eps`: positive `tilt_y_deg` tilts the boresight toward **East** (mirror of elevation, which tilts toward South); positive `tilt_z_deg` rotates the azimuth reference **East toward North**, the same sense as the turntable, and at zenith is exactly an azimuth offset of the same sign (`Rz(eps) @ Rx(0) @ Rz(az) == Rx(0) @ Rz(az + eps)`), so at zenith `eps_z` is not separately identifiable from turntable azimuth.

### Data files

The `data/` directory contains `.npz` files (gitignored) with beam patterns and horizon masks in both HEALPix and MWSS samplings. MWSS variants (`*_mwss.npz`) are the defaults. Beam shape: `(N_freqs, N_theta, N_phi)`. Horizon: `(N_theta, N_phi)` with NaN for open sky.

`load_beam()` loads the file named by the config's `beam.file`; `load_config("<name>")` loads a packaged config by name. The config's `frequencies` must be the beam's frequencies, because croissant requires the two to match exactly.

| Config | Beam file | Frequencies | Use |
|---|---|---|---|
| `eigsep` (default) | `eigsep_bowtie_v002_mwss.npz` | the 52 HFSS frequencies, 50.78–250.00 MHz, 3.906 MHz apart (D5 channels c = 208, 224, …, 1024) | new work |
| `eigsep_1mhz` | `eigsep_bowtie_v002_1mhz_mwss.npz` | 51–250 MHz, 1 MHz | studies that need a 1 MHz grid; the beam between channels is a cubic spline |
| `eigsep_v001` | `eigsep_bowtie_v001_mwss.npz` | 46.875–246.09 MHz as labelled (**one step low**) | frozen; reproduces studies made with the v001 default |
| `eigsep_v001_1mhz` | `eigsep_bowtie_v001_1mhz_mwss.npz` | 50–246 MHz | frozen |
| `eigsep_v000` | `eigsep_bowtie_v000_mwss.npz` | 50–250 MHz, 1 MHz | frozen; `horizon_position` and `horizon_chromaticity` (instrument paper) pin it |

- **v002** is |E|² from Dominic's **native** HFSS complex far field: the 1° θ/φ export (`~/Documents/research/eigsep/beam_models/hfss_native_sep2026/`, built into `bowtie_native_mwss_L180.npz` by its `build_mwss.py`), which is exactly MWSS at L = 180. Each channel is normalised to directivity (integral 4π) with the source's own quadrature; `realized_efficiency` is stored alongside. The transform is exact and truncated to the horizon's lmax 128 (dropped power ≤ 2e-12). Rebuild with `uv run python eigsim/scripts/make_bowtie_v002.py`; each file carries `description` and `provenance`. **No HEALPix anywhere: use MWSS beams only** (eigsep_analysis MOD-62).
- **v001** was built from `data-analysis/hfss_beam_maps/bowtie_beam.npz`, the same export resampled onto HEALPix nside 32, whose **frequency labels are one 3.906 MHz step low**: v001 slice i is the v002 beam at index i (pattern within a few percent, the HEALPix residue) but labelled 46.875 + 3.90625·i MHz instead of 50.78125 + 3.90625·i MHz (confirmed by Dominic, 2026-09-16; pinned by `test_v001_is_v002_one_frequency_label_low`). At a given label the two beams differ by up to ~30 % pointwise. Kept frozen only to reproduce earlier results.
- **v000** is an older bowtie model of unrecorded provenance. It agrees with v001 at 50 MHz but not above about 150 MHz (pattern correlation 0.33 at 246 MHz); that comparison was made on v001's mislabelled grid and has not been repeated against v002.
- Do not change `eigsep_v000.yaml`; add a new config instead.

### Comparing with EIGSEP data

The beams are free-space antenna models. They include neither balun loss nor the coax from the balun to the RF switch. EIGSEP calibrates at the switch, so a calibrated antenna temperature or a measured antenna S11 includes the balun and that coax. The Deployment 5 coax was destroyed, so there are no S-parameters for it, and every comparison with D5 data needs a balun and coax model with priors. Keep that model out of eigsim: the generator adds it (`eigsep_cal/docs/api.md` § 3, § 4.3; D5 notes in the eigsep_analysis workspace `docs/interface.md` § 3).

The receiver temperature differs between the two entry points. `simulate()` adds the config's `receiver.temperature`; `simulate_path()` adds nothing, so its output is `SkyTemperature.t_ant_k` (spec § 5.1, § 7). Pass `t_rcvr=0.0` when feeding path-mode output to `correct_ground_loss()`, or it subtracts a receiver term that was never added. The scalar is a placeholder that eigsep_cal's `ReceiverModel` supersedes, but do **not** remove it from `simulate()` yet: the spec pins the grid output as unchanged (§ 7), `horizon_position` and `horizon_chromaticity` save `t_sys` with it in and subtract it downstream, and the notebook re-runs check npz byte-identity. Dropping it is a follow-up for after the instrument paper is accepted.

### Key external dependencies

- **croissant-sim** — `Simulator`, `Beam`, `Sky`, utility functions (Euler angle conversion, etc.)
- **s2fft** — Spherical harmonic transforms (forward/inverse) and Wigner-D rotations
- **JAX** — Vectorization (`jax.vmap`) and array operations in rotation code
