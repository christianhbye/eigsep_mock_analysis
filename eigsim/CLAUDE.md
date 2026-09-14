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
config.py          data.py
   |                  |
   v                  v
simulate.py <---- rotations.py
   |                  |
   v                  v
croissant-sim     s2fft (JAX)
```

- **`config.py`** — Loads YAML experiment config (defaults in `src/eigsim/configs/eigsep.yaml`). Returns a dict with location, frequencies, ground temperature, beam/horizon file paths.
- **`data.py`** — Loads pre-computed `.npz` beam patterns and horizon masks from `data/`. Default files use MWSS sampling.
- **`rotations.py`** — Models the EIGSEP mechanical drive (elevation via Rx, azimuth via Rz). `rotate_beam_data()` does forward SHT -> Wigner-D rotation -> inverse SHT using s2fft+JAX (`jax.vmap` over frequencies).
- **`simulate.py`** — `simulate()` orchestrates multi-orientation runs: for each (elevation, azimuth) pair, rotates the beam, creates a `croissant.Simulator`, runs it, and stacks results into `(N_orientations, N_times, N_freqs)`.

### Drive rotation convention

- Composition order: `R = Rx(elevation) @ Rz(azimuth)`
- Elevation 0 = zenith; positive tilts toward South (right-hand rule about East axis)
- Azimuth positive = counterclockwise from above

### Data files

The `data/` directory contains `.npz` files (gitignored) with beam patterns and horizon masks in both HEALPix and MWSS samplings. MWSS variants (`*_mwss.npz`) are the defaults. Beam shape: `(N_freqs, N_theta, N_phi)`. Horizon: `(N_theta, N_phi)` with NaN for open sky.

`load_beam()` loads the file named by the config's `beam.file`; `load_config("<name>")` loads a packaged config by name. The config's `frequencies` must be the beam's frequencies, because croissant requires the two to match exactly.

| Config | Beam file | Frequencies | Use |
|---|---|---|---|
| `eigsep` (default) | `eigsep_bowtie_v001_mwss.npz` | the 52 HFSS channels, 46.875–246.09 MHz, 3.906 MHz apart (D5 channels k = 192, 208, …, 1008) | new work |
| `eigsep_1mhz` | `eigsep_bowtie_v001_1mhz_mwss.npz` | 50–246 MHz, 1 MHz | studies that need a 1 MHz grid; the beam between channels is a cubic spline |
| `eigsep_v000` | `eigsep_bowtie_v000_mwss.npz` | 50–250 MHz, 1 MHz | frozen; `horizon_position` and `horizon_chromaticity` (instrument paper) pin it |

- **v001** is |E|² from Dominic's HFSS complex far field (`data-analysis/hfss_beam_maps/bowtie_beam.npz`), which matches BK's 2025-10-31 bowtie-on-box simulation. Each channel is normalised to directivity (integral 4π); `realized_efficiency` is stored alongside. The source is HEALPix nside 32, transformed at lmax 64 and zero-padded to lmax 128 to share the horizon's grid. Rebuild with `uv run python eigsim/scripts/make_bowtie_v001.py`; each file carries `description` and `provenance`.
- **v000** is an older bowtie model of unrecorded provenance. It agrees with v001 at 50 MHz but not above about 150 MHz (pattern correlation 0.33 at 246 MHz).
- Do not change `eigsep_v000.yaml`; add a new config instead.

### Comparing with EIGSEP data

The beams are free-space antenna models. They include neither balun loss nor the coax from the balun to the RF switch. EIGSEP calibrates at the switch, so a calibrated antenna temperature or a measured antenna S11 includes the balun and that coax. The Deployment 5 coax was destroyed, so there are no S-parameters for it, and every comparison with D5 data needs a balun and coax model with priors. Keep that model out of eigsim: the generator adds it (`eigsep_cal/docs/interface.md` § 3, § 4.3, branch `rebuild`).

### Key external dependencies

- **croissant-sim** — `Simulator`, `Beam`, `Sky`, utility functions (Euler angle conversion, etc.)
- **s2fft** — Spherical harmonic transforms (forward/inverse) and Wigner-D rotations
- **JAX** — Vectorization (`jax.vmap`) and array operations in rotation code
