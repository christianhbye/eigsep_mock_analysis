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

### Key external dependencies

- **croissant-sim** — `Simulator`, `Beam`, `Sky`, utility functions (Euler angle conversion, etc.)
- **s2fft** — Spherical harmonic transforms (forward/inverse) and Wigner-D rotations
- **JAX** — Vectorization (`jax.vmap`) and array operations in rotation code
