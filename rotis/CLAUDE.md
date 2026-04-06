# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync --dev                          # Install package + dev deps
uv run pytest rotis/tests -v           # Run rotis tests
uv run pytest rotis/tests/test_fisher.py  # Run a single test file
uv run pytest -k "test_name"           # Run a single test by name
uv run ruff check .                    # Lint
uv run ruff format .                   # Format
```

Always use `uv run` to invoke Python tools, never bare `python` or `pytest`.

## Architecture

rotis performs joint inference of the EIGSEP antenna beam pattern and sky brightness temperature from rotation-diversity observations. It is **not** a forward simulator -- it imports `eigsim` for all data generation and simulation.

### Module dependency flow (planned)

```
eigsim (data + forward model)
   |
   v
fisher.py       estimator.py
   |                |
   v                v
als.py          s2fft (Wigner D-matrices)
   |
   v
JAX (autodiff, vmap, jit)
```

- **`fisher.py`** (Milestone 1) -- Fisher information matrix, coverage kernel, LST sampling error, sky asymmetry SNR. Characterizes the information content and systematic floor of the rotation grid before any inference.
- **`estimator.py`** (Milestone 2) -- Constructive estimator implementing the three-step identifiability proof: Wigner projection, rank-1 SVD, ground-term linear system.
- **`als.py`** (Milestone 2) -- Alternating least squares, Gibbs sampler, beam prior from HFSS, prior sweep.
- **`multifreq.py`** (Milestone 3) -- Multi-frequency stacking.
- **`gsm_basis.py`** (Milestone 3) -- GSM eigenmode decomposition.
- **`inference.py`** (Milestone 3) -- Bayesian model comparison for 21-cm signal.
- **`injection_recovery.py`** (Milestone 3) -- Injection-recovery validation.

### Key conventions (inherited from eigsim)

- **alm format:** s2fft `[L, 2L-1]` complex array, all m including negative
- **Rotation matrices:** 3x3 arrays, composition `R = Rx(elevation) @ Rz(azimuth)`
- **Wigner D-matrices:** accessed directly from s2fft (not via eigsim)
- **Config:** loaded via `eigsim.load_config()`, returns a dict

### What rotis does NOT do

- No croissant imports (forward simulation goes through eigsim)
- No HEALPix pixel operations (works in harmonic domain)
- No s2fft spherical harmonic transforms (only Wigner D-matrices)

### Mathematical reference

The identifiability theorem and estimation algorithm are derived in `memo/sample631.tex`. The full implementation spec with milestones is in `PROBLEM_STATEMENT.md`.
