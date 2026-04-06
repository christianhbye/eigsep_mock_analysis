# EIGSEP Mock Analysis

[![eigsim](https://codecov.io/gh/christianhbye/eigsep_mock_analysis/branch/main/graph/badge.svg?flag=eigsim)](https://codecov.io/gh/christianhbye/eigsep_mock_analysis?flags[0]=eigsim)

Simulation and analysis workspace for the [EIGSEP](https://github.com/eigsep) experiment — a 21 cm cosmology instrument targeting the Cosmic Dawn signal.

## Repository structure

```
mock_analysis/
├── eigsim/              # Simulation package
│   ├── pyproject.toml
│   ├── src/eigsim/
│   └── tests/
├── notebooks/           # Analysis notebooks
│   ├── data_challenges/ # Mock data challenge analyses
│   ├── spatial_filter/  # Spatial filtering studies
│   ├── cryofunk.ipynb
│   ├── eigenmodes.ipynb
│   └── linear_mapmaking.ipynb
├── pyproject.toml       # Workspace root
└── CLAUDE.md
```

## Getting started

This project uses [uv](https://docs.astral.sh/uv/) for Python dependency management.

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all packages and dev dependencies
uv sync --dev

# Run tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .
```

## Packages

| Package | Description | Status |
|---------|-------------|--------|
| `eigsim` | EIGSEP simulation code | In development |

## Notebooks

| Notebook | Description |
|----------|-------------|
| `cryofunk.ipynb` | Cryogenic function basis analysis |
| `eigenmodes.ipynb` | Eigenmode decomposition (JAX) |
| `linear_mapmaking.ipynb` | Linear map-making with croissant |
| `data_challenges/` | Mock observation data challenges |
| `spatial_filter/` | Spatial filtering in harmonic and pixel space |
