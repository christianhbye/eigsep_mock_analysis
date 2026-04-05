# EIGSEP Mock Analysis

Monorepo for EIGSEP simulation and analysis code.

## Structure

- `eigsim/` — Simulation package (own pyproject.toml, src layout)
- `notebooks/` — Jupyter notebooks organized by project
  - `data_challenges/` — Data challenge analysis
  - `spatial_filter/` — Spatial filtering studies
  - Standalone notebooks at top level
- Future packages (e.g., inference) will follow the same pattern as `eigsim/`

## Development

- **Package manager:** uv (workspace mode)
- **Linting:** ruff
- **Testing:** pytest
- **CI:** GitHub Actions (lint + test on push/PR to main)

## Commands

```bash
uv sync --dev          # Install all workspace packages + dev deps
uv run pytest          # Run tests
uv run ruff check .    # Lint
uv run ruff format .   # Format
```

## Conventions

- Use src layout for all packages (`<pkg>/src/<pkg>/`)
- Each package has its own pyproject.toml
- Data files (*.npz, *.npy, *.fits, *.hdf5) are gitignored — do not commit large data
- Notebooks should not import from local paths outside the installed packages
