# EIGSEP Mock Analysis

Monorepo for EIGSEP simulation and analysis code.

## Structure

- `eigsim/` — Simulation package (own pyproject.toml, src layout)
- `rotis/` — Inference package for joint beam/sky recovery (own pyproject.toml, src layout)
- `notebooks/` — Jupyter notebooks organized by project
  - `data_challenges/` — Data challenge analysis
  - `spatial_filter/` — Spatial filtering studies
  - Standalone notebooks at top level
- Future packages (e.g., inference) will follow the same pattern as `eigsim/`

## Development

- **Package manager:** uv (workspace mode)
- **Linting:** ruff
- **Testing:** pytest
- **Pre-commit:** ruff via pre-commit hooks
- **CI:** GitHub Actions (shared lint + per-package tests + release-please)
- **Releases:** release-please in manifest mode (per-package versioning)

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
- Use conventional commits (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, etc.)
- New packages get their own test workflow (`.github/workflows/<pkg>-test.yml`) with path filters
- Never use `python -c "..."` for multiline scripts. Write to a temp file and run that instead.
