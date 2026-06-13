# Horizon Position Sensitivity

How much does the EIGSEP antenna temperature change when the antenna
position is wrong by 0.1, 1, or 10 m? This quantifies the spec on how
well the suspended antenna's position must be known, for forward
modelling. Zenith pointing only.

Design: `../docs/superpowers/specs/2026-06-13-horizon-position-sensitivity-design.md`
Plan:   `../docs/superpowers/plans/2026-06-13-horizon-position-sensitivity.md`

## Method (one line)

Continuous horizon curve from `eigsep_terrain.calc_horizon` per position
-> anti-aliased (fractional) open-sky mask -> `eigsim.simulate`
(zenith-only) -> difference vs nominal. Fractional masks resolve
sub-pixel horizon shifts that a boolean mask would floor to zero.

## Pipeline (two environments!)

`make_horizons.py` needs `eigsep_terrain`; everything else needs
`eigsim`. They are separate uv envs. `make_horizons.py` also needs
`PYTHONPATH` set because eigsep_terrain uses a flat (non-src) layout and
is not installed into its venv.

```bash
# 1. horizon curves (eigsep_terrain env)
PYTHONPATH=/home/christian/Documents/research/eigsep/eigsep_terrain \
uv run --project /home/christian/Documents/research/eigsep/eigsep_terrain \
    python horizon_position/make_horizons.py        # -> output/horizons_position.npz

# 2. per-position waterfalls (mock_analysis env, from monorepo root)
uv run python horizon_position/run_sims.py          # -> output/position_sims.npz

# 3. analysis
#    open notebooks/horizon_position.ipynb

# tests
uv run pytest horizon_position/ -v                  # pure-module unit tests
EIGSEP_SMOKE=1 uv run pytest horizon_position/test_smoke.py -v
```

## Outputs (`output/`, gitignored)

- `horizons_position.npz` — `alpha_h(az)` per position (+ enu, az grid).
- `position_sims.npz` — `t_sys (19, n_times, n_freqs)`, `fgnd (19, n_freqs)`, axes, metadata.

## Analysis modes

`analysis.delta_waterfall(..., mode)` with `mode` in:
- `uncorrected` (default) — raw `t_sys` difference.
- `oracle` — ground-loss corrected with each position's true `fgnd`.
- `miscorrected` — corrected with the *nominal* `fgnd` (position error
  unknown); equals `uncorrected / (1 - fgnd_nominal)`.
