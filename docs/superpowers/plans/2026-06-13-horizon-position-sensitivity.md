# Horizon Position Sensitivity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantify how a 0.1/1/10 m error in the EIGSEP antenna position changes the antenna temperature vs. frequency and LST, relative to the nominal position, as a spec on how well the position must be known.

**Architecture:** A new self-contained analysis project `horizon_position/` (sibling of `horizon_chromaticity/`). `eigsep_terrain.calc_horizon` gives a *continuous* horizon elevation curve `α_h(φ)` per antenna position; we turn each into an **anti-aliased (fractional) open-sky mask** on the beam's MWSS grid and run `eigsim.simulate` (zenith-only) per position. Because the mask is fractional, sub-pixel horizon shifts resolve without a boolean-grid floor. Differencing the per-position waterfalls gives ΔT, analyzed in three ground-loss modes.

**Tech Stack:** Python, NumPy, `eigsim` (+ croissant, s2fft, pygdsm GSM16), `eigsep_terrain` (DEM + ray/horizon tools), pytest. Two separate uv environments (see below).

**Spec:** `docs/superpowers/specs/2026-06-13-horizon-position-sensitivity-design.md`

---

## Two-environment note (critical)

`eigsep_terrain` and `eigsim` live in **separate** uv environments:

- **`make_horizons.py`** imports `eigsep_terrain` → run with the eigsep_terrain env:
  ```bash
  uv run --project /home/christian/Documents/research/eigsep/eigsep_terrain \
      python horizon_position/make_horizons.py
  ```
- **`run_sims.py`** and the pure modules/tests import `eigsim`/`s2fft` → run with the default mock_analysis env from the monorepo root:
  ```bash
  uv run python horizon_position/run_sims.py
  uv run pytest horizon_position/ -v
  ```

`eigsep_terrain` is **not** importable in the mock_analysis env (verified), so `make_horizons.py` is the only script that needs the eigsep_terrain env. It hands off a plain `.npz` (`α_h(φ)` per position) consumed by `run_sims.py`.

## File structure

```
horizon_position/
  positions.py        # the 19 ENU positions (pure)
  masks.py            # anti-aliased open-sky weight from alpha_h (pure; uses s2fft grid)
  analysis.py         # ground-loss modes + summary statistics (pure)
  make_horizons.py    # eigsep_terrain env: alpha_h(phi) per position -> output/horizons_position.npz
  run_sims.py         # eigsim env: t_sys^p, fgnd^p for all 19 -> output/position_sims.npz
  test_positions.py   # unit tests (mock_analysis env)
  test_masks.py       # unit tests (mock_analysis env)
  test_analysis.py    # unit tests (mock_analysis env)
  test_smoke.py       # EIGSEP_SMOKE-gated end-to-end (run_sims) test
  notebooks/horizon_position.ipynb   # modes + spec curves (loads npz + analysis.py)
  output/.gitignore   # ignore everything in output/
  README.md
  CLAUDE.md
```

Tests for the pure modules add the project dir to `sys.path` (these dirs are plain script collections, not packages — matching `horizon_chromaticity/`).

---

## Task 1: Scaffold the project directory

**Files:**
- Create: `horizon_position/output/.gitignore`
- Create: `horizon_position/.gitignore`

- [ ] **Step 1: Create the output gitignore**

Create `horizon_position/output/.gitignore`:

```
*
!.gitignore
```

- [ ] **Step 2: Create a project gitignore for the DEM cache**

Create `horizon_position/.gitignore`:

```
__pycache__/
*.pyc
```

- [ ] **Step 3: Verify the directory exists and is ignored**

Run: `cd /home/christian/Documents/research/eigsep/mock_analysis && ls horizon_position/output/ && git status --porcelain horizon_position/`
Expected: `output/.gitignore` and `.gitignore` show as untracked (`??`); nothing else.

- [ ] **Step 4: Commit**

```bash
cd /home/christian/Documents/research/eigsep/mock_analysis
git add horizon_position/.gitignore horizon_position/output/.gitignore
git commit -m "chore(horizon_position): scaffold project directory"
```

---

## Task 2: `positions.py` — the 19 antenna positions

**Files:**
- Create: `horizon_position/positions.py`
- Test: `horizon_position/test_positions.py`

- [ ] **Step 1: Write the failing test**

Create `horizon_position/test_positions.py`:

```python
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from positions import NOMINAL_ENU, build_positions  # noqa: E402


def test_count_and_nominal_first():
    pos = build_positions()
    assert len(pos) == 19
    assert pos[0][0] == "nominal"
    assert np.allclose(pos[0][1], NOMINAL_ENU)


def test_names_unique():
    names = [n for n, _ in build_positions()]
    assert len(set(names)) == 19


def test_shifts_apply_to_correct_axis():
    d = dict(build_positions())
    # x = East (index 0), y = North (1), z = Up (2)
    assert np.allclose(d["x_p_10"] - NOMINAL_ENU, [10.0, 0.0, 0.0])
    assert np.allclose(d["y_m_1"] - NOMINAL_ENU, [0.0, -1.0, 0.0])
    assert np.allclose(d["z_p_0p1"] - NOMINAL_ENU, [0.0, 0.0, 0.1])


def test_nominal_unmodified():
    # build_positions must not mutate the module constant
    build_positions()
    assert np.allclose(NOMINAL_ENU, [1648.0, 2024.0, 1796.0])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/christian/Documents/research/eigsep/mock_analysis && uv run pytest horizon_position/test_positions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'positions'`.

- [ ] **Step 3: Write the implementation**

Create `horizon_position/positions.py`:

```python
"""The 19 antenna ENU positions for the position-sensitivity sweep.

Nominal position is eigsep_terrain site ``1P`` (114 m above the quarry
floor). Each of the three axes (x=East, y=North, z=Up) is perturbed one
at a time by +/-{0.1, 1, 10} m, giving 1 + 3*3*2 = 19 positions.
"""

import numpy as np

NOMINAL_ENU = np.array([1648.0, 2024.0, 1796.0])  # eigsep_terrain site 1P
SHIFTS_M = (0.1, 1.0, 10.0)
AXES = ("x", "y", "z")
_AXIS_IDX = {"x": 0, "y": 1, "z": 2}


def build_positions():
    """Return an ordered list of ``(name, enu)`` for the 19 positions.

    Order: nominal first, then for each axis (x, y, z) and magnitude
    (0.1, 1, 10) the minus shift then the plus shift. Names look like
    ``x_p_10`` (+10 m East) or ``z_m_0p1`` (-0.1 m Up).
    """
    out = [("nominal", NOMINAL_ENU.copy())]
    for axis in AXES:
        idx = _AXIS_IDX[axis]
        for mag in SHIFTS_M:
            for sign in (-1.0, +1.0):
                enu = NOMINAL_ENU.copy()
                enu[idx] += sign * mag
                sgn = "p" if sign > 0 else "m"
                mag_s = ("%g" % mag).replace(".", "p")
                out.append((f"{axis}_{sgn}_{mag_s}", enu))
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /home/christian/Documents/research/eigsep/mock_analysis && uv run pytest horizon_position/test_positions.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
cd /home/christian/Documents/research/eigsep/mock_analysis
git add horizon_position/positions.py horizon_position/test_positions.py
git commit -m "feat(horizon_position): define the 19 antenna positions"
```

---

## Task 3: `masks.py` — anti-aliased open-sky weight

**Files:**
- Create: `horizon_position/masks.py`
- Test: `horizon_position/test_masks.py`

**Context:** `α_h(φ)` is the horizon elevation (rad) vs azimuth, azimuth = `atan2(E, N)` (North→East, the `calc_horizon` convention). The beam grid (MWSS) has polar angle `θ` (0 = zenith, π = nadir) and azimuth `φ` measured from ENU East (croissant `beam_rot=0`). Open sky ⇔ elevation `> α_h` ⇔ `θ < θ_h` where `θ_h = π/2 − α_h`. Frame map between the two azimuths: `φ = π/2 − az`, so evaluate `α_h` at `az = π/2 − φ`. The weight is the fraction of each θ-cell lying above the horizon (in `[0, 1]`).

- [ ] **Step 1: Write the failing test**

Create `horizon_position/test_masks.py`:

```python
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from masks import boolean_weight, mwss_grid, open_sky_weight  # noqa: E402

LMAX = 128
N_AZ = 720


def _az_grid(n=N_AZ):
    return np.linspace(0.0, 2 * np.pi, n, endpoint=False)


def test_grid_shapes():
    thetas, phis = mwss_grid(LMAX)
    assert thetas.shape == (LMAX + 2,)   # MWSS ntheta = L + 1 = lmax + 2
    assert phis.shape == (2 * (LMAX + 1),)
    assert np.isclose(thetas[0], 0.0)
    assert np.isclose(thetas[-1], np.pi)


def test_flat_horizon_is_half_open():
    # alpha_h = 0 everywhere -> open exactly above the equator (theta < pi/2)
    thetas, phis = mwss_grid(LMAX)
    W = open_sky_weight(np.zeros(N_AZ), _az_grid(), thetas, phis)
    # rows well above the equator are fully open, well below fully blocked
    assert np.all(W[thetas < np.deg2rad(80)] > 0.99)
    assert np.all(W[thetas > np.deg2rad(100)] < 0.01)
    # solid-angle-weighted open fraction ~ 0.5
    w = np.sin(thetas)
    frac = (W * w[:, None]).sum() / (np.ones_like(W) * w[:, None]).sum()
    assert abs(frac - 0.5) < 0.01


def test_all_blocked_and_all_open():
    thetas, phis = mwss_grid(LMAX)
    W_block = open_sky_weight(np.full(N_AZ, np.pi / 2), _az_grid(), thetas, phis)
    W_open = open_sky_weight(np.full(N_AZ, -np.pi / 2), _az_grid(), thetas, phis)
    assert np.all(W_block < 1e-9)
    assert np.all(W_open > 1 - 1e-9)


def test_monotonic_in_theta():
    thetas, phis = mwss_grid(LMAX)
    rng = np.random.default_rng(0)
    W = open_sky_weight(rng.uniform(-0.3, 0.3, N_AZ), _az_grid(), thetas, phis)
    # open-sky weight must be non-increasing from zenith to nadir
    assert np.all(np.diff(W, axis=0) <= 1e-9)


def test_subpixel_shift_changes_weight():
    # a 0.05 deg change in horizon elevation must change W continuously
    # (not floored to zero) and scale with the shift size.
    thetas, phis = mwss_grid(LMAX)
    base = np.full(N_AZ, np.deg2rad(10.0))
    W0 = open_sky_weight(base, _az_grid(), thetas, phis)
    d_small = np.abs(
        open_sky_weight(base + np.deg2rad(0.05), _az_grid(), thetas, phis) - W0
    ).sum()
    d_big = np.abs(
        open_sky_weight(base + np.deg2rad(0.5), _az_grid(), thetas, phis) - W0
    ).sum()
    assert d_small > 0.0       # sub-pixel shift registers (no floor)
    assert d_small < d_big     # and scales with the shift size


def test_frame_mapping_blocks_east():
    # horizon high only near az=90deg (East) must reduce open sky near phi=0
    # (phi=0 is ENU East in croissant), not near phi=90deg.
    thetas, phis = mwss_grid(LMAX)
    az = _az_grid()
    alpha = np.deg2rad(40.0) * np.exp(-((az - np.pi / 2) ** 2) / (2 * 0.1 ** 2))
    W = open_sky_weight(alpha, az, thetas, phis)
    # near-horizon ring just above the equator
    ring = np.argmin(np.abs(thetas - np.deg2rad(85)))
    east = np.argmin(np.abs(phis - 0.0))
    north = np.argmin(np.abs(phis - np.pi / 2))
    assert W[ring, east] < W[ring, north]


def test_boolean_weight_is_zero_one():
    thetas, phis = mwss_grid(LMAX)
    rng = np.random.default_rng(1)
    B = boolean_weight(rng.uniform(-0.3, 0.3, N_AZ), _az_grid(), thetas, phis)
    assert set(np.unique(B)).issubset({0.0, 1.0})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/christian/Documents/research/eigsep/mock_analysis && uv run pytest horizon_position/test_masks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'masks'`.

- [ ] **Step 3: Write the implementation**

Create `horizon_position/masks.py`:

```python
"""Anti-aliased open-sky weight on the beam's MWSS grid.

The horizon comes from eigsep_terrain as an elevation curve alpha_h(az),
azimuth = atan2(E, N) (North->East). The beam grid is MWSS with polar
angle theta (0 = zenith, pi = nadir) and azimuth phi from ENU East
(croissant beam_rot=0). Open sky <=> elevation > alpha_h <=> theta <
theta_h, with theta_h = pi/2 - alpha_h. The frame map between the two
azimuths is phi = pi/2 - az.

The returned weight W(theta, phi) in [0, 1] is the fraction of each
theta-cell that lies above the horizon, so a sub-pixel horizon shift
changes the boundary cell continuously (no boolean-grid floor).
"""

import numpy as np
import s2fft.sampling.s2_samples as s2


def mwss_grid(lmax):
    """Return ``(thetas, phis)`` [rad] for the MWSS grid at ``lmax``."""
    L = lmax + 1
    thetas = np.asarray(s2.thetas(L, sampling="mwss"))
    phis = np.asarray(s2.phis_equiang(L, sampling="mwss"))
    return thetas, phis


def _theta_edges(thetas):
    """Cell edges: midpoints between thetas, with poles at 0 and pi."""
    mid = 0.5 * (thetas[1:] + thetas[:-1])
    return np.concatenate([[0.0], mid, [np.pi]])


def _alpha_on_phi(alpha_h, az_grid, phis):
    """Interpolate alpha_h(az) onto MWSS azimuths via phi = pi/2 - az."""
    az_of_phi = np.mod(np.pi / 2 - phis, 2 * np.pi)
    # periodic linear interpolation in azimuth
    xp = np.concatenate([az_grid, az_grid[:1] + 2 * np.pi])
    fp = np.concatenate([alpha_h, alpha_h[:1]])
    order = np.argsort(xp)
    return np.interp(az_of_phi, xp[order], fp[order])


def open_sky_weight(alpha_h, az_grid, thetas, phis):
    """Fractional open-sky weight ``W(theta, phi)`` in ``[0, 1]``.

    Parameters
    ----------
    alpha_h : (n_az,) horizon elevation [rad] vs azimuth.
    az_grid : (n_az,) azimuth [rad] of each ``alpha_h`` sample
        (= ``atan2(E, N)``).
    thetas, phis : MWSS grid axes [rad] from :func:`mwss_grid`.

    Returns
    -------
    W : (n_theta, n_phi) float, 1 = open sky, 0 = blocked.
    """
    alpha_h = np.asarray(alpha_h, dtype=np.float64)
    theta_h = np.pi / 2 - _alpha_on_phi(alpha_h, np.asarray(az_grid), phis)
    edges = _theta_edges(thetas)
    lo = edges[:-1][:, None]
    hi = edges[1:][:, None]
    # fraction of [lo, hi] with theta < theta_h (open)
    frac = (theta_h[None, :] - lo) / (hi - lo)
    return np.clip(frac, 0.0, 1.0)


def boolean_weight(alpha_h, az_grid, thetas, phis):
    """Boolean (0/1) open-sky mask via a cell-center test (cross-checks)."""
    W = open_sky_weight(alpha_h, az_grid, thetas, phis)
    return (W >= 0.5).astype(np.float64)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /home/christian/Documents/research/eigsep/mock_analysis && uv run pytest horizon_position/test_masks.py -v`
Expected: PASS (7 passed). If `test_frame_mapping_blocks_east` fails, the `φ = π/2 − az` mapping sign is wrong — try `az_of_phi = np.mod(phis - np.pi/2, 2*np.pi)` and re-run; the test encodes the physical requirement (a ridge due East blocks the East side of the sky).

- [ ] **Step 5: Commit**

```bash
cd /home/christian/Documents/research/eigsep/mock_analysis
git add horizon_position/masks.py horizon_position/test_masks.py
git commit -m "feat(horizon_position): anti-aliased open-sky weight from horizon curve"
```

---

## Task 4: `analysis.py` — ground-loss modes and summary stats

**Files:**
- Create: `horizon_position/analysis.py`
- Test: `horizon_position/test_analysis.py`

**Context:** `GLC(t_sys, fgnd) = (t_sys − T_rcvr − fgnd·T_gnd)/(1 − fgnd)`. Three modes give ΔT vs nominal: uncorrected (`t_sys`), oracle (`GLC` with each position's own `fgnd`), mis-corrected (`GLC` with the nominal `fgnd`). The mis-corrected ΔT must equal `Δt_sys/(1 − fgnd⁰)`.

- [ ] **Step 1: Write the failing test**

Create `horizon_position/test_analysis.py`:

```python
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis  # noqa: E402

T_GND, T_RCVR = 300.0, 50.0


def _fake(P=4, nt=5, nf=3, seed=0):
    rng = np.random.default_rng(seed)
    t_sys = rng.uniform(1000.0, 5000.0, (P, nt, nf))
    fgnd = rng.uniform(0.02, 0.2, (P, nf))
    return t_sys, fgnd


def test_glc_roundtrip():
    # GLC inverts t_sys = (1-f)*t_sky + f*Tgnd + Trcvr
    rng = np.random.default_rng(2)
    t_sky = rng.uniform(1000.0, 5000.0, (3, 4))
    f = rng.uniform(0.05, 0.3, (3, 4))
    t_sys = (1 - f) * t_sky + f * T_GND + T_RCVR
    out = analysis.glc(t_sys, f, T_GND, T_RCVR)
    assert np.allclose(out, t_sky)


def test_uncorrected_nominal_is_zero():
    t_sys, fgnd = _fake()
    d = analysis.delta_waterfall(t_sys, fgnd, "uncorrected", T_GND, T_RCVR)
    assert np.allclose(d[0], 0.0)
    assert d.shape == t_sys.shape


def test_miscorrected_identity():
    # mode 3 must equal uncorrected delta / (1 - fgnd_nominal)
    t_sys, fgnd = _fake()
    d_unc = analysis.delta_waterfall(t_sys, fgnd, "uncorrected", T_GND, T_RCVR)
    d_mis = analysis.delta_waterfall(t_sys, fgnd, "miscorrected", T_GND, T_RCVR)
    f0 = fgnd[0][None, None, :]
    assert np.allclose(d_mis, d_unc / (1 - f0))


def test_oracle_nominal_is_zero():
    t_sys, fgnd = _fake()
    d = analysis.delta_waterfall(t_sys, fgnd, "oracle", T_GND, T_RCVR)
    assert np.allclose(d[0], 0.0)


def test_summary_and_reductions_shapes():
    t_sys, fgnd = _fake(P=4, nt=5, nf=3)
    d = analysis.delta_waterfall(t_sys, fgnd, "uncorrected", T_GND, T_RCVR)
    s = analysis.summary_stats(d)
    assert s["rms"].shape == (4,) and s["max"].shape == (4,)
    assert analysis.rms_over_time(d).shape == (4, 3)
    assert analysis.rms_over_freq(d).shape == (4, 5)
    assert np.allclose(s["rms"][0], 0.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/christian/Documents/research/eigsep/mock_analysis && uv run pytest horizon_position/test_analysis.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'analysis'`.

- [ ] **Step 3: Write the implementation**

Create `horizon_position/analysis.py`:

```python
"""Antenna-temperature differences, ground-loss modes, summary stats.

All functions are pure NumPy. ``t_sys`` has shape ``(P, n_times,
n_freqs)`` and ``fgnd`` has shape ``(P, n_freqs)``; index 0 is the
nominal position.
"""

import numpy as np

MODES = ("uncorrected", "oracle", "miscorrected")


def glc(t_sys, fgnd, t_gnd, t_rcvr):
    """Ground-loss correction ``(t_sys - t_rcvr - fgnd*t_gnd)/(1 - fgnd)``.

    ``fgnd`` is broadcast over the time axis when it has one fewer
    dimension than ``t_sys``.
    """
    fgnd = np.asarray(fgnd)
    if fgnd.ndim == np.ndim(t_sys) - 1:
        fgnd = np.expand_dims(fgnd, axis=-2)  # insert time axis
    return (t_sys - t_rcvr - fgnd * t_gnd) / (1.0 - fgnd)


def delta_waterfall(t_sys, fgnd, mode, t_gnd, t_rcvr, nominal=0):
    """ΔT vs the nominal position for every position, for ``mode``.

    Returns an array shaped like ``t_sys`` (``(P, n_times, n_freqs)``).
    """
    if mode == "uncorrected":
        field = t_sys
    elif mode == "oracle":
        field = glc(t_sys, fgnd, t_gnd, t_rcvr)
    elif mode == "miscorrected":
        f0 = np.broadcast_to(fgnd[nominal], fgnd.shape)
        field = glc(t_sys, f0, t_gnd, t_rcvr)
    else:
        raise ValueError(f"unknown mode {mode!r}; choose one of {MODES}")
    return field - field[nominal][None]


def summary_stats(delta):
    """Per-position RMS and max|ΔT| over the (time, freq) plane."""
    return {
        "rms": np.sqrt(np.mean(delta ** 2, axis=(1, 2))),
        "max": np.max(np.abs(delta), axis=(1, 2)),
    }


def rms_over_time(delta):
    """RMS over LST -> spectrum per position, shape ``(P, n_freqs)``."""
    return np.sqrt(np.mean(delta ** 2, axis=1))


def rms_over_freq(delta):
    """RMS over frequency -> time series per position, shape ``(P, n_times)``."""
    return np.sqrt(np.mean(delta ** 2, axis=2))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /home/christian/Documents/research/eigsep/mock_analysis && uv run pytest horizon_position/test_analysis.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
cd /home/christian/Documents/research/eigsep/mock_analysis
git add horizon_position/analysis.py horizon_position/test_analysis.py
git commit -m "feat(horizon_position): ground-loss modes and summary statistics"
```

---

## Task 5: `make_horizons.py` — horizon curves per position

**Files:**
- Create: `horizon_position/make_horizons.py`

**Context:** Runs in the **eigsep_terrain** env. Builds the Marjum DEM (from the TIFs/XML already in `eigsep_terrain/data`, cached to `output/marjum_dem.npz`), then `calc_horizon` for each of the 19 positions on a uniform `n_az` azimuth grid. `calc_horizon(e, n, u, n_az)` returns `(hangles, crds)` where `hangles[j]` is the horizon elevation [rad] at azimuth `j·2π/n_az` (azimuth = `atan2(E, N)`). This is a script, not unit-tested (no eigsep_terrain in the test env); it is verified by running it.

- [ ] **Step 1: Write the script**

Create `horizon_position/make_horizons.py`:

```python
"""Compute horizon elevation profiles alpha_h(az) for the 19 positions.

Runs in the eigsep_terrain environment (it imports eigsep_terrain, which
is NOT available in the mock_analysis env):

    uv run --project /home/christian/Documents/research/eigsep/eigsep_terrain \
        python horizon_position/make_horizons.py

Output: output/horizons_position.npz with
  names     (19,)        position names
  enu       (19, 3)      antenna ENU positions [m]
  az_grid   (n_az,)      azimuths [rad], = atan2(E, N), North->East
  alpha_h   (19, n_az)   horizon elevation [rad] per position
  n_az      scalar
  pos_sha   hash of enu  (staleness guard for run_sims.py)
"""

import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
from eigsep_terrain.marjum_dem import MarjumDEM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from positions import build_positions  # noqa: E402

N_AZ = 720
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEM_CACHE = OUTPUT_DIR / "marjum_dem.npz"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("Building / loading Marjum DEM...")
    dem = MarjumDEM(cache_file=str(DEM_CACHE))

    positions = build_positions()
    names = [n for n, _ in positions]
    enu = np.array([e for _, e in positions], dtype=np.float64)
    az_grid = np.linspace(0.0, 2 * np.pi, N_AZ, endpoint=False)

    alpha_h = np.empty((len(positions), N_AZ), dtype=np.float64)
    for i, (name, e) in enumerate(positions):
        hangles, _ = dem.calc_horizon(
            float(e[0]), float(e[1]), float(e[2]), n_az=N_AZ
        )
        alpha_h[i] = np.asarray(hangles, dtype=np.float64)
        deg = np.degrees([hangles.min(), np.median(hangles), hangles.max()])
        print(f"  [{i:2d}] {name:10s} alpha_h(min,med,max) deg = "
              f"{deg[0]:6.2f} {deg[1]:6.2f} {deg[2]:6.2f}")

    pos_sha = hashlib.sha256(np.ascontiguousarray(enu).tobytes()).hexdigest()
    out = OUTPUT_DIR / "horizons_position.npz"
    np.savez(
        out,
        names=np.array(names),
        enu=enu,
        az_grid=az_grid,
        alpha_h=alpha_h,
        n_az=N_AZ,
        pos_sha=pos_sha,
    )
    print(f"Saved {out}  alpha_h shape {alpha_h.shape}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script (builds the DEM the first time — slow)**

Run:
```bash
cd /home/christian/Documents/research/eigsep/mock_analysis
uv run --project /home/christian/Documents/research/eigsep/eigsep_terrain \
    python horizon_position/make_horizons.py
```
Expected: prints 19 lines of `alpha_h(min,med,max)` and `Saved .../horizons_position.npz alpha_h shape (19, 720)`. The DEM build/cache may take a few minutes the first time.

- [ ] **Step 3: Verify the output is sane**

Run:
```bash
cd /home/christian/Documents/research/eigsep/mock_analysis
uv run python - <<'PY'
import numpy as np
d = np.load("horizon_position/output/horizons_position.npz", allow_pickle=True)
ah = d["alpha_h"]
print("shape", ah.shape, "finite", np.isfinite(ah).all())
# nominal vs +10 m East must differ; nominal vs +0.1 m must differ but less
nom, x01, x10 = ah[0], None, None
names = list(d["names"])
x01 = ah[names.index("x_p_0p1")]
x10 = ah[names.index("x_p_10")]
print("max|d alpha| 0.1m deg", np.degrees(np.abs(x01 - nom).max()))
print("max|d alpha| 10m  deg", np.degrees(np.abs(x10 - nom).max()))
assert np.isfinite(ah).all()
assert np.abs(x01 - nom).max() > 0           # 0.1 m still moves the horizon
assert np.abs(x10 - nom).max() > np.abs(x01 - nom).max()  # 10 m moves it more
print("OK")
PY
```
Expected: `finite True`, the 10 m shift moves the horizon more than the 0.1 m shift, and `OK`.

- [ ] **Step 4: Commit**

```bash
cd /home/christian/Documents/research/eigsep/mock_analysis
git add horizon_position/make_horizons.py
git commit -m "feat(horizon_position): compute horizon curves for the 19 positions"
```

---

## Task 6: `run_sims.py` — per-position waterfalls

**Files:**
- Create: `horizon_position/run_sims.py`

**Context:** Runs in the **mock_analysis** env. Mirrors `horizon_chromaticity/run_sims.py` (config, sky, time array, checkpointing) but loops over the 19 positions instead of orientations, all zenith-only. For each position it builds the anti-aliased weight `W_p` and calls `eigsim.simulate` (with `beam_kw={"horizon": W_p}`) and `eigsim.compute_fgnd`. The sky ALM is precomputed once and reused.

- [ ] **Step 1: Write the script**

Create `horizon_position/run_sims.py`:

```python
"""Run zenith-only t_sys and fgnd for each of the 19 antenna positions.

Loads the horizon curves from make_horizons.py, builds an anti-aliased
open-sky mask per position, and runs eigsim.simulate (zenith pointing,
N_ori=1) plus eigsim.compute_fgnd. Per-position batches are checkpointed
to output/pos<tag>_batch_*.npz and merged into output/position_sims.npz.

Usage (from the monorepo root):
    uv run python horizon_position/run_sims.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import croissant as cro
import numpy as np
from astropy import units as u
from astropy.time import Time
from pygdsm import GlobalSkyModel16

import eigsim

sys.path.insert(0, str(Path(__file__).resolve().parent))
from masks import mwss_grid, open_sky_weight  # noqa: E402

T_START = "2026-07-01 06:00:00"  # UTC, matches horizon_chromaticity
SIDEREAL_DAY_S = cro.constants.sidereal_day["earth"]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-times", type=int, default=1436,
                   help="time samples over one sidereal day")
    p.add_argument("--freq-stride", type=int, default=1,
                   help="use every Nth config frequency (smoke tests only)")
    p.add_argument("--output-tag", default="",
                   help="suffix for batch/output filenames (smoke tests only)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = eigsim.load_config()

    hz_file = OUTPUT_DIR / "horizons_position.npz"
    if not hz_file.exists():
        raise SystemExit(f"{hz_file} not found - run make_horizons.py first")
    hz = np.load(hz_file, allow_pickle=True)
    names = [str(n) for n in hz["names"]]
    alpha_h = hz["alpha_h"]
    az_grid = hz["az_grid"]
    enu = hz["enu"]
    pos_sha = str(hz["pos_sha"])
    n_pos = len(names)

    print("Loading beam...")
    beam_freqs_hz, beam_data, lmax = eigsim.load_beam()
    freqs_mhz = np.array(cfg["frequencies"], dtype=float)[:: args.freq_stride]
    freq_idx = np.isin(beam_freqs_hz / 1e6, freqs_mhz)
    beam_data = beam_data[freq_idx]
    n_freqs = len(freqs_mhz)
    assert beam_data.shape[0] == n_freqs
    thetas, phis = mwss_grid(lmax)

    print("Generating sky model (GSM16)...")
    sky_cfg = cfg["sky"]
    gsm = GlobalSkyModel16(
        freq_unit="MHz",
        data_unit="TRJ",
        resolution=sky_cfg["resolution"],
        include_cmb=sky_cfg["include_cmb"],
    )
    sky_map = gsm.generate(freqs_mhz)
    sky = cro.Sky(sky_map, freqs_mhz, sampling="healpix", coord="galactic")

    print("Building time array...")
    t_start = Time(T_START, scale="utc")
    t_end = t_start + SIDEREAL_DAY_S * u.s
    times = cro.utils.time_array(t_start=t_start, t_end=t_end, N_times=args.n_times)
    times_jd = times.jd

    print("Pre-computing sky ALM...")
    sky_alm = eigsim.precompute_sky_alm(sky)

    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Running {n_pos} positions x {args.n_times} times x {n_freqs} freqs...")
    wall0 = time.time()
    batch_files = []
    for i, name in enumerate(names):
        bf = OUTPUT_DIR / f"pos{args.output_tag}_batch_{i:02d}.npz"
        batch_files.append(bf)
        if bf.exists():
            npz = np.load(bf)
            if str(npz["pos_sha"]) != pos_sha:
                raise SystemExit(
                    f"{bf} was produced with different positions "
                    "(horizons_position.npz changed). Delete the stale "
                    f"pos{args.output_tag}_batch_*.npz files and rerun."
                )
            print(f"  [{i:2d}] {name:10s} found on disk, skipping")
            continue
        print(f"  [{i:2d}] {name:10s} simulating...")
        t0 = time.time()
        W = open_sky_weight(alpha_h[i], az_grid, thetas, phis)
        t_sys = eigsim.simulate(
            beam_data, freqs_mhz, sky, times_jd, [0.0], [0.0],
            beam_kw={"horizon": W}, sky_alm=sky_alm,
        )  # (1, n_times, n_freqs)
        fgnd = eigsim.compute_fgnd(
            beam_data, freqs_mhz, [0.0], [0.0], beam_kw={"horizon": W},
        )  # (1, n_freqs)
        np.savez(
            bf,
            t_sys=np.asarray(t_sys)[0],
            fgnd=np.asarray(fgnd)[0],
            pos_sha=pos_sha,
        )
        print(f"       done in {time.time() - t0:.0f}s")

    print(f"All positions complete in {(time.time() - wall0) / 60:.1f} min")

    t_sys = np.stack([np.load(f)["t_sys"] for f in batch_files], axis=0)
    fgnd = np.stack([np.load(f)["fgnd"] for f in batch_files], axis=0)
    assert t_sys.shape == (n_pos, args.n_times, n_freqs)
    assert fgnd.shape == (n_pos, n_freqs)

    out = OUTPUT_DIR / f"position_sims{args.output_tag}.npz"
    np.savez_compressed(
        out,
        t_sys=t_sys,             # (n_pos, n_times, n_freqs)
        fgnd=fgnd,               # (n_pos, n_freqs)
        names=np.array(names),
        enu=enu,
        freqs_mhz=freqs_mhz,
        times_jd=times_jd,
        t_start=T_START,
        n_times=args.n_times,
        t_ground=cfg["ground"]["temperature"],
        t_receiver=cfg["receiver"]["temperature"],
        lon=cfg["location"]["lon"],
        lat=cfg["location"]["lat"],
        alt=cfg["location"]["alt"],
        sky_model=sky_cfg["model"],
        beam_lmax=lmax,
        pos_sha=pos_sha,
        eigsim_version=eigsim.__version__,
    )
    print(f"Saved {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    for f in batch_files:
        f.unlink(missing_ok=True)
    print("Batch files cleaned up.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Quick smoke run (tiny: strided freqs, few times)**

Run:
```bash
cd /home/christian/Documents/research/eigsep/mock_analysis
uv run python horizon_position/run_sims.py --freq-stride 40 --n-times 8 --output-tag _smoke
```
Expected: 19 positions simulate without error; prints `Saved .../position_sims_smoke.npz`. Then clean it up:
```bash
rm horizon_position/output/position_sims_smoke.npz
```

- [ ] **Step 3: Commit**

```bash
cd /home/christian/Documents/research/eigsep/mock_analysis
git add horizon_position/run_sims.py
git commit -m "feat(horizon_position): per-position zenith-only t_sys and fgnd"
```

---

## Task 7: Smoke test (EIGSEP_SMOKE-gated)

**Files:**
- Create: `horizon_position/test_smoke.py`

**Context:** Mirrors `horizon_chromaticity/test_smoke.py`: gated behind `EIGSEP_SMOKE=1` so plain `pytest` skips it. It assumes `output/horizons_position.npz` already exists (from Task 5) and runs `run_sims.py` as a subprocess on a tiny grid, then checks the output. Does not run `make_horizons.py` (needs the other env).

- [ ] **Step 1: Write the smoke test**

Create `horizon_position/test_smoke.py`:

```python
"""End-to-end smoke test for run_sims.py (gated behind EIGSEP_SMOKE=1).

Requires output/horizons_position.npz to exist (run make_horizons.py in
the eigsep_terrain env first). Run with:

    EIGSEP_SMOKE=1 uv run pytest horizon_position/test_smoke.py -v
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"

pytestmark = pytest.mark.skipif(
    os.environ.get("EIGSEP_SMOKE") != "1",
    reason="set EIGSEP_SMOKE=1 to run (spawns subprocess, compiles JAX)",
)


def test_run_sims_smoke():
    if not (OUT / "horizons_position.npz").exists():
        pytest.skip("run make_horizons.py (eigsep_terrain env) first")
    tag = "_pytest"
    cmd = [
        sys.executable, str(HERE / "run_sims.py"),
        "--freq-stride", "40", "--n-times", "6", "--output-tag", tag,
    ]
    subprocess.run(cmd, check=True, cwd=HERE.parent)
    out = OUT / f"position_sims{tag}.npz"
    try:
        d = np.load(out, allow_pickle=True)
        assert d["t_sys"].shape[0] == 19
        assert d["fgnd"].shape[0] == 19
        assert np.isfinite(d["t_sys"]).all()
        # nominal (index 0) and +10 m East must differ; t_sys positive
        names = [str(n) for n in d["names"]]
        i10 = names.index("x_p_10")
        assert d["t_sys"].min() > 0
        assert not np.allclose(d["t_sys"][0], d["t_sys"][i10])
    finally:
        out.unlink(missing_ok=True)
        for b in OUT.glob(f"pos{tag}_batch_*.npz"):
            b.unlink()
```

- [ ] **Step 2: Run the smoke test**

Run:
```bash
cd /home/christian/Documents/research/eigsep/mock_analysis
EIGSEP_SMOKE=1 uv run pytest horizon_position/test_smoke.py -v
```
Expected: PASS (or `SKIP` if `horizons_position.npz` is absent — in that case run Task 5 first).

- [ ] **Step 3: Confirm plain pytest skips it**

Run: `cd /home/christian/Documents/research/eigsep/mock_analysis && uv run pytest horizon_position/test_smoke.py -v`
Expected: 1 skipped.

- [ ] **Step 4: Commit**

```bash
cd /home/christian/Documents/research/eigsep/mock_analysis
git add horizon_position/test_smoke.py
git commit -m "test(horizon_position): EIGSEP_SMOKE end-to-end run_sims test"
```

---

## Task 8: Validation — frame mapping and anti-aliasing cross-checks

**Files:**
- Create: `horizon_position/test_validation.py`

**Context:** Two checks from the spec. (a) **Frame mapping:** the nominal `α_h(φ)` mapped to the beam grid must reproduce the existing nominal `horizon_mwss.npz` (same site, croissant frame). (b) **Anti-aliasing at 10 m:** the fractional weight and a boolean mask must give the same ground fraction for a 10 m shift (large enough that booleanizing is fine), confirming anti-aliasing does not bias large shifts. Both are gated behind `EIGSEP_SMOKE` only because (a) needs `horizons_position.npz`; (b) is pure and always runs.

- [ ] **Step 1: Write the validation tests**

Create `horizon_position/test_validation.py`:

```python
"""Frame-mapping and anti-aliasing validation (spec "Validation")."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from masks import boolean_weight, mwss_grid, open_sky_weight  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
LMAX = 128


def _open_fraction(W, thetas):
    w = np.sin(thetas)[:, None]
    return (W * w).sum() / (np.ones_like(W) * w).sum()


def test_antialias_matches_boolean_open_fraction():
    # For a generic horizon the solid-angle open fraction from the
    # fractional and boolean masks must agree to ~grid resolution.
    thetas, phis = mwss_grid(LMAX)
    az = np.linspace(0.0, 2 * np.pi, 720, endpoint=False)
    rng = np.random.default_rng(3)
    alpha = np.deg2rad(rng.uniform(-5, 15, 720))
    Wf = open_sky_weight(alpha, az, thetas, phis)
    Wb = boolean_weight(alpha, az, thetas, phis)
    assert abs(_open_fraction(Wf, thetas) - _open_fraction(Wb, thetas)) < 0.01


@pytest.mark.skipif(
    not (OUT / "horizons_position.npz").exists(),
    reason="run make_horizons.py (eigsep_terrain env) first",
)
def test_nominal_frame_matches_existing_horizon_mwss():
    import eigsim

    horizon_raw, lmax = eigsim.load_horizon()  # nominal, croissant frame
    existing_open = np.isnan(horizon_raw)      # NaN = open sky in the raw file
    thetas, phis = mwss_grid(lmax)

    hz = np.load(OUT / "horizons_position.npz", allow_pickle=True)
    names = [str(n) for n in hz["names"]]
    nom = hz["alpha_h"][names.index("nominal")]
    W = open_sky_weight(nom, hz["az_grid"], thetas, phis)
    our_open = W >= 0.5

    # The two nominal masks come from different methods (calc_horizon vs
    # ray-trace+nearest-neighbour) but the same site: they must agree on
    # the large majority of cells. A gross azimuth-frame error would push
    # agreement toward 0.5.
    agree = (our_open == existing_open).mean()
    # A gross azimuth-frame error pushes agreement toward 0.5; method
    # differences near the jagged horizon cost a few percent at most.
    assert agree > 0.85, f"nominal masks agree only {agree:.2f} (frame error?)"
```

- [ ] **Step 2: Run the pure check (always) and the framed check (if data present)**

Run:
```bash
cd /home/christian/Documents/research/eigsep/mock_analysis
uv run pytest horizon_position/test_validation.py -v
```
Expected: `test_antialias_matches_boolean_open_fraction` PASSES; `test_nominal_frame_matches_existing_horizon_mwss` PASSES if `horizons_position.npz` exists (else SKIP). If the frame test fails with agreement near 0.5, fix the `φ = π/2 − az` mapping in `masks.py` (Task 3, Step 4 note) and rerun Tasks 3 and 8.

- [ ] **Step 3: Commit**

```bash
cd /home/christian/Documents/research/eigsep/mock_analysis
git add horizon_position/test_validation.py
git commit -m "test(horizon_position): frame-mapping and anti-aliasing validation"
```

---

## Task 9: Full production runs

**Files:**
- Uses: `horizon_position/make_horizons.py`, `horizon_position/run_sims.py`

**Context:** Generate the real data products. `make_horizons.py` in the eigsep_terrain env, `run_sims.py` in the mock_analysis env. This is a run step, not code; expected wall time tens of minutes.

- [ ] **Step 1: Generate the horizon curves (if not already done in Task 5)**

Run:
```bash
cd /home/christian/Documents/research/eigsep/mock_analysis
uv run --project /home/christian/Documents/research/eigsep/eigsep_terrain \
    python horizon_position/make_horizons.py
```
Expected: `output/horizons_position.npz` with `alpha_h` shape `(19, 720)`.

- [ ] **Step 2: Run the full per-position simulation**

Run:
```bash
cd /home/christian/Documents/research/eigsep/mock_analysis
uv run python horizon_position/run_sims.py
```
Expected: 19 positions complete (resumable via checkpoints); `Saved .../position_sims.npz`. Tens of minutes.

- [ ] **Step 3: Sanity-check the products**

Run:
```bash
cd /home/christian/Documents/research/eigsep/mock_analysis
uv run python - <<'PY'
import sys; sys.path.insert(0, "horizon_position")
import numpy as np, analysis
d = np.load("horizon_position/output/position_sims.npz", allow_pickle=True)
t_sys, fgnd = d["t_sys"], d["fgnd"]
names = [str(n) for n in d["names"]]
tg, tr = float(d["t_ground"]), float(d["t_receiver"])
dl = analysis.delta_waterfall(t_sys, fgnd, "uncorrected", tg, tr)
s = analysis.summary_stats(dl)
order = lambda nm: names.index(nm)
print("S(nominal) =", s["rms"][0], "(must be 0)")
for nm in ["x_p_0p1", "x_p_1", "x_p_10", "z_p_0p1", "z_p_1", "z_p_10"]:
    print(f"  S({nm:8s}) = {s['rms'][order(nm)]:.4g} K   max = {s['max'][order(nm)]:.4g} K")
# convergence: 0.1 m < 1 m < 10 m (same axis/sign)
for ax in "xyz":
    vals = [s["rms"][order(f"{ax}_p_{m}")] for m in ["0p1","1","10"]]
    print(ax, "rms 0.1/1/10:", [f"{v:.3g}" for v in vals])
    assert vals[0] < vals[1] < vals[2], f"non-monotonic on {ax}"
assert np.isclose(s["rms"][0], 0.0)
print("OK")
PY
```
Expected: `S(nominal) = 0`, RMS increases with shift magnitude on every axis, and `OK`. (Per the spec's physical expectations, z RMS should be flatter in LST and x/y more LST-modulated — examined in the notebook.)

---

## Task 10: Analysis notebook

**Files:**
- Create: `horizon_position/notebooks/horizon_position.ipynb`

**Context:** Loads `output/position_sims.npz` and `analysis.py`, computes the three modes, and produces the spec deliverables: the scalar `S` vs shift-magnitude curve, the resolved reductions, and representative waterfalls. Build the notebook by writing a script and converting it, so the steps are reproducible.

- [ ] **Step 1: Write the notebook source script**

Create `horizon_position/notebooks/_build_notebook.py`:

```python
"""Generate horizon_position.ipynb from a list of cells."""

from pathlib import Path

import nbformat as nbf

cells_src = [
    # 0: imports + load
    '''import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.cwd().parent))
import analysis

d = np.load("../output/position_sims.npz", allow_pickle=True)
t_sys, fgnd = d["t_sys"], d["fgnd"]
names = [str(n) for n in d["names"]]
freqs = d["freqs_mhz"]
times_jd = d["times_jd"]
lst_hours = (times_jd - times_jd[0]) * 24.0
T_GND, T_RCVR = float(d["t_ground"]), float(d["t_receiver"])
print(t_sys.shape, fgnd.shape, "fgnd_nominal mean", fgnd[0].mean())''',

    # 1: choose mode + compute delta + summary
    '''MODE = "uncorrected"   # "uncorrected" | "oracle" | "miscorrected"
delta = analysis.delta_waterfall(t_sys, fgnd, MODE, T_GND, T_RCVR)
stats = analysis.summary_stats(delta)
for nm, rms, mx in zip(names, stats["rms"], stats["max"]):
    print(f"{nm:10s}  RMS {rms:9.4g} K   max {mx:9.4g} K")''',

    # 2: spec curve S vs shift magnitude, per axis/sign/mode
    '''def s_for(axis, sign, mode):
    dl = analysis.delta_waterfall(t_sys, fgnd, mode, T_GND, T_RCVR)
    st = analysis.summary_stats(dl)
    mags = [0.1, 1.0, 10.0]
    tags = {0.1: "0p1", 1.0: "1", 10.0: "10"}
    return mags, [st["rms"][names.index(f"{axis}_{sign}_{tags[m]}")] for m in mags]

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax_i, mode in zip(axes, analysis.MODES):
    for axis in "xyz":
        for sign, ls in [("p", "-"), ("m", "--")]:
            mags, s = s_for(axis, sign, mode)
            ax_i.loglog(mags, s, ls, marker="o", label=f"{axis}{sign}")
    ax_i.set_title(mode); ax_i.set_xlabel("shift [m]"); ax_i.grid(True, which="both", alpha=.3)
axes[0].set_ylabel("RMS |dT| over (LST, freq) [K]")
axes[0].legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig("fig_spec_curve.pdf"); fig.show()''',

    # 3: resolved reductions for one shift -- spectrum and LST series
    '''SHIFT = "10"   # magnitude tag
fig, (axf, axt) = plt.subplots(1, 2, figsize=(12, 4))
for axis, c in zip("xyz", ["C0", "C1", "C2"]):
    i = names.index(f"{axis}_p_{SHIFT}")
    axf.plot(freqs, analysis.rms_over_time(delta)[i], c, label=f"{axis}+")
    axt.plot(lst_hours, analysis.rms_over_freq(delta)[i], c, label=f"{axis}+")
axf.set_xlabel("freq [MHz]"); axf.set_ylabel("RMS_LST |dT| [K]"); axf.set_title(f"spectrum, {SHIFT} m")
axt.set_xlabel("LST [h]"); axt.set_ylabel("RMS_freq |dT| [K]"); axt.set_title(f"LST series, {SHIFT} m")
for a in (axf, axt): a.legend(); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig("fig_reductions.pdf"); fig.show()''',

    # 4: representative waterfall
    '''i = names.index(f"x_p_{SHIFT}")
fig, ax = plt.subplots(figsize=(7, 4))
im = ax.pcolormesh(freqs, lst_hours, delta[i], shading="auto", cmap="RdBu_r")
ax.set_xlabel("freq [MHz]"); ax.set_ylabel("LST [h]")
ax.set_title(f"dT waterfall: {names[i]} ({MODE})")
fig.colorbar(im, label="dT [K]"); fig.tight_layout(); fig.savefig("fig_waterfall.pdf"); fig.show()''',
]

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_code_cell(s) for s in cells_src]
out = Path(__file__).resolve().parent / "horizon_position.ipynb"
nbf.write(nb, out)
print("wrote", out)
```

- [ ] **Step 2: Build and execute the notebook**

Run:
```bash
cd /home/christian/Documents/research/eigsep/mock_analysis/horizon_position/notebooks
uv run python _build_notebook.py
uv run jupyter nbconvert --to notebook --execute --inplace horizon_position.ipynb
```
Expected: `wrote .../horizon_position.ipynb`, then nbconvert completes without error and writes `fig_spec_curve.pdf`, `fig_reductions.pdf`, `fig_waterfall.pdf`.

- [ ] **Step 3: Eyeball the physical predictions**

Open `fig_spec_curve.pdf` and `fig_reductions.pdf`. Confirm (spec "Physical expectations"): `S` rises ~linearly with shift on log–log for small shifts; the LST series (`fig_reductions`) is more structured for x/y than for z, and the z spectrum looks more like a broadband offset. Note any deviation in the notebook's final markdown cell (add one if useful).

- [ ] **Step 4: Commit**

```bash
cd /home/christian/Documents/research/eigsep/mock_analysis
git add horizon_position/notebooks/_build_notebook.py horizon_position/notebooks/horizon_position.ipynb
git commit -m "feat(horizon_position): analysis notebook (modes, spec curve, reductions)"
```

---

## Task 11: Documentation (README + CLAUDE.md)

**Files:**
- Create: `horizon_position/README.md`
- Create: `horizon_position/CLAUDE.md`

- [ ] **Step 1: Write the README**

Create `horizon_position/README.md`:

```markdown
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
`eigsim`. They are separate uv envs.

```bash
# 1. horizon curves (eigsep_terrain env)
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
```

- [ ] **Step 2: Write the CLAUDE.md**

Create `horizon_position/CLAUDE.md`:

```markdown
# CLAUDE.md

Guidance for Claude Code in this directory.

## What this is

Self-contained analysis (for a paper): how antenna-position error
changes the EIGSEP antenna temperature vs freq/LST, as a spec on
position knowledge. Uses `eigsim` and `eigsep_terrain`; not a package.
Sibling of `horizon_chromaticity/`. Zenith pointing only.

Spec:  `../docs/superpowers/specs/2026-06-13-horizon-position-sensitivity-design.md`
Plan:  `../docs/superpowers/plans/2026-06-13-horizon-position-sensitivity.md`

## Two environments (important)

- `make_horizons.py` imports `eigsep_terrain` (NOT in the mock_analysis
  env). Run it with `uv run --project <eigsep_terrain path> python ...`.
- `run_sims.py`, the pure modules, and the tests use `eigsim`/`s2fft`
  in the default env: `uv run python ...` / `uv run pytest ...`.

## Method / critical conventions

- The horizon is a **continuous** elevation curve `alpha_h(az)` from
  `calc_horizon` (azimuth = `atan2(E, N)`, North->East). It is turned
  into an **anti-aliased (fractional)** open-sky mask `W in [0,1]` on the
  MWSS beam grid (`masks.open_sky_weight`). Fractional weighting is what
  lets sub-pixel (0.1 m) horizon shifts register — a boolean mask floors
  them to zero.
- **Frame map:** croissant beam/grid azimuth `phi` is from ENU East;
  `calc_horizon` azimuth is from North. They are related by
  `phi = pi/2 - az`. Verified against the nominal `horizon_mwss.npz` in
  `test_validation.py`.
- `eigsim.simulate`/`compute_fgnd` apply the horizon as a float multiply
  (croissant stores it as-is, no booleanization) and normalize by the
  full-sphere beam integral, so a fractional `W` flows through correctly.
- Set `os.environ.setdefault("JAX_ENABLE_X64", "1")` before any
  jax/s2fft/croissant/eigsim/eigsep_terrain import.
- `output/` is gitignored. The notebook loads npz + `analysis.py` only;
  it never imports the scripts.

## Files

- `positions.py` / `masks.py` / `analysis.py` — pure, unit-tested.
- `make_horizons.py` -> `output/horizons_position.npz` (eigsep_terrain env).
- `run_sims.py` -> `output/position_sims.npz` (eigsim env; resumable
  per-position checkpoints `pos*_batch_*.npz`; `pos_sha` guards against a
  stale `horizons_position.npz`).
- `notebooks/horizon_position.ipynb` — modes, spec curve, reductions.
```

- [ ] **Step 3: Verify the full unit-test suite passes**

Run: `cd /home/christian/Documents/research/eigsep/mock_analysis && uv run pytest horizon_position/ -v`
Expected: all pure-module tests pass; smoke test skipped (no `EIGSEP_SMOKE`).

- [ ] **Step 4: Commit**

```bash
cd /home/christian/Documents/research/eigsep/mock_analysis
git add horizon_position/README.md horizon_position/CLAUDE.md
git commit -m "docs(horizon_position): README and CLAUDE.md"
```

---

## Self-review notes (for the implementer)

- **Frame mapping is the main risk.** If `test_validation.py`'s nominal-frame
  check shows ~50% agreement, the `φ = π/2 − az` relation in
  `masks.py::_alpha_on_phi` is wrong (sign/offset). The test is decisive —
  fix `masks.py` and rerun Tasks 3 and 8 before the production run.
- **0.1 m must not floor to zero.** Task 5 Step 3 and Task 9 Step 3 assert
  the 0.1 m horizon/T differences are non-zero and monotonic in magnitude.
  If 0.1 m gives exactly zero ΔT, anti-aliasing is being lost — check that
  `W` is float (not boolean) all the way into `eigsim.simulate`, and that
  `cro.Beam` is not booleanizing (it stores `jnp.asarray(horizon)`).
- **Edge aliasing knob.** If the 10 m anti-aliased-vs-boolean check or the
  monotonicity check looks noisy, resample the beam to a finer `L_work` in
  `run_sims.py` (inverse SHT of the beam ALM onto a finer MWSS grid) and
  build masks at that grid. Default keeps the native `lmax=128` grid.
- **Compile overhead.** `eigsim.simulate` re-JITs per position (19 compiles);
  expected, and checkpoints make reruns cheap.
```
