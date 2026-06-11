# Horizon Chromaticity Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained `horizon_chromaticity/` analysis project that simulates noiseless EIGSEP spectra under three horizon models (no horizon, constant-θ "quarry" cut, realistic EIGSEP horizon) and saves npz outputs for notebook analysis; also fix the NaN-horizon bug in the eigsim canonical sim script.

**Architecture:** Two scripts in a new top-level monorepo folder: `make_horizons.py` builds three boolean masks (True = open sky) on the MWSS grid and solves the quarry cut angle by matching blocked solid angle to the EIGSEP horizon; `run_sims.py --case <name>` runs `eigsim.simulate()` over the full canonical orientation grid with batch checkpoint/resume and merges to `output/chromaticity_<case>.npz`. Smoke tests (env-var gated so they don't slow normal `pytest`) exercise both scripts on tiny inputs.

**Tech Stack:** Python via `uv run`, eigsim/croissant-sim/s2fft/JAX, pygdsm (GSM16), numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-06-10-horizon-chromaticity-design.md`

**Key background for the engineer:**

- `croissant.Beam(horizon=...)` expects a **boolean** mask, True = above horizon (open sky). The file `eigsim/data/horizon_mwss.npz` stores distance-to-terrain: **finite = blocked, NaN = open sky**. Convert with `np.isnan(raw)`. Passing the raw array makes `eigsim.simulate()` return all-NaN (verified).
- MWSS sampling with `lmax=128`: `L = lmax + 1 = 129`, grid shape `(N_theta, N_phi) = (L + 1, 2L) = (130, 258)`, ring colatitudes `theta = np.linspace(0, pi, L + 1)`.
- s2fft MWSS quadrature weights `s2fft.utils.quadrature_jax.quad_weights(L=L, sampling="mwss")` have shape `(L + 1,)`; the per-**pixel** solid-angle weight is `w[ring]`, so total sphere = `nphi * w.sum() = 4π`. This is the same weighting `eigsim.simulate()` uses internally.
- Always run Python via `uv run` from the monorepo root `/home/christian/Documents/research/eigsep/mock_analysis`.
- Scripts must set `os.environ.setdefault("JAX_ENABLE_X64", "1")` **before** any jax/s2fft/croissant/eigsim import.

---

### Task 1: Fix NaN-horizon bug in eigsim canonical sim script

The canonical sim script passes the raw distance/NaN horizon array to `cro.Beam(horizon=...)`, producing 100% NaN output (`eigsim/output/batch_0000.npz` is all NaN — verified).

**Files:**
- Modify: `eigsim/scripts/run_canonical_sim.py:46`
- Delete: `eigsim/output/batch_0000.npz` (stale, all-NaN)

- [ ] **Step 1: Apply the fix**

In `eigsim/scripts/run_canonical_sim.py`, change line 46:

```python
horizon, _ = eigsim.load_horizon()
```

to:

```python
horizon, _ = eigsim.load_horizon()
# File stores distance-to-terrain (finite = blocked, NaN = open sky);
# croissant.Beam needs a boolean mask with True = above horizon.
horizon = np.isnan(horizon)
```

- [ ] **Step 2: Verify the fix produces finite output**

Write `/tmp/verify_horizon_fix.py`:

```python
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import croissant as cro
import numpy as np

import eigsim

beam_freqs_hz, beam_data, lmax = eigsim.load_beam()
horizon, _ = eigsim.load_horizon()
horizon = np.isnan(horizon)

freqs = beam_freqs_hz[:2] / 1e6
sky_map = np.full((2, 12 * 16**2), 1000.0)
sky = cro.Sky(sky_map, freqs, sampling="healpix", coord="galactic")
t = eigsim.simulate(
    beam_data[:2], freqs, sky, np.array([2460857.75]),
    [0.0], [0.0], beam_kw={"horizon": horizon},
)
t = np.asarray(t)
print("t_sys:", t.ravel())
assert np.all(np.isfinite(t)), "still NaN!"
assert np.all(t > 0)
print("OK: finite, positive")
```

Run: `cd /home/christian/Documents/research/eigsep/mock_analysis && uv run python /tmp/verify_horizon_fix.py`
Expected: prints two finite positive temperatures and `OK: finite, positive`

- [ ] **Step 3: Delete the stale NaN output**

```bash
rm /home/christian/Documents/research/eigsep/mock_analysis/eigsim/output/batch_0000.npz
```

(`output/` is gitignored, so this is not a git operation.)

- [ ] **Step 4: Run eigsim tests to confirm nothing broke**

Run: `cd /home/christian/Documents/research/eigsep/mock_analysis && uv run pytest eigsim/tests -q`
Expected: all tests PASS (script change only; tests should be unaffected)

- [ ] **Step 5: Commit**

```bash
cd /home/christian/Documents/research/eigsep/mock_analysis
git add eigsim/scripts/run_canonical_sim.py
git commit -m "fix(eigsim): convert horizon to boolean mask in canonical sim script

The horizon npz stores distance-to-terrain (finite = blocked,
NaN = open sky), but croissant.Beam expects a boolean mask with
True = above horizon. Passing the raw array poisoned the whole
simulation with NaN."
```

---

### Task 2: Scaffold `horizon_chromaticity/` project folder

**Files:**
- Create: `horizon_chromaticity/README.md`
- Create: `horizon_chromaticity/.gitignore`
- Create: `horizon_chromaticity/notebooks/.gitkeep`

- [ ] **Step 1: Create the folder and README**

Create `horizon_chromaticity/README.md`:

```markdown
# Horizon Chromaticity Comparison

Analysis project (for a paper) comparing the spectral chromaticity of
simulated EIGSEP spectra under three horizon models. Uses the
`eigsim` package; does not ship with it.

Design spec: `../docs/superpowers/specs/2026-06-10-horizon-chromaticity-design.md`

## Horizon cases

All cases are boolean masks on the MWSS grid, True = open sky:

| Case        | Description                                                        |
|-------------|--------------------------------------------------------------------|
| `nohorizon` | θ = π cut: everything open, ground fraction 0                     |
| `quarry`    | Constant-θ cut; θ_c solved to match the blocked solid angle of the EIGSEP horizon (nearest MWSS ring boundary) |
| `eigsep`    | Realistic horizon: `np.isnan(horizon)` of `eigsim/data/horizon_mwss.npz` |

## Workflow

From the monorepo root:

```bash
# 1. Build the three horizon masks -> output/horizons.npz
uv run python horizon_chromaticity/make_horizons.py

# 2. Run the simulation for each case (hours each; checkpoint/resume
#    via output/<case>_batch_*.npz, safe to interrupt and rerun)
uv run python horizon_chromaticity/run_sims.py --case nohorizon
uv run python horizon_chromaticity/run_sims.py --case quarry
uv run python horizon_chromaticity/run_sims.py --case eigsep

# 3. Analyze in notebooks/ (chromaticity metrics live there)
```

## Simulation parameters

Full canonical grid (36 elevations x 36 azimuths = 1296 orientations),
one sidereal day (1436 times), 201 channels (50-250 MHz), GSM16 sky,
**noiseless** (raw t_sys; add noise in notebooks if needed).

## Outputs (`output/`, gitignored)

- `horizons.npz` — the three masks + quarry θ_c + blocked solid angles
- `chromaticity_<case>.npz` — `t_sys (N_ori, N_times, N_freqs)`, axes
  (`freqs_mhz`, `times_jd`, flat per-orientation `elevations`/`azimuths`
  plus grid axes `elev_vals`/`az_vals`), horizon + config metadata

## Smoke tests

Gated behind an env var so plain `uv run pytest` stays fast:

```bash
EIGSEP_SMOKE=1 uv run pytest horizon_chromaticity/test_smoke.py -v
```

Takes a few minutes (JAX compilation). Requires the eigsim data files.
```

- [ ] **Step 2: Create `.gitignore` and notebooks dir**

Create `horizon_chromaticity/.gitignore`:

```
output/
```

Create empty file `horizon_chromaticity/notebooks/.gitkeep`.

- [ ] **Step 3: Commit**

```bash
cd /home/christian/Documents/research/eigsep/mock_analysis
git add horizon_chromaticity/README.md horizon_chromaticity/.gitignore horizon_chromaticity/notebooks/.gitkeep
git commit -m "chore(horizon_chromaticity): scaffold analysis project folder"
```

---

### Task 3: `make_horizons.py` + mask tests

**Files:**
- Create: `horizon_chromaticity/make_horizons.py`
- Create: `horizon_chromaticity/test_smoke.py` (mask tests; sim smoke test added in Task 4)

- [ ] **Step 1: Write the failing tests**

Create `horizon_chromaticity/test_smoke.py`:

```python
"""Smoke tests for the horizon chromaticity project.

These run the actual scripts on tiny inputs. They need the eigsim
data files and take a few minutes (JAX compilation), so they are
gated behind an env var:

    EIGSEP_SMOKE=1 uv run pytest horizon_chromaticity/test_smoke.py -v
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("EIGSEP_SMOKE") != "1",
    reason="set EIGSEP_SMOKE=1 to run smoke tests",
)

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
CASES = ("nohorizon", "quarry", "eigsep")


@pytest.fixture(scope="module")
def horizons():
    subprocess.run(
        [sys.executable, str(PROJECT_DIR / "make_horizons.py")], check=True
    )
    return np.load(OUTPUT_DIR / "horizons.npz")


def _quad_weights(lmax):
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import s2fft

    return np.asarray(
        s2fft.utils.quadrature_jax.quad_weights(L=lmax + 1, sampling="mwss")
    )


def test_mask_shapes_and_dtypes(horizons):
    lmax = int(horizons["lmax"])
    shape = (lmax + 2, 2 * (lmax + 1))  # (L + 1, 2L) with L = lmax + 1
    for case in CASES:
        assert horizons[case].shape == shape
        assert horizons[case].dtype == np.bool_


def test_nohorizon_all_open(horizons):
    assert horizons["nohorizon"].all()


def test_eigsep_mask_matches_file(horizons):
    import eigsim

    raw, _ = eigsim.load_horizon()
    assert np.array_equal(horizons["eigsep"], np.isnan(raw))


def test_quarry_is_ring_cut(horizons):
    mask = horizons["quarry"]
    i_cut = int(horizons["i_cut"])
    # whole rings: open above the cut, blocked below, nothing partial
    assert mask[:i_cut].all()
    assert not mask[i_cut:].any()


def test_quarry_solid_angle_matches_eigsep(horizons):
    lmax = int(horizons["lmax"])
    w = _quad_weights(lmax)
    nphi = horizons["quarry"].shape[1]

    omega_quarry = (w[:, None] * ~horizons["quarry"]).sum()
    omega_eigsep = (w[:, None] * ~horizons["eigsep"]).sum()
    assert omega_eigsep == pytest.approx(
        float(horizons["omega_blocked_target"])
    )
    # the ring cut can't do better than one ring of solid angle
    max_ring_omega = (nphi * w).max()
    assert abs(omega_quarry - omega_eigsep) <= max_ring_omega
    # and both block a substantial chunk of the sphere (~half)
    assert 0.3 * 4 * np.pi < omega_eigsep < 0.7 * 4 * np.pi


def test_fgnd_per_case(horizons):
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import croissant as cro

    import eigsim

    beam_freqs_hz, beam_data, _ = eigsim.load_beam()
    freqs = beam_freqs_hz[:1] / 1e6
    fgnd = {}
    for case in CASES:
        beam = cro.Beam(
            beam_data[:1],
            freqs,
            sampling="mwss",
            niter=0,
            horizon=horizons[case],
        )
        fgnd[case] = float(np.asarray(beam.compute_fgnd())[0])
    assert fgnd["nohorizon"] == pytest.approx(0.0, abs=1e-12)
    assert fgnd["quarry"] > 0.01
    assert fgnd["eigsep"] > 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/christian/Documents/research/eigsep/mock_analysis && EIGSEP_SMOKE=1 uv run pytest horizon_chromaticity/test_smoke.py -v`
Expected: ERROR in the `horizons` fixture — `make_horizons.py` does not exist (subprocess `CalledProcessError` / FileNotFoundError)

- [ ] **Step 3: Write `make_horizons.py`**

Create `horizon_chromaticity/make_horizons.py`:

```python
"""Build the three horizon masks for the chromaticity comparison.

Cases (boolean masks on the MWSS grid, True = open sky):

- ``nohorizon``: theta = pi cut, all pixels open.
- ``quarry``: constant-theta cut at the MWSS ring boundary whose
  blocked solid angle best matches the EIGSEP horizon.
- ``eigsep``: realistic horizon from ``horizon_mwss.npz`` (the file
  stores distance-to-terrain; finite = blocked, NaN = open sky).

Saves ``output/horizons.npz`` and prints a summary.

Usage
-----
uv run python horizon_chromaticity/make_horizons.py
"""

import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import s2fft

import eigsim

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main():
    horizon_raw, lmax = eigsim.load_horizon()
    ntheta, nphi = horizon_raw.shape
    L = lmax + 1
    assert (ntheta, nphi) == (L + 1, 2 * L), "not MWSS sampling"

    # Per-ring quadrature weights; the per-pixel weight is w[ring].
    # Same weighting eigsim.simulate() uses internally.
    w = np.asarray(
        s2fft.utils.quadrature_jax.quad_weights(L=L, sampling="mwss")
    )
    theta = np.linspace(0.0, np.pi, ntheta)

    mask_eigsep = np.isnan(horizon_raw)  # True = open sky
    omega_target = (w[:, None] * ~mask_eigsep).sum()

    # Quarry: block whole rings i >= i_cut; pick the cut minimizing
    # |blocked solid angle - target|.
    ring_omega = nphi * w
    blocked = np.array(
        [ring_omega[i:].sum() for i in range(ntheta + 1)]
    )
    i_cut = int(np.argmin(np.abs(blocked - omega_target)))
    mask_quarry = np.zeros((ntheta, nphi), dtype=bool)
    mask_quarry[:i_cut] = True
    theta_c = theta[i_cut - 1] if i_cut > 0 else np.nan  # last open ring

    mask_nohorizon = np.ones((ntheta, nphi), dtype=bool)

    omega_blocked = {
        case: (w[:, None] * ~mask).sum()
        for case, mask in [
            ("nohorizon", mask_nohorizon),
            ("quarry", mask_quarry),
            ("eigsep", mask_eigsep),
        ]
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / "horizons.npz"
    np.savez(
        out,
        nohorizon=mask_nohorizon,
        quarry=mask_quarry,
        eigsep=mask_eigsep,
        lmax=lmax,
        i_cut=i_cut,
        theta_c_rad=theta_c,
        theta_c_deg=np.degrees(theta_c),
        omega_blocked_target=omega_target,
        omega_blocked_nohorizon=omega_blocked["nohorizon"],
        omega_blocked_quarry=omega_blocked["quarry"],
        omega_blocked_eigsep=omega_blocked["eigsep"],
    )

    print(f"Saved {out}")
    print(
        f"  quarry cut: ring {i_cut}, "
        f"theta_c = {np.degrees(theta_c):.2f} deg"
    )
    for case in ("nohorizon", "quarry", "eigsep"):
        om = omega_blocked[case]
        print(
            f"  {case:10s} blocked solid angle = {om:7.4f} sr "
            f"({om / (4 * np.pi):5.1%} of sphere)"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/christian/Documents/research/eigsep/mock_analysis && EIGSEP_SMOKE=1 uv run pytest horizon_chromaticity/test_smoke.py -v`
Expected: all 6 tests PASS. Note the printed quarry θ_c (should be a bit below 90° since the EIGSEP horizon blocks the lower hemisphere plus terrain above it).

- [ ] **Step 5: Sanity-check that plain pytest skips smoke tests**

Run: `cd /home/christian/Documents/research/eigsep/mock_analysis && uv run pytest horizon_chromaticity/test_smoke.py -q`
Expected: all tests SKIPPED ("set EIGSEP_SMOKE=1 to run smoke tests")

- [ ] **Step 6: Lint and commit**

```bash
cd /home/christian/Documents/research/eigsep/mock_analysis
uv run ruff check horizon_chromaticity/ && uv run ruff format horizon_chromaticity/
git add horizon_chromaticity/make_horizons.py horizon_chromaticity/test_smoke.py
git commit -m "feat(horizon_chromaticity): build horizon masks with solid-angle-matched quarry cut"
```

---

### Task 4: `run_sims.py` + end-to-end smoke test

**Files:**
- Create: `horizon_chromaticity/run_sims.py`
- Modify: `horizon_chromaticity/test_smoke.py` (append the end-to-end test)

- [ ] **Step 1: Write the failing test**

Append to `horizon_chromaticity/test_smoke.py`:

```python
@pytest.mark.parametrize("case", CASES)
def test_run_sims_end_to_end(case, horizons):
    """Tiny full run per case: finite, positive, correctly shaped."""
    outfile = OUTPUT_DIR / f"chromaticity_{case}_smoke.npz"
    outfile.unlink(missing_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_DIR / "run_sims.py"),
            "--case", case,
            "--n-times", "3",
            "--max-orientations", "2",
            "--freq-stride", "50",
            "--batch-size", "2",
            "--output-tag", "_smoke",
        ],
        check=True,
    )
    d = np.load(outfile)
    t_sys = d["t_sys"]
    # 2 orientations, 3 times, 5 freqs (201 channels, stride 50)
    assert t_sys.shape == (2, 3, 5)
    assert np.all(np.isfinite(t_sys))
    assert np.all(t_sys > 0)
    assert str(d["case"]) == case
    assert len(d["elevations"]) == 2
    assert len(d["azimuths"]) == 2
    outfile.unlink()
```

- [ ] **Step 2: Run the new test to verify it fails**

Run: `cd /home/christian/Documents/research/eigsep/mock_analysis && EIGSEP_SMOKE=1 uv run pytest horizon_chromaticity/test_smoke.py -v -k end_to_end`
Expected: 3 FAILs (`run_sims.py` does not exist)

- [ ] **Step 3: Write `run_sims.py`**

Create `horizon_chromaticity/run_sims.py`:

```python
"""Run a noiseless EIGSEP chromaticity simulation for one horizon case.

Loads the horizon mask built by ``make_horizons.py``, runs
``eigsim.simulate()`` over the canonical orientation grid for one
sidereal day, and saves raw noiseless ``t_sys`` to
``output/chromaticity_<case>.npz``.

Orientations are processed in batches; completed batches on disk
(``output/<case>_batch_*.npz``) are reused on the next run, so the
script is safe to interrupt and rerun.

Usage
-----
uv run python horizon_chromaticity/make_horizons.py   # once, first
uv run python horizon_chromaticity/run_sims.py --case eigsep
"""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import croissant as cro
import numpy as np
from astropy import units as u
from astropy.time import Time
from pygdsm import GlobalSkyModel16

import eigsim

CASES = ("nohorizon", "quarry", "eigsep")
T_START = "2026-07-01 06:00:00"  # UTC (July 1 2026 00:00 Mountain Time)
SIDEREAL_DAY_S = cro.constants.sidereal_day["earth"]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--case", required=True, choices=CASES)
    p.add_argument(
        "--n-times", type=int, default=1436,
        help="time samples over one sidereal day (default ~1 min cadence)",
    )
    p.add_argument(
        "--batch-size", type=int, default=100,
        help="orientations per checkpoint batch",
    )
    p.add_argument(
        "--max-orientations", type=int, default=None,
        help="truncate the orientation grid (smoke tests only)",
    )
    p.add_argument(
        "--freq-stride", type=int, default=1,
        help="use every Nth config frequency (smoke tests only)",
    )
    p.add_argument(
        "--output-tag", default="",
        help="suffix for output/batch filenames (smoke tests only)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = eigsim.load_config()

    horizons_file = OUTPUT_DIR / "horizons.npz"
    if not horizons_file.exists():
        raise SystemExit(
            f"{horizons_file} not found — run make_horizons.py first"
        )
    hz = np.load(horizons_file)
    horizon = hz[args.case]
    theta_c_deg = float(hz["theta_c_deg"]) if args.case == "quarry" else np.nan
    omega_blocked = float(hz[f"omega_blocked_{args.case}"])

    print("Loading beam...")
    beam_freqs_hz, beam_data, lmax = eigsim.load_beam()
    freqs_mhz = np.array(cfg["frequencies"], dtype=float)[:: args.freq_stride]
    freq_idx = np.isin(beam_freqs_hz / 1e6, freqs_mhz)
    beam_data = beam_data[freq_idx]
    n_freqs = len(freqs_mhz)
    assert beam_data.shape[0] == n_freqs
    print(f"  Selected {n_freqs}/{len(beam_freqs_hz)} beam channels")

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
    times = cro.utils.time_array(
        t_start=t_start, t_end=t_end, N_times=args.n_times
    )
    times_jd = times.jd

    ori = cfg["orientations"]
    elev_vals = np.array(ori["elevations"], dtype=float)
    az_vals = np.array(ori["azimuths"], dtype=float)
    elev_grid, az_grid = np.meshgrid(elev_vals, az_vals, indexing="ij")
    elevations = elev_grid.ravel()
    azimuths = az_grid.ravel()
    if args.max_orientations is not None:
        elevations = elevations[: args.max_orientations]
        azimuths = azimuths[: args.max_orientations]
    n_ori = len(elevations)
    print(f"Case '{args.case}': {n_ori} orientations")

    print("Pre-computing sky ALM...")
    sky_alm = eigsim.precompute_sky_alm(sky)

    OUTPUT_DIR.mkdir(exist_ok=True)
    n_batches = int(np.ceil(n_ori / args.batch_size))
    print(
        f"Running simulation ({n_ori} orientations x {args.n_times} times "
        f"x {n_freqs} freqs) in {n_batches} batches of {args.batch_size}..."
    )

    wall_start = time.time()
    batch_files = []
    for b in range(n_batches):
        i0 = b * args.batch_size
        i1 = min(i0 + args.batch_size, n_ori)
        batch_file = (
            OUTPUT_DIR / f"{args.case}{args.output_tag}_batch_{b:04d}.npz"
        )
        batch_files.append(batch_file)

        if batch_file.exists():
            print(
                f"  Batch {b + 1}/{n_batches} [{i0}:{i1}] — "
                "found on disk, skipping"
            )
            continue

        print(f"  Batch {b + 1}/{n_batches} [{i0}:{i1}]")
        t0 = time.time()
        t_sys = eigsim.simulate(
            beam_data,
            freqs_mhz,
            sky,
            times_jd,
            elevations[i0:i1],
            azimuths[i0:i1],
            beam_kw={"horizon": horizon},
            sky_alm=sky_alm,
            verbose=True,
        )
        np.savez(batch_file, t_sys=np.asarray(t_sys))
        print(f"  Batch {b + 1}/{n_batches} done in {time.time() - t0:.0f}s")

    print(f"All batches complete in {(time.time() - wall_start) / 3600:.1f} h")

    print("Merging batches...")
    t_sys = np.concatenate(
        [np.load(f)["t_sys"] for f in batch_files], axis=0
    )
    assert t_sys.shape == (n_ori, args.n_times, n_freqs)

    outfile = OUTPUT_DIR / f"chromaticity_{args.case}{args.output_tag}.npz"
    print(f"Saving to {outfile}...")
    np.savez_compressed(
        outfile,
        # Simulation output (noiseless system temperature)
        t_sys=t_sys,
        # Axes
        freqs_mhz=freqs_mhz,
        times_jd=times_jd,
        elevations=elevations,  # flat, one per orientation
        azimuths=azimuths,  # flat, one per orientation
        elev_vals=elev_vals,  # grid axis values
        az_vals=az_vals,  # grid axis values
        # Horizon metadata
        case=args.case,
        theta_c_deg=theta_c_deg,
        omega_blocked=omega_blocked,
        # Config / metadata
        t_start=T_START,
        n_times=args.n_times,
        lon=cfg["location"]["lon"],
        lat=cfg["location"]["lat"],
        alt=cfg["location"]["alt"],
        world=cfg["world"],
        t_ground=cfg["ground"]["temperature"],
        t_receiver=cfg["receiver"]["temperature"],
        sky_model=sky_cfg["model"],
        sky_resolution=sky_cfg["resolution"],
        sky_include_cmb=sky_cfg["include_cmb"],
        beam_file=cfg["beam"]["file"],
        beam_lmax=lmax,
    )
    print(f"Done. Output size: {outfile.stat().st_size / 1e6:.0f} MB")

    for f in batch_files:
        f.unlink(missing_ok=True)
    print("Batch files cleaned up.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the full smoke suite to verify everything passes**

Run: `cd /home/christian/Documents/research/eigsep/mock_analysis && EIGSEP_SMOKE=1 uv run pytest horizon_chromaticity/test_smoke.py -v`
Expected: all 9 tests PASS (6 mask tests + 3 end-to-end cases). The end-to-end tests take a few minutes total (JAX compilation per subprocess).

- [ ] **Step 5: Lint and commit**

```bash
cd /home/christian/Documents/research/eigsep/mock_analysis
uv run ruff check horizon_chromaticity/ && uv run ruff format horizon_chromaticity/
git add horizon_chromaticity/run_sims.py horizon_chromaticity/test_smoke.py
git commit -m "feat(horizon_chromaticity): noiseless chromaticity sim runner with checkpoint/resume"
```

---

### Task 5: Production runs (manual, after implementation)

Not part of the coding work — kick these off when ready (each takes
roughly one canonical-sim runtime; safe to interrupt and resume):

```bash
cd /home/christian/Documents/research/eigsep/mock_analysis
uv run python horizon_chromaticity/make_horizons.py
uv run python horizon_chromaticity/run_sims.py --case nohorizon
uv run python horizon_chromaticity/run_sims.py --case quarry
uv run python horizon_chromaticity/run_sims.py --case eigsep
```

Expected products: `horizon_chromaticity/output/chromaticity_<case>.npz`,
each `(1296, 1436, 201)` float64 (~GB-scale compressed).
