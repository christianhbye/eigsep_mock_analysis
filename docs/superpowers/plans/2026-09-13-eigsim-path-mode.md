# eigsim D5 Path Mode Implementation Plan

> **History:** executed in PR #13. `eigsep_cal/docs/interface.md` has since been split into `eigsep_cal/docs/api.md` and the eigsep_analysis workspace `docs/interface.md`, so the spec pointers below are stale.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `eigsim.simulate_path`, which simulates one antenna orientation per time sample (as D5 motor telemetry gives) and returns the free-space antenna temperature `(n_time, n_freq)` with no receiver term.

**Why:** once the antenna moves, orientation and time are two views of one sample, so the natural unit is the sample, not an orientation × time grid. Today's options both waste work. One `simulate()` call computes every orientation at every time. A loop of `simulate()` calls, one per orientation, recomputes the beam's spherical-harmonic transform and recompiles the JIT function each time (`_build_orientation_fn` makes a new `jax.jit` closure per call). The Jul 17 raster has many orientations, so that matters.

**Architecture:** Split `simulate()` into a shared setup step (beam alm, reference croissant simulator, phases, JIT'd per-orientation function) and a per-orientation run. `simulate()` keeps its grid output. `simulate_path()` groups samples by unique (elevation, azimuth), runs each group once on that group's time phases, and reassembles in sample order. The mount composition stays in `drive_rotation_matrix`, unchanged.

**Tech Stack:** Python 3.12, JAX (x64), croissant-sim, s2fft, numpy, pytest, ruff (line length 88). uv workspace at the mock_analysis root.

**Spec:** `eigsep_cal/docs/interface.md` (branch `feat/forward-model`) § 7 (eigsim deliverables) and § 5.1 (`SkyTemperature.t_ant_k`). Read § 7 before starting.

## Global Constraints

- eigsim never imports eigsep_cal and knows nothing about receivers. It returns plain arrays (spec § 1).
- Precision: float64 throughout; eigsim enables JAX x64 on import (spec § 2).
- Path mode does **not** add `receiver.temperature` (spec § 7).
- The existing `simulate()` and its grid output stay unchanged (spec § 7).
- Rotation: `drive_rotation_matrix` composes `R_X(el) @ R_Z(az)`, body→top. That is already the physical azimuth-outer mount. **Do not flip it** (spec § 7; workspace logbook 2026-09-13).
- Frequencies: accept any strictly increasing array, in particular D5 channels `f_k = k * 250 / 1024` MHz. Beam and simulation frequencies must match exactly (croissant).
- Array axes: time, then frequency: `(n_time, n_freq)` (spec § 2).
- Commands run from the monorepo root with `uv run`. Never `python -c` for multiline code.
- Conventional commits. Branch `feat/eigsim-path-mode` off `main`.

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `eigsim/src/eigsim/simulate.py` | Modify | Extract `_Setup`, `_setup`, `_run_orientation`; refactor `simulate` onto them; add `simulate_path` |
| `eigsim/src/eigsim/__init__.py` | Modify | Export `simulate_path` |
| `eigsim/tests/test_path.py` | Create | Path-mode tests (spec § 7) |
| `eigsim/CLAUDE.md` | Modify | Document path mode |

---

### Task 1: Shared setup for simulate()

**Files:**
- Modify: `eigsim/src/eigsim/simulate.py` (`simulate`, currently from `def simulate(` to the end of the file)

**Interfaces:**
- Consumes: the existing `drive_rotation_matrix` (`eigsim/rotations.py`), `rotmat_to_eulerZYZ` (`croissant.rotations`), both already imported in `simulate.py`, and `_build_orientation_fn(beam_L, sim_L, sampling, nside, eul_topo)`, whose returned function takes `(beam_alm, euler_drive, dl_topo, horizon, quad_weights, sky_alm, phases, beam_norm, Tgnd)` and returns `(n_times, n_freqs)` for `phases` of shape `(n_times, 2 * sim_lmax + 1)`.
- Produces (module-private, used by Task 2):
  - `_Setup`: a `NamedTuple` with fields `alm, run, dl_topo, horizon, quad_weights, sky_alm, phases, beam_norm, Tgnd, t_rcvr`.
  - `_setup(beam_data, freqs, sky, times_jd, config, sampling, beam_kw, sky_alm, sim_kw) -> _Setup`
  - `_run_orientation(setup: _Setup, elevation_deg, azimuth_deg, phases) -> jax.Array` of shape `(phases.shape[0], n_freqs)`, noiseless antenna temperature without the receiver term.

- [ ] **Step 1: Create the branch**

```bash
git checkout main && git pull && git checkout -b feat/eigsim-path-mode
```

- [ ] **Step 2: Record the current output as a safety net**

Create `eigsim/tests/test_path.py` with the shared fixtures and a pin on `simulate()` (later tasks add to this file):

```python
"""Tests for simulate_path, eigsim's D5 path mode (interface spec § 7)."""

import jax

jax.config.update("jax_enable_x64", True)

import croissant as cro  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import s2fft  # noqa: E402
from astropy.time import Time  # noqa: E402
from eigsim.config import load_config  # noqa: E402
from eigsim.simulate import simulate  # noqa: E402

LMAX = 16
L = LMAX + 1
SAMPLING = "mwss"
NTHETA = L + 1
NPHI = 2 * L
# D5 channels k = 192, 208, 528: non-uniform on purpose.
FREQS_MHZ = np.array([192, 208, 528]) * 250 / 1024
RCVR_TEMP = load_config()["receiver"]["temperature"]


def _grids():
    thetas = s2fft.sampling.s2_samples.thetas(L=L, sampling=SAMPLING)
    phis = s2fft.sampling.s2_samples.phis_equiang(L=L, sampling=SAMPLING)
    return np.meshgrid(thetas, phis, indexing="ij")


def _beam():
    """Chromatic beam with azimuthal structure."""
    th, ph = _grids()
    pattern = 0.5 + 0.3 * np.cos(th) + 0.2 * np.sin(th) * np.cos(ph)
    scale = 1.0 + 0.1 * np.arange(FREQS_MHZ.size)
    return pattern[None] * scale[:, None, None]


def _sky():
    th, ph = _grids()
    pattern = 1000 + 500 * np.cos(th) + 200 * np.sin(th) * np.cos(ph)
    data = np.broadcast_to(pattern[None], (FREQS_MHZ.size, NTHETA, NPHI)).copy()
    return cro.Sky(data, FREQS_MHZ, sampling=SAMPLING, coord="equatorial")


def _times(n):
    """n samples 10 minutes apart on the night of Jul 16/17."""
    t0 = Time("2026-07-17 04:00:00", scale="utc").jd
    return t0 + np.arange(n) * 600.0 / 86400.0


def _croissant_t_ant(beam, sky, times):
    """Direct croissant antenna temperature for the unrotated beam."""
    cfg = load_config()
    loc = cfg["location"]
    sim = cro.Simulator(
        cro.Beam(beam, FREQS_MHZ, sampling=SAMPLING),
        sky,
        times,
        FREQS_MHZ,
        lon=loc["lon"],
        lat=loc["lat"],
        alt=loc["alt"],
        world=cfg["world"],
        Tgnd=cfg["ground"]["temperature"],
    )
    return np.asarray(sim.sim())


class TestSimulateUnchanged:
    def test_unrotated_is_croissant_plus_receiver(self):
        """simulate() keeps its grid output and receiver term (spec § 7)."""
        beam, sky, times = _beam(), _sky(), _times(2)
        got = np.asarray(simulate(beam, FREQS_MHZ, sky, times, [0.0], [0.0]))
        assert got.shape == (1, 2, FREQS_MHZ.size)
        np.testing.assert_allclose(
            got[0], _croissant_t_ant(beam, sky, times) + RCVR_TEMP, rtol=0, atol=1e-10
        )
```

- [ ] **Step 3: Run the existing suite before refactoring**

Run: `uv run pytest eigsim/tests -q`
Expected: all pass. Note the count.

- [ ] **Step 4: Refactor**

In `eigsim/src/eigsim/simulate.py`, add `from typing import NamedTuple` to the imports. Replace the body of `simulate` (everything after its docstring) and add the helpers directly above `def simulate(`:

```python
class _Setup(NamedTuple):
    """Everything one simulation shares across orientations."""

    alm: object
    run: object
    dl_topo: object
    horizon: object
    quad_weights: object
    sky_alm: object
    phases: object
    beam_norm: object
    Tgnd: object
    t_rcvr: float


def _setup(beam_data, freqs, sky, times_jd, config, sampling, beam_kw, sky_alm, sim_kw):
    """Precompute what every orientation of one simulation shares."""
    cfg = load_config(config)
    beam_kw = beam_kw or {}

    loc = cfg["location"]
    defaults = dict(
        lon=loc["lon"],
        lat=loc["lat"],
        alt=loc["alt"],
        world=cfg["world"],
        Tgnd=cfg["ground"]["temperature"],
    )
    defaults.update(sim_kw)

    # Pre-compute the forward SHT of the unrotated beam once.
    beam_data = np.asarray(beam_data)
    lmax = cro.utils.lmax_from_ntheta(beam_data.shape[1], sampling)
    beam_L = lmax + 1
    nside = None
    if sampling == "healpix":
        nside = cro.utils.hp_npix2nside(beam_data.shape[1])
    alm = beam_to_alm(beam_data, lmax, sampling, nside=nside)

    # Reference Simulator for the frame-rotation parameters.
    ref_beam = cro.Beam(beam_data, freqs, sampling=sampling, niter=0, **beam_kw)
    ref_sim = cro.Simulator(ref_beam, sky, times_jd, freqs, **defaults)
    if sky_alm is None:
        sky_alm = ref_sim.precompute_sky_alm()

    # Truncate dl_topo and sky_alm to the simulation resolution.
    sim_L = ref_sim.lmax + 1
    d = beam_L - sim_L
    end = d + 2 * sim_L - 1

    if sampling == "healpix":
        npix = 12 * nside**2
        quad_weights = jnp.ones(npix) * (4 * jnp.pi / npix)
    else:
        quad_weights = s2fft.utils.quadrature_jax.quad_weights(
            L=beam_L, sampling=sampling, nside=nside
        )

    return _Setup(
        alm=alm,
        run=_build_orientation_fn(beam_L, sim_L, sampling, nside, ref_sim.eul_topo),
        dl_topo=ref_sim.dl_topo[:sim_L, d:end, d:end],
        horizon=ref_beam.horizon,
        quad_weights=quad_weights,
        sky_alm=cro.utils.reduce_lmax(sky_alm, ref_sim.lmax),
        phases=ref_sim.phases,
        beam_norm=ref_beam.compute_norm(),
        Tgnd=jnp.asarray(defaults["Tgnd"], dtype=jnp.float64),
        t_rcvr=cfg["receiver"]["temperature"],
    )


def _run_orientation(setup, elevation_deg, azimuth_deg, phases):
    """Antenna temperature for one orientation at the times of *phases*."""
    R = drive_rotation_matrix(float(elevation_deg), float(azimuth_deg))
    euler = jnp.asarray(rotmat_to_eulerZYZ(R), dtype=jnp.float64)
    return setup.run(
        setup.alm,
        euler,
        setup.dl_topo,
        setup.horizon,
        setup.quad_weights,
        setup.sky_alm,
        phases,
        setup.beam_norm,
        setup.Tgnd,
    )
```

New body of `simulate` (keep its signature and docstring):

```python
    setup = _setup(
        beam_data, freqs, sky, times_jd, config, sampling, beam_kw, sky_alm, sim_kw
    )

    n_ori = len(elevations)
    results = []
    for i, (elev, az) in enumerate(zip(elevations, azimuths)):
        if verbose:
            print(f"    orientation {i + 1}/{n_ori}    ", end="\r", flush=True)
        results.append(_run_orientation(setup, elev, az, setup.phases))

    if verbose:
        print()

    return jnp.stack(results) + setup.t_rcvr
```

- [ ] **Step 5: Run the whole eigsim suite**

Run: `uv run pytest eigsim/tests -q`
Expected: the same count as Step 3 plus one, all passing.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format . && uv run ruff check .
git add eigsim/src/eigsim/simulate.py eigsim/tests/test_path.py
git commit -m "refactor(eigsim): share simulate() setup across orientations"
```

---

### Task 2: simulate_path

**Files:**
- Modify: `eigsim/src/eigsim/simulate.py` (add `simulate_path` after `simulate`)
- Modify: `eigsim/src/eigsim/__init__.py`
- Test: `eigsim/tests/test_path.py`

**Interfaces:**
- Consumes: `_setup`, `_run_orientation`, `_Setup.phases` (Task 1).
- Produces: `eigsim.simulate_path(beam_data, freqs_mhz, sky, times_jd, elevations_deg, azimuths_deg, config=None, sampling="mwss", beam_kw=None, sky_alm=None, verbose=False, **sim_kw) -> jax.Array`, float64, shape `(n_time, n_freq)`. Sample `i` is the antenna temperature at `times_jd[i]` with orientation `(elevations_deg[i], azimuths_deg[i])`. No receiver term. This is `SkyTemperature.t_ant_k` in the spec (§ 5.1); the generator converts `times_unix` to Julian days (spec § 2).

- [ ] **Step 1: Write the failing tests**

In `eigsim/tests/test_path.py`, change the import to `from eigsim.simulate import simulate, simulate_path` and append:

```python
class TestSimulatePath:
    def test_matches_grid_mode_minus_receiver(self):
        """At matching (orientation, time) samples, path = grid - receiver."""
        beam, sky, times = _beam(), _sky(), _times(4)
        els = np.array([0.0, 30.0, -60.0])
        azs = np.array([0.0, 45.0, 150.0])
        grid = np.asarray(simulate(beam, FREQS_MHZ, sky, times, els, azs))
        ori_of_sample = np.array([2, 0, 1, 2])

        path = np.asarray(
            simulate_path(
                beam, FREQS_MHZ, sky, times, els[ori_of_sample], azs[ori_of_sample]
            )
        )

        want = grid[ori_of_sample, np.arange(times.size)] - RCVR_TEMP
        np.testing.assert_allclose(path, want, rtol=0, atol=1e-10)

    def test_grouping_matches_per_sample_runs(self):
        """Grouping by unique orientation equals running each sample alone."""
        beam, sky, times = _beam(), _sky(), _times(5)
        els = np.array([0.0, 30.0, 0.0, 30.0, 0.0])
        azs = np.array([0.0, 45.0, 0.0, 45.0, 0.0])

        batched = np.asarray(simulate_path(beam, FREQS_MHZ, sky, times, els, azs))
        single = np.stack(
            [
                np.asarray(
                    simulate_path(
                        beam,
                        FREQS_MHZ,
                        sky,
                        times[i : i + 1],
                        els[i : i + 1],
                        azs[i : i + 1],
                    )
                )[0]
                for i in range(times.size)
            ]
        )

        np.testing.assert_allclose(batched, single, rtol=0, atol=1e-10)

    def test_no_receiver_term(self):
        """Uniform sky, open horizon: T_ant is the sky, with nothing added."""
        t0 = 1000.0
        beam = np.ones((FREQS_MHZ.size, NTHETA, NPHI))
        data = np.full((FREQS_MHZ.size, NTHETA, NPHI), t0)
        sky = cro.Sky(data, FREQS_MHZ, sampling=SAMPLING, coord="equatorial")
        open_sky = np.ones((NTHETA, NPHI), dtype=bool)

        got = np.asarray(
            simulate_path(
                beam,
                FREQS_MHZ,
                sky,
                _times(2),
                [0.0, 30.0],
                [0.0, 45.0],
                beam_kw={"horizon": open_sky},
            )
        )

        np.testing.assert_allclose(got, t0, rtol=1e-8)

    def test_shape_and_dtype(self):
        got = simulate_path(
            _beam(), FREQS_MHZ, _sky(), _times(3), [0.0] * 3, [0.0] * 3
        )
        assert got.shape == (3, FREQS_MHZ.size)
        assert got.dtype == np.float64

    @pytest.mark.parametrize(
        "els, azs", [([0.0, 0.0], [0.0, 0.0, 0.0]), ([0.0, 0.0, 0.0], [0.0])]
    )
    def test_length_mismatch_raises(self, els, azs):
        with pytest.raises(ValueError, match="one orientation per time"):
            simulate_path(_beam(), FREQS_MHZ, _sky(), _times(3), els, azs)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest eigsim/tests/test_path.py -k SimulatePath -v`
Expected: FAIL with `ImportError: cannot import name 'simulate_path'`.

- [ ] **Step 3: Implement**

Append to `eigsim/src/eigsim/simulate.py`:

```python
def simulate_path(
    beam_data,
    freqs_mhz,
    sky,
    times_jd,
    elevations_deg,
    azimuths_deg,
    config=None,
    sampling="mwss",
    beam_kw=None,
    sky_alm=None,
    verbose=False,
    **sim_kw,
):
    """Simulate one antenna orientation per time sample (D5 path mode).

    Motor telemetry gives one (elevation, azimuth) per sample. Samples
    are grouped by unique orientation and each group is simulated once,
    at that group's times only; D5 orientations repeat (static at night,
    a raster on Jul 17), so this is much cheaper than a full grid.

    Unlike :func:`simulate`, no receiver temperature is added: the
    result is the free-space antenna temperature, ``t_ant_k`` of
    ``SkyTemperature`` in the eigsep_cal interface spec (§ 5.1). It
    includes sky, horizon and the configured ground model, but no balun
    or coax.

    Each distinct group size compiles the per-orientation function
    once more, because JIT specialises on the number of times.

    Parameters
    ----------
    beam_data : array_like
        Unrotated beam power pattern.
    freqs_mhz : array_like
        Frequencies in MHz, matching the beam's frequency axis.
    sky : croissant.Sky
        Sky model.
    times_jd : array_like
        Sample times in Julian day, shape ``(n_time,)``.
    elevations_deg, azimuths_deg : array_like
        Drive angles in degrees, one per sample, shape ``(n_time,)``.
    config : str, Path, or None
        Config for :func:`~eigsim.config.load_config`.
    sampling : str
        Beam sampling scheme.
    beam_kw : dict or None
        Extra kwargs for ``croissant.Beam`` (e.g. *horizon*).
    sky_alm : jax.Array or None
        Pre-computed sky ALM from :func:`precompute_sky_alm`.
    verbose : bool
        Print per-group progress.
    **sim_kw
        Override Simulator kwargs (lon, lat, alt, world, Tgnd, lmax).

    Returns
    -------
    t_ant : jax.Array
        Noiseless antenna temperature, shape ``(n_time, n_freq)``.

    """
    times_jd = np.asarray(times_jd, dtype=np.float64)
    elevations_deg = np.asarray(elevations_deg, dtype=np.float64)
    azimuths_deg = np.asarray(azimuths_deg, dtype=np.float64)
    shapes = {times_jd.shape, elevations_deg.shape, azimuths_deg.shape}
    if len(shapes) != 1 or times_jd.ndim != 1:
        raise ValueError(
            "need one orientation per time: times_jd, elevations_deg and "
            f"azimuths_deg must be 1-D with equal length, got {sorted(shapes)}"
        )

    setup = _setup(
        beam_data, freqs_mhz, sky, times_jd, config, sampling, beam_kw, sky_alm, sim_kw
    )

    orientations, group = np.unique(
        np.column_stack([elevations_deg, azimuths_deg]), axis=0, return_inverse=True
    )
    group = group.reshape(-1)

    pieces, order = [], []
    for g, (elev, az) in enumerate(orientations):
        if verbose:
            print(
                f"    orientation {g + 1}/{len(orientations)}    ", end="\r", flush=True
            )
        idx = np.flatnonzero(group == g)
        pieces.append(_run_orientation(setup, elev, az, setup.phases[idx]))
        order.append(idx)

    if verbose:
        print()

    order = np.concatenate(order)
    return jnp.concatenate(pieces)[np.argsort(order)]
```

In `eigsim/src/eigsim/__init__.py`, add `simulate_path` to the `from .simulate import (...)` block and to `__all__` (alphabetical, after `"simulate"`).

- [ ] **Step 4: Run the path tests, then the whole suite**

Run: `uv run pytest eigsim/tests/test_path.py -v`
Expected: all pass.

Run: `uv run pytest eigsim/tests -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check .
git add eigsim/src/eigsim/simulate.py eigsim/src/eigsim/__init__.py eigsim/tests/test_path.py
git commit -m "feat(eigsim): add simulate_path for one orientation per time sample"
```

---

### Task 3: Documentation and PR

**Files:**
- Modify: `eigsim/CLAUDE.md` (the `simulate.py` bullet under "Module dependency flow")

**Interfaces:**
- Consumes: `simulate_path` (Task 2).
- Produces: nothing new in code.

- [ ] **Step 1: Document path mode**

In `eigsim/CLAUDE.md`, replace the `simulate.py` bullet with:

```markdown
- **`simulate.py`** — `simulate()` orchestrates multi-orientation runs: for each (elevation, azimuth) pair, rotates the beam, runs croissant, and stacks results into `(N_orientations, N_times, N_freqs)`, adding the receiver temperature. `simulate_path()` is D5 path mode: one orientation per time sample, grouped by unique orientation, returning `(N_times, N_freqs)` with **no** receiver term (`SkyTemperature.t_ant_k`, interface spec § 5.1). Both share `_setup()`, so the beam transform and JIT compilation happen once per call, not once per orientation.
```

- [ ] **Step 2: Final checks**

Run: `uv run pytest eigsim/tests -q && uv run ruff check . && uv run ruff format --check .`
Expected: all pass.

- [ ] **Step 3: Commit, push, open the PR**

```bash
git add eigsim/CLAUDE.md
git commit -m "docs(eigsim): document simulate_path"
git push -u origin feat/eigsim-path-mode
gh pr create --base main --title "feat(eigsim): D5 path mode (simulate_path)" --body "Implements eigsep_cal interface spec § 7: simulate_path, grouping by unique orientation, no receiver term, arbitrary (D5) frequency grids. The drive composition is unchanged (body→top R_X(el) @ R_Z(az), already the physical mount)."
```

---

## Downstream (not in this plan)

- Stage 4 (analysis workspace, `eigsep_analysis.beammap`) will export `pointing_export.h5`: `times_unix`, elevation/azimuth medians and 100 draws, per-block position and heading, and `rotation="body_to_top: R_X(el) @ R_Z(az)"`. The generator feeds these to `simulate_path`.
- box-gnd needs its orientation (Q-ARP-01) before it can use path mode.
- rotis `total_rotation` (`R_LST @ R_Z(az) @ R_X(tilt)`) has not been checked for direction; if it acts body→top it is the non-physical composition.
