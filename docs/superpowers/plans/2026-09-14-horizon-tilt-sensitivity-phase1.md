# Horizon and Levelling Sensitivity — Phase 1 (code) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `eigsim` a fractional, phi-integrated open-sky mask and an outer misalignment rotation, so the D5 forward model can take derivatives of antenna temperature with respect to antenna position and instrument levelling.

**Architecture:** The horizon curve `alpha_h(az)` is produced outside `eigsim` by `horizon_position/make_horizons.py`, which also emits an analytic Jacobian `d alpha_h/d(e,n,u)` derived from the DEM pixel that sets each azimuth's horizon. `eigsim` gains a JAX module that turns that curve into a fractional mask `W`, so everything from `alpha_h` to `t_ant_k` is one differentiable graph and position derivatives are a `jax.jvp` along the supplied tangent.

**Tech Stack:** Python 3.12, JAX (x64), s2fft, croissant-sim, numpy, pytest, ruff, uv.

**Spec:** `docs/superpowers/specs/2026-09-14-horizon-tilt-sensitivity-design.md`

## Global Constraints

- Run every Python tool through `uv run`, never bare `python` or `pytest`.
- Never use `python -c "..."` for multiline scripts; write a file and run it.
- Conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).
- `eigsim` sets `JAX_ENABLE_X64=1` at import; keep all mask maths float64.
- MWSS grid at `lmax = 128`: `n_theta = 130`, `n_phi = 258`, `dtheta = 1.395 deg`.
- Horizon curve native resolution is `N_AZ = 46080`; `n_phi` does not divide it, so phi integration sub-samples — never reshape the native curve.
- Do not modify `eigsep_terrain`. It is a read-only upstream.
- Do not change `horizon_position/masks.py`, `run_beam_sims.py`, or the packaged `horizon_mwss.npz`. The RASTI paper must stay reproducible from tag `rasti-round2-figs`.
- `make_horizons.py` runs only in the `eigsep_terrain` environment:
  ```
  PYTHONPATH=/home/christian/Documents/research/eigsep/eigsep_terrain \
  uv run --project /home/christian/Documents/research/eigsep/eigsep_terrain \
      python horizon_position/make_horizons.py
  ```

## Deviations from the spec

Two items changed while planning. Both are deliberate; raise them if you disagree.

1. **`margin` is dropped.** The spec asked the generator to store the runner-up horizon angle per azimuth as a predictor of argmax fragility. Computing it exactly requires the recursive pruning inside `DEM.calc_horizon`, which the spec forbids touching. The *measured* switch fraction (Task 6), obtained by comparing `crds` across the 19 positions, serves the same purpose in the memo and needs no new DEM machinery.
2. **Whether `dT/d(eps)` is autodiff or finite difference is decided by a spike (Task 3),** not assumed. The spec asserts autodiff; that requires gradients to flow through `croissant.rotations.rotmat_to_eulerZYZ` and s2fft's Wigner-d recursion, which is unverified. With only two misalignment parameters, central differences cost four extra simulations, so the fallback is cheap.

## File Structure

| File | Responsibility |
|---|---|
| `eigsim/src/eigsim/horizon.py` | **Create.** JAX fractional open-sky weight from `alpha_h(az)`. Grid helpers, phi integration. No DEM, no terrain dependency. |
| `eigsim/src/eigsim/__init__.py` | **Modify.** Export `open_sky_weight`, `mwss_grid`. |
| `eigsim/src/eigsim/rotations.py` | **Modify.** Add `rotation_matrix_y`, `misalignment_matrix`; add `misalignment=` to `drive_rotation_matrix`. |
| `eigsim/tests/test_horizon.py` | **Create.** Mask properties, phi-integration, gradients. |
| `eigsim/tests/test_rotations.py` | **Modify.** Misalignment identities and the meridian test. |
| `horizon_position/make_horizons.py` | **Modify.** Keep `crds` for all 19 positions; emit the nominal Jacobian. |
| `horizon_position/run_sims.py` | **Modify.** Delete checkpoint/resume; use `eigsim.horizon.open_sky_weight`. |
| `horizon_position/test_jacobian.py` | **Create.** Jacobian vs finite differences; switch-fraction diagnostic. |

---

### Task 1: `eigsim.horizon` — the phi-integrated fractional mask

**Files:**
- Create: `eigsim/src/eigsim/horizon.py`
- Modify: `eigsim/src/eigsim/__init__.py`
- Test: `eigsim/tests/test_horizon.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `eigsim.horizon.mwss_grid(lmax) -> (thetas, phis)` both `np.ndarray`; `eigsim.horizon.open_sky_weight(alpha_h, az_grid, lmax, sub=180) -> jax.Array` of shape `(n_theta, n_phi)`, values in `[0, 1]`, 1 = open sky. `alpha_h` and `az_grid` are `(n_az,)` in radians, `az = atan2(E, N)`.

- [ ] **Step 1: Write the failing tests**

Create `eigsim/tests/test_horizon.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eigsim.horizon import mwss_grid, open_sky_weight

LMAX = 128
N_AZ = 720


def _az_grid(n=N_AZ):
    return np.linspace(0.0, 2 * np.pi, n, endpoint=False)


def _solid_angle_fraction(W, thetas):
    w = np.sin(thetas)[:, None]
    return float((w * np.asarray(W)).sum() / (w * np.ones_like(np.asarray(W))).sum())


def test_grid_shapes():
    thetas, phis = mwss_grid(LMAX)
    assert thetas.shape == (130,)
    assert phis.shape == (258,)


def test_flat_horizon_is_half_open():
    thetas, _ = mwss_grid(LMAX)
    W = open_sky_weight(np.zeros(N_AZ), _az_grid(), LMAX)
    assert _solid_angle_fraction(W, thetas) == pytest.approx(0.5, abs=1e-3)


def test_all_open_and_all_blocked():
    W_open = open_sky_weight(np.full(N_AZ, -np.pi / 2), _az_grid(), LMAX)
    W_blocked = open_sky_weight(np.full(N_AZ, np.pi / 2), _az_grid(), LMAX)
    assert np.allclose(np.asarray(W_open), 1.0)
    assert np.allclose(np.asarray(W_blocked), 0.0)


def test_monotonic_in_theta():
    W = np.asarray(open_sky_weight(np.full(N_AZ, np.deg2rad(10.0)), _az_grid(), LMAX))
    assert np.all(np.diff(W, axis=0) <= 1e-12)


def test_subpixel_shift_registers():
    base = np.full(N_AZ, np.deg2rad(10.0))
    W0 = np.asarray(open_sky_weight(base, _az_grid(), LMAX))
    small = np.abs(
        np.asarray(open_sky_weight(base + np.deg2rad(0.05), _az_grid(), LMAX)) - W0
    ).sum()
    big = np.abs(
        np.asarray(open_sky_weight(base + np.deg2rad(0.5), _az_grid(), LMAX)) - W0
    ).sum()
    assert small > 0.0
    assert small < big


def test_frame_mapping_blocks_east():
    # horizon high only near az=90 deg (East) must reduce open sky near phi=0,
    # which is ENU East in croissant's beam frame.
    thetas, phis = mwss_grid(LMAX)
    az = _az_grid()
    alpha = np.deg2rad(40.0) * np.exp(-((az - np.pi / 2) ** 2) / (2 * 0.1**2))
    W = np.asarray(open_sky_weight(alpha, az, LMAX))
    ring = int(np.argmin(np.abs(thetas - np.deg2rad(85))))
    east = int(np.argmin(np.abs(phis - 0.0)))
    north = int(np.argmin(np.abs(phis - np.pi / 2)))
    assert W[ring, east] < W[ring, north]


def test_converges_in_sub():
    alpha = np.deg2rad(10.0) + np.deg2rad(3.0) * np.sin(7 * _az_grid())
    W90 = np.asarray(open_sky_weight(alpha, _az_grid(), LMAX, sub=90))
    W180 = np.asarray(open_sky_weight(alpha, _az_grid(), LMAX, sub=180))
    W360 = np.asarray(open_sky_weight(alpha, _az_grid(), LMAX, sub=360))
    assert np.abs(W180 - W360).max() < np.abs(W90 - W180).max()


def test_one_bin_spike_does_not_alias():
    # A spike one native bin wide must be attenuated by the cell integral
    # itself, with no reduce_azimuth step anywhere. This is issue #10's
    # property stated as a test.
    n_az = 46080
    az = _az_grid(n_az)
    alpha = np.full(n_az, np.deg2rad(5.0))
    alpha[n_az // 3] = np.deg2rad(45.0)
    W = np.asarray(open_sky_weight(alpha, az, LMAX))
    W_flat = np.asarray(open_sky_weight(np.full(n_az, np.deg2rad(5.0)), az, LMAX))
    diff = np.abs(W - W_flat)
    # the spike is ~1/179 of one phi cell, so it moves that cell a little and
    # every other cell not at all
    assert diff.max() < 0.05
    assert (diff > 1e-12).sum() < 3 * 130


def test_rejects_a_descending_az_grid():
    with pytest.raises(ValueError, match="ascending"):
        open_sky_weight(np.zeros(N_AZ), _az_grid()[::-1], LMAX)


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same 1-D shape"):
        open_sky_weight(np.zeros(N_AZ - 1), _az_grid(), LMAX)


def test_check_grads_on_the_mask():
    # numerical-vs-autodiff agreement, at a small band limit for speed
    from jax.test_util import check_grads

    az = _az_grid(64)
    alpha = jnp.asarray(np.deg2rad(10.0) + np.deg2rad(2.0) * np.sin(3 * az))

    def total(a):
        return open_sky_weight(a, az, 16, sub=8).sum()

    check_grads(total, (alpha,), order=1, modes=("rev",), atol=1e-4, rtol=1e-4)


def test_gradient_flows_and_is_finite():
    alpha = jnp.asarray(np.full(N_AZ, np.deg2rad(10.0)))
    az = jnp.asarray(_az_grid())

    def total(a):
        return open_sky_weight(a, az, LMAX).sum()

    g = jax.grad(total)(alpha)
    assert np.all(np.isfinite(np.asarray(g)))
    assert np.abs(np.asarray(g)).sum() > 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd eigsim && uv run pytest tests/test_horizon.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eigsim.horizon'`

- [ ] **Step 3: Write the implementation**

Create `eigsim/src/eigsim/horizon.py`:

```python
"""Fractional open-sky weight on the beam's MWSS grid, in JAX.

The horizon arrives as an elevation curve ``alpha_h(az)`` with
``az = atan2(E, N)`` (North->East).  The beam grid is MWSS with polar
angle theta (0 = zenith) and azimuth phi measured from ENU East
(croissant ``beam_rot=0``), so the frame map is ``phi = pi/2 - az``.
Open sky <=> elevation > alpha_h <=> theta < theta_h with
``theta_h = pi/2 - alpha_h``.

``W(theta, phi)`` in [0, 1] is the fraction of each grid **cell** above
the horizon, integrated over both theta and phi.  Integrating over the
phi cell is what band-limits the horizon: a curve sampled far finer than
the ~258 MWSS azimuths is averaged into the cell rather than
point-sampled at its centre, so no separate azimuth reduction step is
needed or wanted (eigsep_mock_analysis issue #10).

Everything is built in ``jnp`` so that ``dW/d alpha_h`` comes from
autodiff: a boolean mask has zero gradient almost everywhere and cannot
carry a horizon derivative at all.
"""

import jax.numpy as jnp
import numpy as np
import s2fft.sampling.s2_samples as s2

#: phi sub-samples per cell.  Fixed by the convergence test, not chosen:
#: 258 x 180 = 46440 fine azimuths against the native curve's 46080, so
#: the integral samples the curve at essentially its own resolution.
DEFAULT_SUB = 180


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


def open_sky_weight(alpha_h, az_grid, lmax, sub=DEFAULT_SUB):
    """Fractional open-sky weight ``W(theta, phi)`` in ``[0, 1]``.

    Parameters
    ----------
    alpha_h : (n_az,) array
        Horizon elevation [rad] vs azimuth.  Differentiable input.
    az_grid : (n_az,) array
        Azimuth [rad] of each ``alpha_h`` sample, ``= atan2(E, N)``,
        ascending on ``[0, 2 pi)``.
    lmax : int
        Band limit of the beam grid.
    sub : int
        Sub-samples per phi cell for the azimuth integral.

    Returns
    -------
    W : (n_theta, n_phi) jax.Array
        1 = open sky, 0 = blocked.

    """
    # az_grid is a fixed grid, validated concretely; alpha_h is the
    # differentiable input and is never inspected for its values.
    az_np = np.asarray(az_grid, dtype=np.float64)
    alpha_h = jnp.asarray(alpha_h)
    if az_np.ndim != 1 or alpha_h.shape != az_np.shape:
        raise ValueError(
            f"alpha_h {alpha_h.shape} and az_grid {az_np.shape} must be "
            "the same 1-D shape"
        )
    if np.any(np.diff(az_np) <= 0):
        raise ValueError(
            "az_grid must be strictly ascending; alpha_h is defined on "
            "az = atan2(E, N) over [0, 2*pi)"
        )
    if az_np[0] < 0.0 or az_np[-1] >= 2 * np.pi:
        raise ValueError(f"az_grid must lie in [0, 2*pi), got [{az_np[0]}, {az_np[-1]}]")
    az_grid = jnp.asarray(az_np)

    thetas, phis = mwss_grid(lmax)
    n_theta, n_phi = thetas.size, phis.size
    dphi = 2.0 * np.pi / n_phi

    # phi cell centres -> sub sample points spanning each cell
    off = (jnp.arange(sub) + 0.5) / sub - 0.5
    fine_phi = (jnp.asarray(phis)[:, None] + off[None, :] * dphi).ravel()

    az_of_phi = jnp.mod(jnp.pi / 2 - fine_phi, 2 * jnp.pi)
    alpha_fine = jnp.interp(az_of_phi, az_grid, alpha_h, period=2 * jnp.pi)
    theta_h = jnp.pi / 2 - alpha_fine

    edges = _theta_edges(thetas)
    lo = jnp.asarray(edges[:-1])[:, None]
    hi = jnp.asarray(edges[1:])[:, None]
    # Fraction of [lo, hi] with theta < theta_h.  Linear in theta, not
    # sin-theta weighted: cells are ~1.4 deg so sin is nearly constant
    # across one, making this first-order accurate in the sub-cell
    # horizon position.
    frac = jnp.clip((theta_h[None, :] - lo) / (hi - lo), 0.0, 1.0)
    return frac.reshape(n_theta, n_phi, sub).mean(axis=2)
```

- [ ] **Step 4: Export from the package**

In `eigsim/src/eigsim/__init__.py`, add after the `from .data import ...` line:

```python
from .horizon import mwss_grid, open_sky_weight
```

and add `"mwss_grid"` and `"open_sky_weight"` to `__all__`, keeping it alphabetical.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd eigsim && uv run pytest tests/test_horizon.py -v`
Expected: PASS, 12 tests.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add eigsim/src/eigsim/horizon.py eigsim/src/eigsim/__init__.py eigsim/tests/test_horizon.py
git commit -m "feat(eigsim): phi-integrated fractional open-sky mask in JAX"
```

---

### Task 2: `rotation_matrix_y` and outer misalignment

**Files:**
- Modify: `eigsim/src/eigsim/rotations.py:31-65`
- Test: `eigsim/tests/test_rotations.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `rotation_matrix_y(angle_rad) -> np.ndarray (3,3)`; `misalignment_matrix(tilt_y_deg=0.0, tilt_z_deg=0.0) -> np.ndarray (3,3)`; `drive_rotation_matrix(elevation_deg, azimuth_deg, misalignment=None) -> np.ndarray (3,3)` where `misalignment` is a `(3,3)` matrix applied **outside** the drive, `R_mis @ Rx(el) @ Rz(az)`.

- [ ] **Step 1: Write the failing tests**

Append to `eigsim/tests/test_rotations.py`:

```python
def test_misalignment_none_is_unchanged():
    from eigsim.rotations import drive_rotation_matrix, rotation_matrix_x, rotation_matrix_z

    R = drive_rotation_matrix(23.0, 61.0)
    expected = rotation_matrix_x(np.radians(23.0)) @ rotation_matrix_z(np.radians(61.0))
    assert np.array_equal(R, expected)


def test_outer_x_misalignment_is_an_elevation_offset():
    # Rotations about a shared axis commute and add, so an outer X
    # misalignment is exactly an elevation encoder offset and carries no
    # independent information.
    from eigsim.rotations import drive_rotation_matrix, rotation_matrix_x

    mis = rotation_matrix_x(np.radians(2.0))
    tilted = drive_rotation_matrix(23.0, 61.0, misalignment=mis)
    offset = drive_rotation_matrix(25.0, 61.0)
    assert np.allclose(tilted, offset, atol=1e-12)


def test_drive_alone_keeps_boresight_on_the_meridian():
    # The drive is Rx(el) @ Rz(az); Rz leaves +Z fixed, so the boresight
    # is Rx(el) @ zhat, whose East component is identically zero for every
    # commanded orientation.
    from eigsim.rotations import drive_rotation_matrix

    zhat = np.array([0.0, 0.0, 1.0])
    for el in (0.0, 15.0, 47.0, -30.0):
        for az in (0.0, 90.0, 217.0):
            boresight = drive_rotation_matrix(el, az) @ zhat
            assert abs(boresight[0]) < 1e-12


def test_y_and_z_misalignments_move_boresight_off_the_meridian():
    # This is why they are identifiable: no commanded (el, az) can
    # reproduce a boresight with a non-zero East component.
    from eigsim.rotations import drive_rotation_matrix, misalignment_matrix

    zhat = np.array([0.0, 0.0, 1.0])

    b_y = drive_rotation_matrix(20.0, 35.0, misalignment=misalignment_matrix(tilt_y_deg=1.5)) @ zhat
    assert abs(b_y[0]) > 1e-4

    b_z = drive_rotation_matrix(20.0, 35.0, misalignment=misalignment_matrix(tilt_z_deg=1.5)) @ zhat
    assert abs(b_z[0]) > 1e-4


def test_rotation_matrix_y_is_orthonormal():
    from eigsim.rotations import rotation_matrix_y

    R = rotation_matrix_y(np.radians(17.0))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd eigsim && uv run pytest tests/test_rotations.py -k "misalign or meridian or matrix_y" -v`
Expected: FAIL — `ImportError: cannot import name 'rotation_matrix_y'`

- [ ] **Step 3: Implement**

In `eigsim/src/eigsim/rotations.py`, after `rotation_matrix_z`:

```python
def rotation_matrix_y(angle_rad):
    """Rotation matrix around the Y-axis (North in ENU).

    The drive cannot produce this rotation — it has only an elevation
    axis (X) and a turntable (Z) — which is exactly why a Y tilt is an
    identifiable misalignment rather than an encoder offset.
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def misalignment_matrix(tilt_y_deg=0.0, tilt_z_deg=0.0):
    """Outer (mount-to-ground) misalignment of the whole instrument.

    ``tilt_y_deg`` is the levelling error about the North axis and
    ``tilt_z_deg`` the error of the azimuth reference against true North.
    A tilt about X is omitted on purpose: it is exactly an elevation
    encoder offset (``Rx(eps) @ Rx(el) == Rx(eps + el)``) and carries no
    independent information.
    """
    return rotation_matrix_z(np.radians(tilt_z_deg)) @ rotation_matrix_y(
        np.radians(tilt_y_deg)
    )
```

Then replace the body of `drive_rotation_matrix`. Change the signature to
`def drive_rotation_matrix(elevation_deg, azimuth_deg, misalignment=None):`,
add to its docstring's Parameters section:

```
    misalignment : (3, 3) array or None
        Outer mount-to-ground misalignment, applied *outside* the drive
        as ``R_mis @ Rx(el) @ Rz(az)`` so that it is static in the
        topocentric frame and does not rotate with the drive.  ``None``
        reproduces the bare drive exactly.
```

and replace the return statement with:

```python
    R = rotation_matrix_x(np.radians(elevation_deg)) @ rotation_matrix_z(
        np.radians(azimuth_deg)
    )
    if misalignment is None:
        return R
    return np.asarray(misalignment) @ R
```

- [ ] **Step 4: Run the whole eigsim suite**

Run: `cd eigsim && uv run pytest -v`
Expected: PASS. `test_regression.py` is the guard that the boolean horizon
path and the existing simulate outputs are untouched by Tasks 1 and 2 — a
failure there means the new mask or rotation leaked into the default path,
which it must not.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add eigsim/src/eigsim/rotations.py eigsim/tests/test_rotations.py
git commit -m "feat(eigsim): add rotation_matrix_y and outer drive misalignment"
```

---

### Task 3: Spike — can gradients flow through the beam rotation?

This decides whether `dT/d(eps_y, eps_z)` in Task 7 is autodiff or central differences. Timebox it: if it does not work in one sitting, take the finite-difference branch and move on. **Nothing built here is kept.**

**Files:**
- Create (throwaway): `/tmp/claude-*/scratchpad/spike_rot_grad.py`

**Interfaces:**
- Consumes: Task 2's `misalignment_matrix`.
- Produces: a recorded decision only — no code.

- [ ] **Step 1: Write the probe**

`rotate_alm_to_beam` takes `elevation_deg`/`azimuth_deg`, not Euler angles,
and builds the matrix with **numpy** before handing it to croissant's
`rotmat_to_eulerZYZ` (`rotations.py:146-148`). So `jax.grad` over the drive
angles cannot work as the code stands — that part is ours to convert and is
not what the spike is asking.

The load-bearing question is the part we do **not** control: whether s2fft's
Wigner-d machinery is differentiable in the Euler angles, given that `beta`
enters through a precomputed `dl_array`. Probe exactly that:

```python
"""Throwaway: is s2fft's Wigner-D rotation differentiable in the Euler angles?"""

import jax
import jax.numpy as jnp
import numpy as np
import s2fft

L = 33  # small and fast; differentiability does not depend on L

rng = np.random.default_rng(0)
flm = jnp.asarray(rng.normal(size=(L, 2 * L - 1)).astype(np.complex128))


def total(euler):
    dl = s2fft.generate_rotate_dls(L, euler[1])
    out = s2fft.utils.rotation.rotate_flms(flm, L=L, rotation=euler, dl_array=dl)
    return jnp.sum(jnp.abs(out) ** 2)


try:
    g = jax.grad(total)(jnp.asarray([0.1, 0.2, 0.3]))
    g = np.asarray(g)
    print("grad:", g, "finite:", bool(np.all(np.isfinite(g))), "nonzero:", bool(np.any(g != 0)))
except Exception as exc:  # noqa: BLE001
    print("FAILED:", type(exc).__name__, exc)
```

Expect this to be the deciding evidence. If it fails, the autodiff branch
would additionally require replacing `drive_rotation_matrix` and
`rotmat_to_eulerZYZ` with `jnp` equivalents — do not start that; take the
finite-difference branch, which costs four simulations for two parameters.

- [ ] **Step 2: Run it**

Run: `uv run --project eigsim python <scratchpad>/spike_rot_grad.py`

- [ ] **Step 3: Record the decision**

Add one line to this plan under Task 7 Step 1 saying which branch applies:
- gradient finite and non-zero → **autodiff branch**
- exception, NaN, or all-zero gradient → **finite-difference branch**

No commit. Delete the probe.

---

### Task 4: Horizon Jacobian in the generator

The derivative is closed-form calculus on the DEM pixel that sets each azimuth's horizon. With `alpha_h = arctan2(U_p - u0, r)` and `r` the horizontal distance to that pixel:

```
d alpha / dE = dz * uhat_E / rho2
d alpha / dN = dz * uhat_N / rho2
d alpha / dU = -r / rho2
```

where `dz = U_p - u0`, `rho2 = r**2 + dz**2`, and `uhat` is the horizontal unit vector from the antenna to the pixel. Sanity: moving up lowers the horizon (`dU` term negative); moving toward a pixel at the antenna's own height changes nothing (`dz = 0`).

`calc_horizon` computes `alpha_h` from `r_min`, the distance to the nearest *edge* of the pixel, while this uses the pixel centre — a half-pixel (0.25 m) difference at ranges of hundreds of metres. Task 6's finite-difference check is what adjudicates whether that matters.

**Files:**
- Modify: `horizon_position/make_horizons.py:74-95`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `output/horizons_position.npz` gains `crds` `(19, 2, n_az)` float64 metres in the DEM frame (`crds[i, 0]` = North, `crds[i, 1]` = East), `dalpha_dE`, `dalpha_dN`, `dalpha_dU` each `(n_az,)` rad/m at the nominal position, and `jac_valid` `(n_az,)` bool. Existing keys are unchanged.

- [ ] **Step 1: Add the Jacobian helper**

In `horizon_position/make_horizons.py`, above `main()`:

```python
def horizon_jacobian(dem, crds, alpha_h, e0, n0, u0):
    """Analytic d alpha_h / d(e, n, u) at one antenna position.

    ``crds`` is ``calc_horizon``'s second return: the DEM pixel that sets
    the horizon in each azimuth bin, in metres in the DEM frame, North
    first.  The derivative is exact wherever that pixel keeps winning;
    Task 6 of the plan measures how often it stops.

    Azimuths where no pixel ever won keep ``calc_horizon``'s initial
    zeros, which is not a horizon; they are returned as invalid.
    """
    n_p, e_p = np.asarray(crds[0], float), np.asarray(crds[1], float)
    valid = ~((alpha_h == 0.0) & (n_p == 0.0) & (e_p == 0.0))

    u_p = np.asarray(dem.interp_alt(e_p, n_p), dtype=np.float64).ravel()
    d_e, d_n = e_p - e0, n_p - n0
    r = np.hypot(d_e, d_n)
    valid &= r > 0.0

    r_safe = np.where(valid, r, 1.0)
    dz = u_p - u0
    rho2 = r_safe**2 + dz**2
    d_alpha_de = np.where(valid, dz * (d_e / r_safe) / rho2, 0.0)
    d_alpha_dn = np.where(valid, dz * (d_n / r_safe) / rho2, 0.0)
    d_alpha_du = np.where(valid, -r_safe / rho2, 0.0)
    return d_alpha_de, d_alpha_dn, d_alpha_du, valid
```

- [ ] **Step 2: Keep `crds` for every position**

In `main()`, replace the allocation and loop body so `crds` is retained:

```python
    alpha_h = np.empty((len(positions), N_AZ), dtype=np.float64)
    crds_all = np.empty((len(positions), 2, N_AZ), dtype=np.float64)
    for i, (name, e) in enumerate(positions):
        hangles, crds = dem.calc_horizon(
            float(e[0]), float(e[1]), float(e[2]), n_az=N_AZ
        )
        alpha_h[i] = np.asarray(hangles, dtype=np.float64)
        crds_all[i] = np.asarray(crds, dtype=np.float64)
        deg = np.degrees([alpha_h[i].min(), np.median(alpha_h[i]), alpha_h[i].max()])
        print(
            f"  [{i:2d}] {name:10s} alpha_h(min,med,max) deg = "
            f"{deg[0]:6.2f} {deg[1]:6.2f} {deg[2]:6.2f}"
        )
```

- [ ] **Step 3: Compute and save the nominal Jacobian**

After the loop, before `pos_sha`:

```python
    i_nom = names.index("nominal")
    e0, n0, u0 = enu[i_nom]
    d_alpha_de, d_alpha_dn, d_alpha_du, jac_valid = horizon_jacobian(
        dem, crds_all[i_nom], alpha_h[i_nom], e0, n0, u0
    )
    print(
        f"  Jacobian at nominal: {jac_valid.sum()}/{N_AZ} azimuths valid, "
        f"|d alpha/dU| median {np.median(np.abs(d_alpha_du[jac_valid])):.3e} rad/m"
    )
```

and add to the `np.savez` call:

```python
        crds=crds_all,
        dalpha_dE=d_alpha_de,
        dalpha_dN=d_alpha_dn,
        dalpha_dU=d_alpha_du,
        jac_valid=jac_valid,
```

- [ ] **Step 4: Regenerate and verify `alpha_h` is unchanged**

Back up the current file first, then regenerate:

```bash
cp horizon_position/output/horizons_position.npz /tmp/horizons_before.npz
PYTHONPATH=/home/christian/Documents/research/eigsep/eigsep_terrain \
uv run --project /home/christian/Documents/research/eigsep/eigsep_terrain \
    python horizon_position/make_horizons.py
```

Then write a check script (not `python -c`) asserting `alpha_h`, `az_grid`, `enu`, `names` and `pos_sha` are byte-identical to the backup, and that `crds`, `dalpha_dE/N/U` and `jac_valid` are present with the shapes above. Run it and confirm it passes. **`alpha_h` must be byte-identical** — only what is *kept* changed, not what is computed.

- [ ] **Step 5: Commit**

```bash
git add horizon_position/make_horizons.py
git commit -m "feat(horizon_position): emit the analytic horizon Jacobian from crds"
```

---

### Task 5: Always run clean, and use eigsim's mask

**Files:**
- Modify: `horizon_position/run_sims.py:106-179`

**Interfaces:**
- Consumes: Task 1's `eigsim.open_sky_weight`.
- Produces: `output/position_sims<tag>.npz` with the same keys as today. `pos_sha` stays in the output as a content identifier for downstream products; only the resume path goes.

- [ ] **Step 1: Delete the checkpoint/resume machinery**

Replace the batch loop (`batch_files = []` through the `for f in batch_files: f.unlink(...)` cleanup) with a straight accumulation:

```python
    t_sys_all, fgnd_all = [], []
    for i, name in enumerate(names):
        print(f"  [{i:2d}] {name:10s} simulating...")
        t0 = time.time()
        W = eigsim.open_sky_weight(alpha_h[i], az_grid, lmax)
        t_sys_all.append(
            np.asarray(
                eigsim.simulate(
                    beam_data,
                    freqs_mhz,
                    sky,
                    times_jd,
                    [0.0],
                    [0.0],
                    beam_kw={"horizon": W},
                    sky_alm=sky_alm,
                )
            )[0]
        )
        fgnd_all.append(
            np.asarray(
                eigsim.compute_fgnd(
                    beam_data,
                    freqs_mhz,
                    [0.0],
                    [0.0],
                    beam_kw={"horizon": W},
                )
            )[0]
        )
        print(f"       done in {time.time() - t0:.0f}s")

    print(f"All positions complete in {(time.time() - wall0) / 60:.1f} min")

    t_sys = np.stack(t_sys_all, axis=0)
    fgnd = np.stack(fgnd_all, axis=0)
```

Keep the two `assert` shape checks and the whole `np.savez_compressed` block as they are. Delete the trailing `for f in batch_files:` cleanup and its print.

- [ ] **Step 2: Drop the reduce_azimuth call and the mask import**

`eigsim.open_sky_weight` integrates over the phi cell, so the azimuth reduction must not happen — it would band-limit the curve twice. Remove the `reduce_azimuth` import and its call, and pass the native `alpha_h` and `az_grid` straight through. Remove the now-unused `open_sky_weight`/`mwss_grid` imports from `masks`, and the `thetas, phis` locals if nothing else uses them.

Add a comment where the reduction used to be:

```python
    # No azimuth reduction: eigsim.open_sky_weight integrates over the phi
    # cell, which *is* the band-limiting.  Reducing first would apply it
    # twice.  See eigsep_mock_analysis issue #10.
```

- [ ] **Step 3: Update the module docstring**

Amend the header comment at `run_sims.py:5-6` so it no longer promises per-position checkpointing; say results accumulate in memory and `position_sims.npz` is written once, and that an interrupted run restarts from scratch (~20 min).

- [ ] **Step 4: Smoke-test with the tag flag**

Run the existing smoke path, which uses `--output-tag` and a small `--n-times`:

```bash
uv run python horizon_position/run_sims.py --output-tag _smoke --n-times 2
```

Expected: completes, writes `output/position_sims_smoke.npz`, leaves no `pos*_batch_*.npz` behind. Then delete the smoke output.

- [ ] **Step 5: Commit**

```bash
git add horizon_position/run_sims.py
git commit -m "refactor(horizon_position): always run clean and use eigsim's mask"
```

---

### Task 6: Validate the Jacobian against the DEM

No simulations here — this is `alpha_h` against `alpha_h`, so it runs in seconds and isolates the horizon derivative from everything downstream.

**Files:**
- Create: `horizon_position/test_jacobian.py`

**Interfaces:**
- Consumes: Task 4's `dalpha_dE/N/U`, `crds`, `jac_valid` in `horizons_position.npz`.
- Produces: the switch-fraction diagnostic used by memo M004.

- [ ] **Step 1: Write the tests**

```python
"""The analytic horizon Jacobian, checked against the DEM itself.

The 19 stored positions are a ready-made finite-difference set: predict
alpha_h(nominal) + J . delta and compare with the curve the DEM actually
produced.  Accuracy must degrade with step size -- a linearization that
still looks good at 10 m is not being tested properly.
"""

from pathlib import Path

import numpy as np
import pytest

OUT = Path(__file__).resolve().parent / "output" / "horizons_position.npz"
pytestmark = pytest.mark.skipif(
    not OUT.exists(), reason="run make_horizons.py first"
)


@pytest.fixture(scope="module")
def hz():
    return np.load(OUT)


def _predict(hz, i_nom, delta):
    J = np.stack([hz["dalpha_dE"], hz["dalpha_dN"], hz["dalpha_dU"]], axis=1)
    return hz["alpha_h"][i_nom] + J @ np.asarray(delta, float)


def _rms_deg(x):
    return float(np.degrees(np.sqrt((x**2).mean())))


@pytest.mark.parametrize("axis,idx", [("x", 0), ("y", 1), ("z", 2)])
def test_linearization_improves_with_smaller_steps(hz, axis, idx):
    names = [str(n) for n in hz["names"]]
    i_nom = names.index("nominal")
    valid = hz["jac_valid"]

    errs = {}
    for step, tag in ((0.1, "0p1"), (1.0, "1")):
        name = f"{axis}_p_{tag}"
        i = names.index(name)
        delta = np.zeros(3)
        delta[idx] = step
        resid = (_predict(hz, i_nom, delta) - hz["alpha_h"][i])[valid]
        errs[step] = _rms_deg(resid)

    # the residual of a first-order model is O(step^2), so a 10x smaller
    # step must reduce it by much more than 10x
    assert errs[0.1] < errs[1.0] / 10.0


def test_derivative_sign_for_moving_up(hz):
    # Raising the antenna lowers every horizon it can see.
    valid = hz["jac_valid"]
    assert np.all(hz["dalpha_dU"][valid] < 0.0)


def test_switch_fraction_is_small_at_small_steps(hz):
    # The derivative is exact only while the same DEM pixel keeps winning.
    # Report the switch fraction; it is the validity statement memo M004
    # quotes, and it must be small where the linearization is trusted.
    names = [str(n) for n in hz["names"]]
    i_nom = names.index("nominal")
    crds = hz["crds"]

    fractions = {}
    for name in names:
        if name == "nominal":
            continue
        i = names.index(name)
        switched = np.any(crds[i] != crds[i_nom], axis=0)
        fractions[name] = float(switched.mean())

    for name, frac in sorted(fractions.items()):
        print(f"  {name:10s} argmax switched in {100 * frac:6.2f}% of azimuths")

    small = [f for n, f in fractions.items() if n.endswith("_0p1")]
    assert max(small) < 0.25
```

- [ ] **Step 2: Run them**

Run: `uv run pytest horizon_position/test_jacobian.py -v -s`
Expected: PASS, with the switch-fraction table printed.

If `test_linearization_improves_with_smaller_steps` fails, the likely cause is the half-pixel `r_min` discrepancy noted in Task 4; try `r = dz / np.tan(alpha_h)` — the radius `calc_horizon` actually used — guarding `alpha_h == 0`, and re-run. Record which form was needed.

- [ ] **Step 3: Commit**

```bash
git add horizon_position/test_jacobian.py
git commit -m "test(horizon_position): validate the horizon Jacobian against the DEM"
```

---

### Task 7: End-to-end re-run and derivative

**Files:**
- Create: `horizon_position/make_sensitivity.py`
- Modify: `horizon_position/output/position_sims.npz` (regenerated)

**Interfaces:**
- Consumes: Tasks 1, 2, 4, 5, and Task 3's recorded decision.
- Produces: `output/position_sensitivity.npz` with `dT_dE`, `dT_dN`, `dT_dU` each `(n_times, n_freqs)` K/m, `dT_deps_y`, `dT_deps_z` each `(n_times, n_freqs)` K/deg, plus `fd_dT_dE/N/U` from the 19-position finite differences for comparison.

- [ ] **Step 1: Record the Task 3 decision**

Write here which branch Task 3 selected, then implement only that branch in Step 3.

**Task 3 decision** (full evidence in
`.superpowers/sdd/2026-09-14-horizon-tilt-sensitivity-phase1/task-3-report.md`):

- **Misalignment derivatives (`dT/d(eps_y, eps_z)`): finite-difference branch.**
  The Wigner-D code `simulate()` actually runs (its private
  `_generate_rotate_dls`/`_rotate_flms`) is differentiable in the Euler
  angles (autodiff vs. central FD agree to 8.0e-11 relative), but at the
  zenith/zero-misalignment operating point `R = I` is a ZYZ gimbal lock
  (`beta = 0`), and `croissant.rotations.rotmat_to_eulerZYZ` is NumPy
  with `np.isclose` branches: not traceable, and for `eps_y` not even
  differentiable as a plain numerical function there — its Euler-angle
  output either loses the sign of `eps_y` entirely or jumps by `pi` as
  `eps_y` crosses the `np.isclose` threshold, so forward/backward finite
  differences disagree by O(1), not truncation error. Fixing this needs
  `jnp` replacements for `drive_rotation_matrix`/`rotmat_to_eulerZYZ`,
  which is out of this spike's and this plan's scope. Use central
  differences at ±0.25 deg on each of the two parameters (four
  simulations), as already planned below.
- **Position derivatives (`dT/d(E, N, U)`): `jax.jvp` through `simulate` directly.**
  `jax.jvp(f, (alpha,), (tangent,))` with `f(alpha) = simulate(...,
  beam_kw={"horizon": open_sky_weight(alpha, az, lmax)})` runs with no
  exceptions at a small test size and agrees with central finite
  differences to max relative error 3.8e-10, and with the affine
  identity (`simulate(horizon=dW) - simulate(horizon=0)`, exact because
  `t_sys` is affine in the horizon weight) to max relative error 2.1e-14
  (machine precision). Implement as `jax.jvp` through `simulate`, one
  call per axis, as sketched in Step 3 below.

- [ ] **Step 2: Re-run the 19 positions**

```bash
uv run python horizon_position/run_sims.py
```

Expected: ~20 minutes, writes `output/position_sims.npz`. This is the first run with the phi-integrated mask, the croissant frame fix (`754627c`) and croissant `v5.3.0.dev2`, so the numbers will differ from the RASTI values — which stay pinned at tag `rasti-round2-figs` and are not affected.

- [ ] **Step 3: Write the sensitivity script**

`make_sensitivity.py` builds the position derivative by pushing the stored tangent through the simulation with `jax.jvp` — one call per axis, no extra simulations:

```python
"""dT_ant/d(position) and dT_ant/d(misalignment) for the D5 forward model."""

import numpy as np
import jax

import eigsim

# Setup is the same as run_sims.py:56-101. Copy that block verbatim:
# it produces cfg, beam_data, freqs_mhz, lmax, sky, times_jd, sky_alm and
# OUTPUT_DIR. Drop only the reduce_azimuth call (removed in Task 5) and the
# mwss_grid/thetas/phis locals, which this script does not use. i_nom is
# names.index("nominal") from horizons_position.npz.

hz = np.load(OUTPUT_DIR / "horizons_position.npz")
az_grid, alpha_nom = hz["az_grid"], hz["alpha_h"][i_nom]
tangents = {
    "E": hz["dalpha_dE"],
    "N": hz["dalpha_dN"],
    "U": hz["dalpha_dU"],
}


def t_sys_of_alpha(alpha):
    W = eigsim.open_sky_weight(alpha, az_grid, lmax)
    return eigsim.simulate(
        beam_data, freqs_mhz, sky, times_jd, [0.0], [0.0],
        beam_kw={"horizon": W}, sky_alm=sky_alm,
    )[0]


for axis, tangent in tangents.items():
    _, dT = jax.jvp(t_sys_of_alpha, (alpha_nom,), (tangent,))
    out[f"dT_d{axis}"] = np.asarray(dT)
```

For the misalignment, implement the branch Task 3 selected — either `jax.jvp` through a function of `(eps_y, eps_z)` that builds the drive rotation via `misalignment_matrix`, or central differences at ±0.25 deg on each of the two parameters (four simulations).

- [ ] **Step 4: Cross-check the JVP against finite differences**

Compute `fd_dT_dU = (t_sys[z_p_0p1] - t_sys[z_m_0p1]) / 0.2` from the re-run `position_sims.npz`, and likewise for E and N. Assert the JVP and the central difference agree to better than 5 per cent in RMS across LST and frequency, and print the actual agreement. A larger disagreement means the tangent, the chain, or the sign convention is wrong — debug before proceeding, do not loosen the tolerance.

- [ ] **Step 5: Run and save**

Run: `uv run python horizon_position/make_sensitivity.py`
Expected: writes `output/position_sensitivity.npz`, prints the JVP-vs-finite-difference agreement per axis.

- [ ] **Step 6: Commit**

```bash
git add horizon_position/make_sensitivity.py
git commit -m "feat(horizon_position): dT_ant sensitivity to position and misalignment"
```

- [ ] **Step 7: Close issue #10**

The re-run in Step 2 is the measurement issue #10 deferred. Close it with a comment recording: the open-sky fraction and `dT_ant` change from the phi-integrated mask as actually measured here, that new work uses `eigsim.open_sky_weight` which has no `reduce_azimuth` to delete, and that `horizon_position/masks.py` stays as-is because the RASTI paper is pinned at `rasti-round2-figs`.

```bash
gh issue close 10 --comment "<the measurement, per above>"
```

---

## Follow-ups (not in this plan)

- `run_beam_sims.py` still uses `horizon_position/masks.py` with `reduce_azimuth`. Migrating it to `eigsim.open_sky_weight` would let `masks.py` be deleted entirely, but it regenerates `beam_sims.npz`, which feeds `beam_comparison.pdf`. Out of scope while the paper is in revision.
- Phase 2: memo M004 in `eigsep_analysis`, planned separately once these numbers exist.
