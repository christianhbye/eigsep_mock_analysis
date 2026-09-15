"""The analytic horizon Jacobian, checked against the DEM and against W.

`horizons_position.npz` (Task 4, fix round 1, commit c70d09b) stores two
different derivatives of the horizon curve, and this file validates each
against the observable it is actually good for:

1. **Pointwise, against `alpha_h` itself**: `dalpha_dE_pixel`,
   `dalpha_dN_pixel`, `dalpha_dU` differentiate one fixed DEM pixel's angle
   ``arctan2(dz, r_min)`` and hold that pixel fixed. `alpha_h(az)` is
   *piecewise constant* (`calc_horizon` assigns one `hor_ang` value per
   winning pixel and copies it across every azimuth bin the pixel's
   footprint overlaps), so this is the *only* population where comparing
   predicted vs. actual `alpha_h` pointwise means anything: azimuths where
   the winning pixel changes between the two positions being compared are
   a jump the pixel partials cannot see by construction, and a horizontal
   move translates the curve's step edges, which a pointwise comparison
   sees as an O(1) discontinuity regardless of how small the step is (see
   task-6-report.md's "Parallax hypothesis test" section for the
   measurements that pinned this down). `test_linearization_improves_
   with_smaller_steps` therefore restricts to azimuths where the same
   pixel wins at both positions being compared.

2. **In W-space, against `eigsim.open_sky_weight`**: `dalpha_dE`,
   `dalpha_dN` (no `_pixel` suffix) are *totals* -- the pixel partial
   minus the azimuthal-parallax term `alpha_h'(az) * d(az_p)/d(e0,n0)`,
   accounting for the fact that a horizontal move also shifts *where*
   (at which azimuth) a fixed terrain point is seen. `dalpha_dU` is
   unchanged (`d az/d u0 = 0`). This total is what `run_sims.py` feeds the
   simulation, and it is a first-order translation of the piecewise-
   constant curve: not meaningful pointwise, but exactly what a
   cell-integrating, linearly-interpolating consumer like
   `eigsim.open_sky_weight` needs (confirmed empirically in
   task-6-report.md: the pixel-only tangent is off by 57-99% on the
   solid-angle-weighted open-sky fraction, the total tangent by <3%).
   `test_total_tangent_matches_open_sky_fraction` checks this with
   `jax.jvp` through `open_sky_weight`, and
   `test_pixel_only_tangent_is_badly_wrong_on_w` is a guard so nobody
   silently drops the parallax term and regresses to the pixel-only
   behaviour without a test noticing.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import time
from pathlib import Path

import jax
import numpy as np
import pytest

import eigsim

OUT = Path(__file__).resolve().parent / "output" / "horizons_position.npz"
pytestmark = pytest.mark.skipif(not OUT.exists(), reason="run make_horizons.py first")

W_LMAX = 128


@pytest.fixture(scope="module")
def hz():
    return np.load(OUT)


@pytest.fixture(scope="module")
def names(hz):
    return [str(n) for n in hz["names"]]


def _calc_horizon_delta(enu_i, enu_nom):
    """(dE, dN, dU) as ``calc_horizon`` actually saw between two positions.

    ``calc_horizon`` takes e0, n0, u0 as plain Python floats, but only u0
    ever meets float32 data: the base case computes ``U - u0`` where U is
    the float32 DEM, and NEP 50 promotion carries that out in float32 --
    u0 is effectively rounded to float32 before the subtraction. e0 and n0
    are combined only with float64 pixel edges throughout, so they see no
    such rounding. Using the nominal 0.1/1/10 m step for dU instead of
    this effective, slightly smaller step would put a constant rounding
    floor into what should be an O(step^2) residual.
    """
    e_i, n_i, u_i = (float(x) for x in enu_i)
    e0, n0, u0 = (float(x) for x in enu_nom)
    du = float(np.float32(u_i)) - float(np.float32(u0))
    return np.array([e_i - e0, n_i - n0, du], dtype=float)


def _rms_deg(x):
    return float(np.degrees(np.sqrt((x**2).mean())))


def _same_pixel_mask(crds, i_nom, i):
    # `!=` already treats NaN as "switched": both `nan != nan` and
    # `nan != x` are True in NumPy, so an unmatched pixel (crds NaN) counts
    # as a switch without any special-casing.
    return ~np.any(crds[i] != crds[i_nom], axis=0)


# ---------------------------------------------------------------------------
# 1. Pointwise: pixel partials against alpha_h, same-winning-pixel azimuths.
# ---------------------------------------------------------------------------


def _predict_pixel(hz, i_nom, delta):
    """First-order alpha_h prediction from the *pixel* partials (pointwise)."""
    J = np.stack(
        [hz["dalpha_dE_pixel"], hz["dalpha_dN_pixel"], hz["dalpha_dU"]], axis=1
    )
    return hz["alpha_h"][i_nom] + J @ np.asarray(delta, float)


def _residual_pixel(hz, i_nom, i):
    delta = _calc_horizon_delta(hz["enu"][i], hz["enu"][i_nom])
    return _predict_pixel(hz, i_nom, delta) - hz["alpha_h"][i]


@pytest.mark.parametrize("axis,idx", [("x", 0), ("y", 1), ("z", 2)])
def test_linearization_improves_with_smaller_steps(hz, names, axis, idx):
    """O(step^2) Taylor scaling of the *pixel* partials on alpha_h.

    Restricted to azimuths where the same DEM pixel wins at nominal and at
    the shifted position -- the only domain where a pointwise comparison
    of `alpha_h` against a first-order prediction is meaningful (module
    docstring). `idx` is unused directly; it exists so the parametrization
    mirrors the (axis, column) pairing used elsewhere in this file.

    Measured RMS residual [deg], '+' direction, same-pixel azimuths only:
      x: 0.1m=8.67e-07  1m=1.39e-04  10m=1.66e-02  ratio(1/.1)=160  ratio(10/1)=119
      y: 0.1m=6.00e-07  1m=3.50e-05  10m=2.16e-03  ratio(1/.1)=58   ratio(10/1)=62
      z: 0.1m=1.38e-06  1m=1.38e-04  10m=1.38e-02  ratio(1/.1)=100  ratio(10/1)=100
    The tightest margin (y) is still ~2.9x the 10x bound and ~3.1x the 20x
    bound asserted below. (Unchanged from before the totals/pixel split:
    `dalpha_dE_pixel`/`dalpha_dN_pixel` are byte-identical to the old
    `dalpha_dE`/`dalpha_dN`.)
    """
    i_nom = names.index("nominal")
    valid = hz["jac_valid"]
    crds = hz["crds"]

    errs = {}
    for step, tag in ((0.1, "0p1"), (1.0, "1"), (10.0, "10")):
        name = f"{axis}_p_{tag}"
        i = names.index(name)
        mask = valid & _same_pixel_mask(crds, i_nom, i)
        errs[step] = _rms_deg(_residual_pixel(hz, i_nom, i)[mask])

    print(
        f"  axis={axis} same-pixel RMS residual [deg]: "
        f"0.1m={errs[0.1]:.3e}  1m={errs[1.0]:.3e}  10m={errs[10.0]:.3e}  "
        f"ratio(1/0.1)={errs[1.0] / errs[0.1]:.1f}  "
        f"ratio(10/1)={errs[10.0] / errs[1.0]:.1f}"
    )

    # the residual of a first-order model is O(step^2): a 10x smaller step
    # must reduce it by much more than 10x (brief's original bound)...
    assert errs[0.1] < errs[1.0] / 10.0
    # ...and a 10x larger step must visibly break it, not just degrade a
    # little (T6-c: "a test that does not break at 10 m is testing the
    # wrong thing"). Both bounds sit well under the smallest measured
    # margin (y's 58x / 62x) above.
    assert errs[10.0] > errs[1.0] * 20.0


def test_derivative_sign_for_moving_up(hz):
    # Raising the antenna lowers every horizon it can see.
    valid = hz["jac_valid"]
    assert np.all(hz["dalpha_dU"][valid] < 0.0)


def test_switch_fraction_table(hz, names):
    """Per-position argmax-switch fraction (memo M004 quotes this table)."""
    i_nom = names.index("nominal")
    crds = hz["crds"]

    fractions = {}
    for name in names:
        if name == "nominal":
            continue
        i = names.index(name)
        fractions[name] = float((~_same_pixel_mask(crds, i_nom, i)).mean())

    for name, frac in sorted(fractions.items()):
        print(f"  {name:10s} argmax switched in {100 * frac:6.2f}% of azimuths")

    # A bigger move must never switch the winning pixel *less* often than a
    # smaller move along the same axis. Measured (average of +/-):
    #   x: 28.1% / 88.7% / 98.7%   y: 27.0% / 89.9% / 98.7%
    #   z:  0.6% /  3.5% / 27.0%
    for axis in ("x", "y", "z"):
        f_0p1 = 0.5 * (fractions[f"{axis}_p_0p1"] + fractions[f"{axis}_m_0p1"])
        f_1 = 0.5 * (fractions[f"{axis}_p_1"] + fractions[f"{axis}_m_1"])
        f_10 = 0.5 * (fractions[f"{axis}_p_10"] + fractions[f"{axis}_m_10"])
        assert f_0p1 < f_1 < f_10


# ---------------------------------------------------------------------------
# 2. W-space: the total tangent against eigsim.open_sky_weight.
# ---------------------------------------------------------------------------


def _w_fn(az_grid):
    return lambda a: eigsim.open_sky_weight(a, az_grid, W_LMAX)


@pytest.fixture(scope="module")
def w_nom(hz, names):
    """W(alpha_h[nominal]) plus the solid-angle weight, computed once."""
    i_nom = names.index("nominal")
    t0 = time.perf_counter()
    W_fn = _w_fn(hz["az_grid"])
    W = np.asarray(W_fn(hz["alpha_h"][i_nom]))
    thetas, _ = eigsim.mwss_grid(W_LMAX)
    w = np.sin(thetas)  # solid-angle weight per theta ring
    denom = float((np.ones_like(W) * w[:, None]).sum())
    dt = time.perf_counter() - t0
    print(f"\n  W(nominal) computed in {dt:.2f}s, shape {W.shape}")
    return {"W_fn": W_fn, "W": W, "w": w, "denom": denom}


def _frac_change(dW, w, denom):
    return float((dW * w[:, None]).sum() / denom)


# Bound for the total tangent's solid-angle-weighted open-sky-fraction error
# (Task 4's fix report, task-4-report.md "Fix round 1", err_stored column;
# T6-b effective deltas):
#   x: 0.138%/0.662% at +/-0.1 m, 0.468%/0.599% at +/-1 m
#   y: 2.868%/0.752% at +/-0.1 m, 0.623%/0.776% at +/-1 m
#   z: 0.018%/0.029% at +/-0.1 m, 0.742%/0.339% at +/-1 m
# Worst case is y_p_0p1 at 2.868%; 10% gives a ~3.5x margin.
W_FRAC_BOUND = 0.10

_W_POSITIONS = [
    f"{axis}_{sgn}_{tag}"
    for axis in ("x", "y", "z")
    for sgn in ("p", "m")
    for tag in ("0p1", "1")
]


@pytest.mark.parametrize("name", _W_POSITIONS)
def test_total_tangent_matches_open_sky_fraction(hz, names, w_nom, name):
    """jax.jvp of open_sky_weight along the total tangent vs. the true W.

    The total tangent (dalpha_dE/dN plus dalpha_dU) is what run_sims.py
    feeds the simulation (Task 7). Checked on the solid-angle-weighted
    open-sky fraction, the scalar closest to what t_ant sees.
    """
    i_nom = names.index("nominal")
    i = names.index(name)
    delta = _calc_horizon_delta(hz["enu"][i], hz["enu"][i_nom])

    alpha_nom = hz["alpha_h"][i_nom]
    dE, dN, dU = hz["dalpha_dE"], hz["dalpha_dN"], hz["dalpha_dU"]
    tangent = dE * delta[0] + dN * delta[1] + dU * delta[2]
    W_true = np.asarray(w_nom["W_fn"](hz["alpha_h"][i]))
    dW_true = W_true - w_nom["W"]
    _, dW_lin = jax.jvp(w_nom["W_fn"], (alpha_nom,), (tangent,))
    dW_lin = np.asarray(dW_lin)

    frac_true = _frac_change(dW_true, w_nom["w"], w_nom["denom"])
    frac_lin = _frac_change(dW_lin, w_nom["w"], w_nom["denom"])
    rel_err = abs(frac_lin - frac_true) / abs(frac_true)

    print(
        f"  {name:8s} frac_true={frac_true:+.4e}  frac_lin={frac_lin:+.4e}  "
        f"rel_err={100 * rel_err:6.3f}%"
    )
    assert rel_err < W_FRAC_BOUND


# Guard: if the parallax term is ever silently dropped and dalpha_dE/dN
# regress to the pixel-only partials, this must fail loudly. Measured
# (Task 4's fix report, err_pixel_only column, same T6-b deltas):
#   x_m_0p1 = 59.412% (smallest of the six x cases in +/-0.1/1 m)
#   y_p_0p1 = 98.967% (smallest of the six y cases in +/-0.1/1 m)
# 40%/80% bounds give margins of ~1.5x / ~1.1x over those worst cases while
# staying well clear of the total tangent's <3% (W_FRAC_BOUND=0.10) --
# unambiguously "badly wrong", not a close call.
_PIXEL_ONLY_GUARD = {"x": ("x_m_0p1", 0.40), "y": ("y_p_0p1", 0.80)}


@pytest.mark.parametrize("axis", ["x", "y"])
def test_pixel_only_tangent_is_badly_wrong_on_w(hz, names, w_nom, axis):
    """The pixel-only tangent (no parallax term) is not a usable W-space
    approximation for a horizontal move -- nobody should drop the parallax
    term without a test noticing."""
    name, bound = _PIXEL_ONLY_GUARD[axis]
    i_nom = names.index("nominal")
    i = names.index(name)
    delta = _calc_horizon_delta(hz["enu"][i], hz["enu"][i_nom])

    alpha_nom = hz["alpha_h"][i_nom]
    tangent_pixel = (
        hz["dalpha_dE_pixel"] * delta[0]
        + hz["dalpha_dN_pixel"] * delta[1]
        + hz["dalpha_dU"] * delta[2]
    )
    W_true = np.asarray(w_nom["W_fn"](hz["alpha_h"][i]))
    dW_true = W_true - w_nom["W"]
    _, dW_lin = jax.jvp(w_nom["W_fn"], (alpha_nom,), (tangent_pixel,))
    dW_lin = np.asarray(dW_lin)

    frac_true = _frac_change(dW_true, w_nom["w"], w_nom["denom"])
    frac_lin = _frac_change(dW_lin, w_nom["w"], w_nom["denom"])
    rel_err = abs(frac_lin - frac_true) / abs(frac_true)

    print(
        f"  {name:8s} pixel-only rel_err={100 * rel_err:6.3f}%  "
        f"(bound {100 * bound:.0f}%)"
    )
    assert rel_err > bound
