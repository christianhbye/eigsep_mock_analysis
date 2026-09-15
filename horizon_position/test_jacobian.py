"""The analytic horizon Jacobian, checked against the DEM itself.

The 19 stored positions are a ready-made finite-difference set: predict
alpha_h(nominal) + J . delta and compare with the curve the DEM actually
produced. Accuracy must degrade with step size -- a linearization that
still looks good at 10 m is not being tested properly.

Two effects from the Task 4 report (task-4-report.md, concerns 2-3) shape
how that comparison has to be built:

- calc_horizon combines the float32 DEM with u0 as ``U - u0``; under NumPy's
  NEP 50 promotion that subtraction runs in float32, so the u0 that
  participates is effectively ``float32(u0)``, not the Python float. e0 and
  n0 are only ever combined with float64 pixel edges (get_en, calc_rmin,
  calc_az_bin_range in eigsep_terrain/utils.py), so they see no such
  rounding. ``_calc_horizon_delta`` below reproduces this so the O(step^2)
  residual comparisons are not contaminated by a constant ~2.4e-5 m
  rounding floor on the Up step (Task 4 report concern 2).
- calc_horizon assigns each azimuth to a single DEM pixel by argmax; the
  Jacobian differentiates that one pixel's angle and cannot see a step that
  hands the azimuth to a different pixel (documented in make_horizons.py).
  At this file's resolution, a horizontal (E/N) move of just 0.1 m already
  reassigns the winning pixel at ~27-28% of valid azimuths (Task 4 report
  point 3), and residual RMS over *all* valid azimuths mixes that O(1)
  pixel-reassignment jump with the genuine O(step^2) Taylor error. Task 4's
  own check script sidestepped this for its informational FD sanity check
  ("It skips azimuths where the pixel changes, because that adjudication is
  Task 6's job."). The tests below do the same: the Taylor-scaling checks
  restrict to azimuths where the same pixel wins at both positions being
  compared, and a separate test quantifies the mixed population instead.
"""

from pathlib import Path

import numpy as np
import pytest

OUT = Path(__file__).resolve().parent / "output" / "horizons_position.npz"
pytestmark = pytest.mark.skipif(not OUT.exists(), reason="run make_horizons.py first")


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
    such rounding (see the module docstring). Using the nominal 0.1/1/10 m
    step for dU instead of this effective, slightly smaller step would put
    a constant rounding floor into what should be an O(step^2) residual.
    """
    e_i, n_i, u_i = (float(x) for x in enu_i)
    e0, n0, u0 = (float(x) for x in enu_nom)
    du = float(np.float32(u_i)) - float(np.float32(u0))
    return np.array([e_i - e0, n_i - n0, du], dtype=float)


def _predict(hz, i_nom, delta):
    J = np.stack([hz["dalpha_dE"], hz["dalpha_dN"], hz["dalpha_dU"]], axis=1)
    return hz["alpha_h"][i_nom] + J @ np.asarray(delta, float)


def _residual(hz, i_nom, i):
    """predicted - actual alpha_h at position ``i``, effective-delta based."""
    delta = _calc_horizon_delta(hz["enu"][i], hz["enu"][i_nom])
    return _predict(hz, i_nom, delta) - hz["alpha_h"][i]


def _rms_deg(x):
    return float(np.degrees(np.sqrt((x**2).mean())))


def _same_pixel_mask(crds, i_nom, i):
    # `!=` already treats NaN as "switched": both `nan != nan` and
    # `nan != x` are True in NumPy, so an unmatched pixel (crds NaN) counts
    # as a switch without any special-casing.
    return ~np.any(crds[i] != crds[i_nom], axis=0)


@pytest.mark.parametrize("axis,idx", [("x", 0), ("y", 1), ("z", 2)])
def test_linearization_improves_with_smaller_steps(hz, names, axis, idx):
    """O(step^2) Taylor scaling, isolated from DEM pixel reassignment.

    Restricted to azimuths where the same DEM pixel wins at nominal and at
    the shifted position (see module docstring) -- exactly the domain the
    Jacobian is documented to be exact on. `idx` is unused directly; it
    exists so the parametrization mirrors the (axis, column) pairing used
    elsewhere in this file.

    Measured RMS residual [deg], '+' direction, same-pixel azimuths only:
      x: 0.1m=8.67e-07  1m=1.39e-04  10m=1.66e-02  ratio(1/.1)=160  ratio(10/1)=119
      y: 0.1m=6.00e-07  1m=3.50e-05  10m=2.16e-03  ratio(1/.1)=58   ratio(10/1)=62
      z: 0.1m=1.38e-06  1m=1.38e-04  10m=1.38e-02  ratio(1/.1)=100  ratio(10/1)=100
    The tightest margin (y) is still ~2.9x the 10x bound and ~3.1x the 20x
    bound asserted below.
    """
    i_nom = names.index("nominal")
    valid = hz["jac_valid"]
    crds = hz["crds"]

    errs = {}
    for step, tag in ((0.1, "0p1"), (1.0, "1"), (10.0, "10")):
        name = f"{axis}_p_{tag}"
        i = names.index(name)
        mask = valid & _same_pixel_mask(crds, i_nom, i)
        errs[step] = _rms_deg(_residual(hz, i_nom, i)[mask])

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


# Measured switched-azimuth resid_sw / dalpha_sw at the 0.1 m step:
#   x_p_0p1=0.980  x_m_0p1=0.978  y_p_0p1=1.019  y_m_0p1=1.020
#   z_p_0p1=0.134  z_m_0p1=0.146
# For z the switched-azimuth residual is a small fraction (<0.3, with a
# ~2x margin) of the switched-azimuth horizon change: switches there are
# mostly near-ties and the Jacobian, evaluated at the old pixel, still
# predicts the new value reasonably. For x/y the ratio is ~1.0: a
# horizontal 0.1 m move that reassigns the winning pixel typically hands
# it to an unrelated ridge, not a near-tied neighbor, so the residual is
# essentially the *entire* horizon jump, not a small correction -- i.e.
# switches DO break the linearization for horizontal moves at this step.
# That is expected (the Jacobian differentiates one fixed pixel's angle
# and cannot see a reassignment) but it is not "small", so no passing
# bound is asserted for x/y; they are marked as known (strict) failures
# instead of inventing one. See task-6-report.md.
_SWITCHED_RESIDUAL_BOUND = 0.3


def _xfail_switch_breaks_linearization(measured_ratio):
    return pytest.mark.xfail(
        strict=True,
        reason=(
            "horizontal 0.1 m step: switched-azimuth residual is not a "
            f"small fraction of the horizon change (measured ratio "
            f"{measured_ratio:.3f} vs bound {_SWITCHED_RESIDUAL_BOUND}); "
            "see task-6-report.md"
        ),
    )


@pytest.mark.parametrize(
    "pos_name",
    [
        pytest.param("x_p_0p1", marks=_xfail_switch_breaks_linearization(0.980)),
        pytest.param("x_m_0p1", marks=_xfail_switch_breaks_linearization(0.978)),
        pytest.param("y_p_0p1", marks=_xfail_switch_breaks_linearization(1.019)),
        pytest.param("y_m_0p1", marks=_xfail_switch_breaks_linearization(1.020)),
        "z_p_0p1",
        "z_m_0p1",
    ],
)
def test_switched_azimuth_residual_vs_horizon_change_at_0p1m(hz, names, pos_name):
    """Does a pixel switch break the linearization at the 0.1 m step?

    Prints, for this position, the switch fraction plus the residual RMS
    and the RMS horizon change, each split over switched vs. unswitched
    valid azimuths -- the diagnostic memo M004 quotes. See the module-level
    comment above for the measured numbers and why x/y are xfail.
    """
    i_nom = names.index("nominal")
    i = names.index(pos_name)
    valid = hz["jac_valid"]
    crds = hz["crds"]

    switched = ~_same_pixel_mask(crds, i_nom, i)
    sw = valid & switched
    un = valid & ~switched
    assert sw.any() and un.any()  # both populations are non-empty here

    resid = _residual(hz, i_nom, i)
    dalpha = hz["alpha_h"][i] - hz["alpha_h"][i_nom]
    resid_sw, resid_un = _rms_deg(resid[sw]), _rms_deg(resid[un])
    dalpha_sw, dalpha_un = _rms_deg(dalpha[sw]), _rms_deg(dalpha[un])

    print(
        f"  {pos_name:10s} switch_frac(valid)={100 * sw.sum() / valid.sum():6.2f}%  "
        f"resid_sw={resid_sw:.3e} deg  resid_un={resid_un:.3e} deg  "
        f"dalpha_sw={dalpha_sw:.3e} deg  dalpha_un={dalpha_un:.3e} deg  "
        f"ratio_sw={resid_sw / dalpha_sw:.4f}  n_sw={sw.sum()}  n_un={un.sum()}"
    )

    assert resid_sw < _SWITCHED_RESIDUAL_BOUND * dalpha_sw
