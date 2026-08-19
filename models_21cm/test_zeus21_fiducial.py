"""Regression test that we drive Zeus21 correctly.

Generator env only:
    uv run --project models_21cm pytest models_21cm/test_zeus21_fiducial.py -v

Values measured 2026-08-19 at Zeus21 9f2d210, precisionboost=1, zmin=4.65,
Pop III and Lyman-Werner feedback on. Tolerances are loose enough for BLAS
differences and tight enough that a physics change trips them.
"""

import numpy as np
import pytest

NU21 = 1420.405751


@pytest.fixture(scope="module")
def fiducial():
    import zeus21

    user = zeus21.User_Parameters(precisionboost=1.0)
    cosmo_in = zeus21.Cosmo_Parameters_Input(
        USE_RELATIVE_VELOCITIES=True, zmin_CLASS=4.5
    )
    classy = zeus21.runclass(cosmo_in)
    cosmo = zeus21.Cosmo_Parameters(user, cosmo_in, classy)
    # Populates ClassCosmo.pars['xi_RR_CF']; Pop III raises KeyError without it.
    zeus21.Correlations(user, cosmo, classy)
    hmf = zeus21.HMF_interpolator(user, cosmo, classy)
    astro = zeus21.Astro_Parameters(user, cosmo, USE_POPIII=True, USE_LW_FEEDBACK=True)
    return zeus21.get_T21_coefficients(user, cosmo, classy, astro, hmf, zmin=4.65)


def test_native_grid_covers_the_paper_band(fiducial):
    z = fiducial.zintegral
    nu = NU21 / (1.0 + z)
    assert z.size == 102
    assert nu.min() == pytest.approx(39.4557, abs=1e-3)
    assert nu.max() == pytest.approx(251.3992, abs=1e-3)
    # The 250 MHz endpoint must sit strictly inside the domain.
    assert nu.max() > 250.0


def test_fiducial_trough_depth_and_position(fiducial):
    T = fiducial.T21avg
    nu = NU21 / (1.0 + fiducial.zintegral)
    i = int(np.argmin(T))
    assert T[i] == pytest.approx(-109.8248, rel=1e-3)
    assert nu[i] == pytest.approx(68.9350, rel=1e-3)


def test_fiducial_emission_peak(fiducial):
    T = fiducial.T21avg
    nu = NU21 / (1.0 + fiducial.zintegral)
    assert T.max() == pytest.approx(21.1001, rel=1e-3)
    assert nu[int(np.argmax(T))] == pytest.approx(129.8680, rel=1e-3)


def test_signal_is_finite_everywhere(fiducial):
    assert np.isfinite(fiducial.T21avg).all()


def test_fiducial_model_is_reionized_by_the_top_of_the_band(fiducial):
    """T21 -> 0 at 251 MHz, which is why padding zeros there is not safe
    in general: only *reionized* models satisfy it."""
    assert fiducial.T21avg[0] == pytest.approx(0.0, abs=1e-6)


def test_neutral_fraction_is_exposed_and_complete(fiducial):
    """selection.reionized() depends on this attribute existing."""
    xHI = fiducial.xHI_avg
    assert np.isfinite(xHI).all()
    assert np.interp(5.9, fiducial.zintegral, xHI) == pytest.approx(0.0, abs=1e-4)
    assert xHI.max() > 0.99
