"""Independent frame/sign controls for the hypotheses compared in the study."""

import numpy as np
import pytest

from horizon_position_d5.candidates import position_tangents


def test_flat_horizon_vertical_and_radial_signs():
    beta = np.arange(0.0, 360.0, 90.0)
    alpha = np.full((1, 4), 0.2)
    distance = np.full_like(alpha, 100.0)
    gradient = np.zeros((1, 4, 2))
    j = position_tangents(alpha, distance, gradient, beta, "e7")
    radial = np.sin(0.2) * np.cos(0.2) / 100 + np.cos(0.2) ** 2 / 6371e3
    assert j[0, 0, 0] == pytest.approx(radial)
    assert j[0, 1, 1] == pytest.approx(radial)
    assert j[0, 2, 0] == pytest.approx(-radial)
    assert j[0, 3, 1] == pytest.approx(-radial)
    assert np.allclose(j[..., 2], -(np.cos(0.2) ** 2) / 100)


def test_e7_periodic_angular_slope_has_correct_sign_and_units():
    beta = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
    alpha = (0.3 + 0.02 * np.sin(beta))[None]
    distance = np.full_like(alpha, 100.0)
    j = position_tangents(
        alpha, distance, np.zeros((1, len(beta), 2)), np.degrees(beta), "e7"
    )
    across = j[..., 0] * np.sin(beta) - j[..., 1] * np.cos(beta)
    assert np.allclose(across, -0.02 * np.cos(beta) / 100, atol=2e-10, rtol=0)


def test_e7_prime_recovers_cross_component_of_dem_gradient():
    beta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    alpha = np.full((2, len(beta)), 0.25)
    distance = np.full_like(alpha, 120.0)
    gradient = np.broadcast_to([0.4, -0.7], (*alpha.shape, 2))
    j = position_tangents(alpha, distance, gradient, np.degrees(beta), "e7prime")
    across = j[..., 0] * np.sin(beta) - j[..., 1] * np.cos(beta)
    expected = np.cos(0.25) ** 2 * (0.4 * np.sin(beta) + 0.7 * np.cos(beta)) / 120
    assert np.allclose(across, expected, atol=1e-15, rtol=0)


def test_rejects_irregular_bearings_and_unknown_formula():
    args = (np.zeros((1, 4)), np.ones((1, 4)), np.zeros((1, 4, 2)))
    with pytest.raises(ValueError, match="full periodic"):
        position_tangents(*args, [0, 45, 90, 180], "e7")
    with pytest.raises(ValueError, match="method"):
        position_tangents(*args, [0, 90, 180, 270], "wrong")
