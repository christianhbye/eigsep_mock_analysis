# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)

import croissant as cro
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.utils import iers
from eigsim.horizon import mwss_grid, open_sky_weight

from horizon_position_d5.temperature import (
    affine_response,
    apply_response,
    make_temperature_functions,
)


@pytest.fixture(autouse=True)
def offline_iers_for_synthetic_controls():
    # Keep this policy local to these 2023 synthetic times. A collection-time
    # assignment would disable fresh tables for unrelated 2026 frame tests.
    with iers.conf.set_temp("auto_download", False):
        yield


def functions(sky_temperature=None, uniform_beam=False):
    th, ph = mwss_grid(4)
    theta, phi = np.meshgrid(th, ph, indexing="ij")
    beam = (1 + 0.2 * np.cos(theta) + 0.1 * np.sin(theta) * np.cos(phi))[None]
    if uniform_beam:
        beam = np.ones_like(beam)
    sky_map = 1200 + 500 * np.cos(theta) + 200 * np.sin(theta) * np.sin(phi)
    if sky_temperature is not None:
        sky_map = np.full_like(theta, sky_temperature)
    sky = cro.Sky(sky_map[None], np.array([100.0]), sampling="mwss", coord="equatorial")
    fast, direct = make_temperature_functions(
        beam,
        np.array([100.0]),
        sky,
        np.array([2460238.7, 2460238.71]),
        np.array([0.0, 20.0]),
        np.array([0.0, 30.0]),
        ground_temperature_k=300.0,
    )
    return fast, direct, theta.shape


def test_affine_response_matches_independent_full_calls():
    fast, direct, shape = functions()
    offset, kernel = affine_response(fast, shape)
    rng = np.random.default_rng(871)
    for weight in (rng.uniform(0.1, 0.9, shape), np.ones(shape), np.zeros(shape)):
        expected = np.asarray(direct(weight))
        assert np.allclose(
            apply_response(offset, kernel, weight), expected, atol=1e-8, rtol=1e-10
        )
    assert np.linalg.norm(kernel) > 0


def test_equal_sky_ground_is_horizon_independent():
    fast, _, shape = functions(300.0)
    offset, kernel = affine_response(fast, shape)
    assert np.max(np.abs(kernel)) < 1e-10
    assert np.allclose(offset, 300.0, rtol=0, atol=1e-9)


def test_alpha_chain_jvp_and_one_sided_clip_control():
    fast, _, shape = functions()
    offset, kernel = affine_response(fast, shape)
    az = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    alpha = jnp.asarray(0.13 + 0.02 * np.sin(az))
    tangent = jnp.asarray(0.7 + 0.2 * np.cos(2 * az))

    def model(a):
        return apply_response(offset, kernel, open_sky_weight(a, az, 4, sub=8))

    _, derivative = jax.jvp(model, (alpha,), (tangent,))
    step = 1e-6
    finite = (model(alpha + step * tangent) - model(alpha - step * tangent)) / (
        2 * step
    )
    assert np.allclose(derivative, finite, atol=1e-6, rtol=1e-6)
    th, _ = mwss_grid(4)
    boundary = np.pi / 2 - (th[1] + th[2]) / 2

    def cell(a):
        return open_sky_weight(jnp.full(len(az), a), az, 4, sub=4)[2, 0]

    step = 1e-7
    left = (cell(boundary) - cell(boundary - step)) / step
    right = (cell(boundary + step) - cell(boundary)) / step
    central = (left + right) / 2
    assert abs(left - right) > 1.0
    assert float(jax.grad(cell)(boundary)) == pytest.approx(float(central), abs=1e-7)


def test_sequential_affine_adjoint_has_correct_output_order():
    coefficients = jnp.arange(24.0).reshape(2, 3, 2, 2) / 19.0
    baseline = jnp.arange(6.0).reshape(2, 3)

    def analytic(weight):
        return baseline + jnp.einsum("tfij,ij->tf", coefficients, weight)

    offset, kernel = affine_response(analytic, (2, 2), sequential=True)
    assert np.array_equal(offset, baseline)
    assert np.array_equal(kernel, coefficients)


def test_uniform_hot_sky_has_known_negative_horizon_response():
    # Uniform beam and uniform 1200 K sky above 300 K ground: only the
    # solid-angle weight changes. Compute its derivative from the cell
    # width and independent NumPy MWSS quadrature, without AD or FD.
    from s2fft.utils.quadrature import quad_weights

    fast, _, shape = functions(1200.0, uniform_beam=True)
    offset, kernel = affine_response(fast, shape)
    theta, phi = mwss_grid(4)
    edges = np.r_[0.0, (theta[1:] + theta[:-1]) / 2, np.pi]
    alpha = 0.13
    row = np.searchsorted(edges, np.pi / 2 - alpha) - 1
    width = edges[row + 1] - edges[row]
    weights = np.asarray(quad_weights(L=5, sampling="mwss"))
    expected = -(1200.0 - 300.0) * weights[row] * len(phi) / (4 * np.pi * width)
    az = np.linspace(0, 2 * np.pi, 20, endpoint=False)

    def model(a):
        mask = open_sky_weight(jnp.full(len(az), a), az, 4, sub=8)
        return apply_response(offset, kernel, mask)

    measured = np.asarray(jax.jacfwd(model)(alpha))
    assert expected < -1.0
    assert np.all(measured < -1.0)
    np.testing.assert_allclose(measured, expected, atol=1e-8, rtol=1e-10)
