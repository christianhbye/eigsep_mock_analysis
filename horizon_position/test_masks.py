import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from masks import (  # noqa: E402
    N_AZ_MASK,
    boolean_weight,
    mwss_grid,
    open_sky_weight,
    reduce_azimuth,
)

LMAX = 128
N_AZ = 720


def _az_grid(n=N_AZ):
    return np.linspace(0.0, 2 * np.pi, n, endpoint=False)


def test_grid_shapes():
    thetas, phis = mwss_grid(LMAX)
    assert thetas.shape == (LMAX + 2,)  # MWSS ntheta = L + 1 = lmax + 2
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
    assert d_small > 0.0  # sub-pixel shift registers (no floor)
    assert d_small < d_big  # and scales with the shift size


def test_frame_mapping_blocks_east():
    # horizon high only near az=90deg (East) must reduce open sky near phi=0
    # (phi=0 is ENU East in croissant), not near phi=90deg.
    thetas, phis = mwss_grid(LMAX)
    az = _az_grid()
    alpha = np.deg2rad(40.0) * np.exp(-((az - np.pi / 2) ** 2) / (2 * 0.1**2))
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


def test_reduce_azimuth_averages_into_bins():
    n_in = N_AZ_MASK * 8
    az = _az_grid(n_in)
    alpha = np.arange(n_in, dtype=float)
    red, az_red = reduce_azimuth(alpha, az)
    assert red.shape == (N_AZ_MASK,)
    assert az_red.shape == (N_AZ_MASK,)
    # each output bin is the mean of its 8 inputs
    assert np.allclose(red, alpha.reshape(N_AZ_MASK, 8).mean(axis=1))
    # the reduced grid is the input grid's every-8th left edge
    assert np.allclose(az_red, az[::8])


def test_reduce_azimuth_keeps_leading_axes():
    n_in = N_AZ_MASK * 4
    alpha = np.random.default_rng(0).normal(size=(19, n_in))
    red, _ = reduce_azimuth(alpha, _az_grid(n_in))
    assert red.shape == (19, N_AZ_MASK)
    assert np.allclose(red[3], alpha[3].reshape(N_AZ_MASK, 4).mean(axis=1))


def test_reduce_azimuth_is_a_noop_at_the_target_length():
    az = _az_grid(N_AZ_MASK)
    alpha = np.random.default_rng(2).normal(size=N_AZ_MASK)
    red, az_red = reduce_azimuth(alpha, az)
    assert np.array_equal(red, alpha)
    assert np.array_equal(az_red, az)


def test_reduce_azimuth_rejects_a_non_multiple():
    import pytest

    with pytest.raises(ValueError):
        reduce_azimuth(np.zeros(1000), _az_grid(1000))


def test_reducing_tames_the_aliasing_open_sky_weight_would_see():
    """A spike one native bin wide must not survive point-sampling intact."""
    thetas, phis = mwss_grid(LMAX)
    n_in = N_AZ_MASK * 64
    alpha = np.zeros(n_in)
    alpha[n_in // 3] = np.deg2rad(2.7)  # one-bin cliff spike
    red, az_red = reduce_azimuth(alpha, _az_grid(n_in))
    # the spike's weight is spread over its bin, not kept at full height
    assert np.isclose(red.max(), np.deg2rad(2.7) / 64)
    # and the mask built from the reduced curve is finite and in range
    W = open_sky_weight(red, az_red, thetas, phis)
    assert np.all((W >= 0.0) & (W <= 1.0))
