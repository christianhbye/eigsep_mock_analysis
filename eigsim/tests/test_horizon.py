import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from eigsim.horizon import mwss_grid, open_sky_weight  # noqa: E402

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
