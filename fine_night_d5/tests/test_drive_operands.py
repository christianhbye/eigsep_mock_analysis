"""Exact coordinates must never silently become nearby coordinates."""

import numpy as np
import pytest
from fine_night_d5.drive_operands import DriveOperands, angle_key


def test_exact_angle_identity_and_missing_coordinate_failure():
    first = np.array([0.0, 0.8, -0.0], dtype=np.float64)
    near = first.copy()
    near[1] = np.nextafter(first[1], 1.0)
    signed = first.copy()
    signed[2] = 0.0
    assert len({angle_key(first), angle_key(near), angle_key(signed)}) == 3
    cache = object.__new__(DriveOperands)
    cache.entries = {angle_key(first): dict(beta_bits="original")}
    cache.tables = {"original": np.arange(3)}
    np.testing.assert_array_equal(cache.table(first), np.arange(3))
    with pytest.raises(ValueError, match="coverage"):
        cache.table(near)
    with pytest.raises(ValueError, match="coverage"):
        cache.table(signed)
    with pytest.raises(ValueError, match="float64"):
        angle_key(first.astype(np.float32))
    with pytest.raises(ValueError, match="finite"):
        angle_key(np.array([0.0, np.nan, 0.0]))


def test_full_mode_rotation_keeps_highest_degree_and_phase_signs():
    from fine_night_d5.drive_operands import rotate_numpy

    L = 4
    center = L - 1
    rng = np.random.default_rng(9)
    alm = rng.normal(size=(2, L, 2 * L - 1)) + 1j * rng.normal(size=(2, L, 2 * L - 1))
    dl = np.broadcast_to(np.eye(2 * L - 1), (L, 2 * L - 1, 2 * L - 1)).copy()
    m = np.arange(-center, center + 1)
    alpha, gamma = np.exp(-1j * m * 0.3), np.exp(-1j * m * 0.7)
    rotated = rotate_numpy(alm, dl, alpha, gamma)
    for ell in range(L):
        valid = abs(m) <= ell
        np.testing.assert_allclose(
            rotated[:, ell, valid],
            alm[:, ell, valid] * np.exp(-1j * m[valid]),
            rtol=1e-14,
            atol=1e-14,
        )
        np.testing.assert_array_equal(rotated[:, ell, ~valid], 0)
    assert np.any(abs(rotated[:, -1]) > 0)
    with pytest.raises(ValueError, match="complete"):
        rotate_numpy(alm.astype(np.complex64), dl, alpha, gamma)

    # A zonal dipole tilted by pi/2 becomes the two m=+/-1 modes.
    dipole = np.zeros((1, L, 2 * L - 1), dtype=np.complex128)
    dipole[0, 1, center] = 2
    d = dl.copy()
    d[1, center - 1 : center + 2, center - 1 : center + 2] = [
        [0.5, 1 / np.sqrt(2), 0.5],
        [-1 / np.sqrt(2), 0, 1 / np.sqrt(2)],
        [0.5, -1 / np.sqrt(2), 0.5],
    ]
    ones = np.ones(2 * L - 1, dtype=np.complex128)
    tilted = rotate_numpy(dipole, d, ones, ones)
    np.testing.assert_allclose(
        tilted[0, 1, center - 1 : center + 2],
        [np.sqrt(2), 0, -np.sqrt(2)],
        rtol=1e-15,
        atol=1e-15,
    )
