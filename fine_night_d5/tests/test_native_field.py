"""Physical projection/gauge checks without a sky simulator."""

import numpy as np
import pytest
from fine_night_d5.native_field import harmonic_power, transverse_power


def test_radial_invariance_normalization_and_gain_gauge():
    r = np.array(
        [
            [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, -1.0],
        ]
    )
    w = np.full(6, 4 * np.pi / 6)
    rng = np.random.default_rng(12)
    e = rng.normal(size=(2, 3, 6)) + 1j * rng.normal(size=(2, 3, 6))
    p, norm = transverse_power(e, r, w)
    p2, norm2 = transverse_power(e + 17j * r[None], r, w)
    np.testing.assert_allclose(p, p2, atol=1e-14)
    np.testing.assert_allclose(norm, norm2, atol=1e-14)
    np.testing.assert_allclose(p @ w, 4 * np.pi)
    arm = np.cross(r.T, np.roll(r, 1, axis=0).T).T
    ep = e - (e * r[None]).sum(1)[:, None] * r[None]
    coupling = abs((e * arm[None]).sum(1)) ** 2
    projected = abs((ep / np.sqrt(norm)[:, None, None] * arm[None]).sum(1)) ** 2
    np.testing.assert_allclose(projected * norm[:, None], coupling)
    yh = np.eye(6)
    ch = e.reshape(2, -1)
    np.testing.assert_allclose(harmonic_power(ch, yh, r, w)[0], p)


def test_invalid_grid_and_zero_field_rejected():
    r = np.eye(3)
    w = np.full(3, 4 * np.pi / 3)
    for fields, weights in [(np.zeros((1, 3, 3)), w), (np.ones((1, 3, 3)), w / 2)]:
        with pytest.raises(ValueError):
            transverse_power(fields, r, weights)
