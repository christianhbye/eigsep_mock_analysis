import numpy as np
import pytest
from eigsep_pointing_sim.drive import (
    uniform_drive,
    uniform_micro_means,
    uniform_window_means,
)


def test_uniform_drive_and_partial_window_integral():
    moves = [(0.25, 1.75, 4.0), (2.25, 3.75, -4.0)]
    values = uniform_drive([0, 0.25, 1, 1.75, 2, 2.25, 3, 3.75, 4], moves, y0=-3)
    np.testing.assert_allclose(values, [-3, -3, 0, 3, 3, 3, 0, -3, -3])
    means = uniform_window_means(np.arange(5.0), moves, y0=-3)
    # First interval's advance integral is 4 * 0.75**2 / 2.
    np.testing.assert_allclose(means, [-1.875, 1.875, 1.875, -1.875])


def test_exact_micro_means_preserve_window_identity_at_32_and_64():
    edges = np.arange(31) * 0.537
    moves = [(0.381, 6.172, 5.341), (8.081, 13.872, -5.341)]
    target = uniform_window_means(edges, moves, y0=-15)
    for n_sub in [32, 64]:
        fine = uniform_micro_means(edges, moves, y0=-15, n_sub=n_sub)
        np.testing.assert_allclose(
            fine["y_mean"].reshape(-1, n_sub).mean(1), target, rtol=0, atol=1e-11
        )
        assert np.diff(fine["time"]).min() > 0
        # Fixed midpoint sampling is not an exact quadrature next to a break.
        midpoint = uniform_drive(fine["time"], moves, y0=-15).reshape(-1, n_sub).mean(1)
        assert np.max(np.abs(midpoint - target)) > 1e-6


def test_integrals_are_stable_at_unix_epoch_and_validate_moves():
    edges = np.arange(8.0)
    moves = np.array([(0.25, 1.75, 4), (3.25, 4.75, -4)])
    offset = 1_784_320_000.0
    shifted = moves.copy()
    shifted[:, :2] += offset
    np.testing.assert_allclose(
        uniform_window_means(edges + offset, shifted, y0=-3),
        uniform_window_means(edges, moves, y0=-3),
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="overlap"):
        uniform_drive([0, 1, 2], [(0, 2, 1), (1, 3, -1)])
    with pytest.raises(ValueError, match="increasing"):
        uniform_window_means([0, 2, 1], moves)
