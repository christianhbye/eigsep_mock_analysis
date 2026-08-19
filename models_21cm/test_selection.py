"""Unit tests for post-generation selection. Main env: `uv run pytest`."""

import numpy as np
import pytest
import selection as sel


def _xhi_grid():
    """Ascending z, and two models: one reionized by z=5.9, one not."""
    z = np.linspace(5.0, 30.0, 26)
    done = np.clip((z - 6.0) / 4.0, 0.0, 1.0)  # xHI = 0 below z = 6
    late = np.clip((z - 3.0) / 4.0, 0.0, 1.0)  # xHI ~ 0.72 at z = 5.9
    return np.vstack([done, late]), z


def test_cut_keeps_reionized_and_drops_late_models():
    xHI, z = _xhi_grid()
    assert sel.reionized(xHI, z).tolist() == [True, False]


def test_cut_threshold_is_configurable():
    xHI, z = _xhi_grid()
    assert sel.reionized(xHI, z, x_max=0.9).tolist() == [True, True]


def test_cut_rejects_a_descending_redshift_grid():
    xHI, z = _xhi_grid()
    with pytest.raises(ValueError, match="ascending"):
        sel.reionized(xHI[:, ::-1], z[::-1])


def test_cut_rejects_a_reference_redshift_off_the_grid():
    xHI, z = _xhi_grid()
    with pytest.raises(ValueError, match="outside"):
        sel.reionized(xHI, z, z_ref=100.0)


def test_subsample_is_sorted_unique_and_in_range():
    idx = sel.figure_subsample(n_survivors=500, n_draw=50, seed=3)
    assert idx.size == 50
    assert np.array_equal(idx, np.unique(idx))
    assert idx.min() >= 0 and idx.max() < 500


def test_subsample_is_reproducible_for_a_seed():
    a = sel.figure_subsample(500, 50, seed=3)
    b = sel.figure_subsample(500, 50, seed=3)
    assert np.array_equal(a, b)


def test_subsample_rejects_drawing_more_than_exist():
    with pytest.raises(ValueError, match="exceeds"):
        sel.figure_subsample(n_survivors=10, n_draw=11, seed=3)
