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


def _xhi_grid_band_top():
    """z grid with control nodes exactly at z=4.6816 and z=5.9, four models:

    0. reionized well before z=5.9 and stays reionized through the band top
    1. reionized by z=5.9 but *re-neutralises* below it (fails only the
       band-top check) -- the real failure mode this function guards
       against, built synthetically since it needs an xHI that rises as
       z falls
    2. not yet reionized at z=5.9 but completes reionization before the
       band top (fails only the z=5.9 check)
    3. reionized at neither redshift
    """
    z = np.array([4.0, 4.6816, 5.0, 5.9, 6.5, 10.0, 20.0, 30.0])
    both = np.array([0.0, 0.0, 0.0, 0.0, 0.05, 0.3, 0.8, 1.0])
    only_ref = np.array([0.2, 0.18, 0.1, 0.0, 0.02, 0.3, 0.8, 1.0])
    only_top = np.array([0.02, 0.02, 0.05, 0.4, 0.6, 0.9, 1.0, 1.0])
    neither = np.array([0.5, 0.5, 0.5, 0.6, 0.7, 0.9, 1.0, 1.0])
    return np.vstack([both, only_ref, only_top, neither]), z


def test_across_band_keeps_only_models_reionized_at_both_redshifts():
    xHI, z = _xhi_grid_band_top()
    assert sel.reionized_across_band(xHI, z).tolist() == [True, False, False, False]


def test_reionized_alone_is_fooled_by_re_neutralisation():
    """The model that re-neutralises below z=5.9 passes the plain cut."""
    xHI, z = _xhi_grid_band_top()
    assert sel.reionized(xHI, z).tolist() == [True, True, False, False]


def test_band_top_check_alone_would_keep_the_late_reionizer():
    xHI, z = _xhi_grid_band_top()
    keep = sel.reionized(xHI, z, z_ref=sel.Z_BAND_TOP, x_max=sel.XHI_MAX_BAND_TOP)
    assert keep.tolist() == [True, False, True, False]


def test_across_band_output_shape_and_dtype_match_reionized():
    xHI, z = _xhi_grid_band_top()
    keep = sel.reionized_across_band(xHI, z)
    assert keep.shape == (xHI.shape[0],)
    assert keep.dtype == bool


def test_across_band_thresholds_are_configurable():
    xHI, z = _xhi_grid_band_top()
    # Loosening x_max_top to 1.0 should recover the plain z=5.9 cut.
    keep = sel.reionized_across_band(xHI, z, x_max_top=1.0)
    assert keep.tolist() == sel.reionized(xHI, z).tolist()


def test_across_band_rejects_a_descending_redshift_grid():
    xHI, z = _xhi_grid_band_top()
    with pytest.raises(ValueError, match="ascending"):
        sel.reionized_across_band(xHI[:, ::-1], z[::-1])


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
