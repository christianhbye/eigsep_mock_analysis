"""Manufactured slopes distinguish arithmetic, convergence and missing work."""

import numpy as np
import pytest

from horizon_position_d5.convergence import (
    Criteria,
    difference_summary,
    evaluate_stencil,
)


def stencil(model=None, cells=2):
    h = np.array([0.01, 0.003, 0.001])
    center = np.full((3, 3, cells), 100.0)
    shifted = np.empty((3, 3, 3, 3, 2, cells))
    for j, step in enumerate(h):
        for a in range(3):
            for s, sign in enumerate([-1, 1]):
                delta = (
                    (a + 1) * sign * step if model is None else model(a, sign * step)
                )
                shifted[:, :, j, a, s] = 100 + delta
    return (
        center,
        shifted,
        dict(
            bearing_spacing_deg=[0.1, 0.05, 0.025],
            mask_subsamples=[180, 360, 720],
            steps_m=h,
            axes=["E", "N", "U"],
            signs=[-1, 1],
            cell_coordinates={"sample_id": [f"sample-{i}" for i in range(cells)]},
            center_present=np.ones(center.shape, dtype=bool),
            displaced_present=np.ones(shifted.shape, dtype=bool),
            edge_limited=np.zeros(cells, dtype=bool),
        ),
    )


def test_linear_slopes_and_constant_temperature():
    for model in (None, lambda a, d: 0):
        center, shifted, kw = stencil(model)
        result = evaluate_stencil(center, shifted, **kw)
        assert np.all(result["forward_status"] == "pass")
        assert np.all(result["behavior"] == "common_slope")
        slopes = np.arange(1, 4) if model is None else np.zeros(3)
        expected = np.broadcast_to(slopes[None, None, None, :, None], (3, 3, 3, 3, 2))
        for side in ("minus", "plus", "central"):
            np.testing.assert_allclose(result[f"{side}_k_per_m"], expected, atol=2e-11)


def test_cusp_keeps_two_limits_and_rejects_common_slope():
    center, shifted, kw = stencil(lambda a, d: abs(d) if a == 0 else (a + 1) * d)
    result = evaluate_stencil(center, shifted, **kw)
    assert np.all(result["forward_status"] == "pass")
    assert np.all(result["one_sided_status"] == "pass")
    assert np.all(result["behavior"][0] == "distinct_one_sided")
    assert np.all(result["common_slope_status"][0] == "fail")
    assert np.all(result["common_slope_status"][1:] == "pass")
    np.testing.assert_allclose(result["minus_k_per_m"][:, :, :, 0], -1)
    np.testing.assert_allclose(result["plus_k_per_m"][:, :, :, 0], 1)
    np.testing.assert_allclose(result["central_k_per_m"][:, :, :, 0], 0)


def test_central_cancellation_does_not_resolve_one_sided_aliasing():
    center, shifted, kw = stencil()
    # Equal-sign bias cancels from the central derivative, but not either side.
    shifted[0] += 0.0001
    shifted[1] += 0.00005
    result = evaluate_stencil(center, shifted, **kw)
    assert np.all(result["forward_status"] == "pass")
    np.testing.assert_allclose(
        result["central_k_per_m"][0], result["central_k_per_m"][2], atol=1e-11
    )
    assert np.all(result["one_sided_status"] == "unresolved")
    assert np.all(result["common_slope_status"] == "unresolved")


def test_angular_controls_cannot_cancel_each_other():
    center, shifted, kw = stencil()
    for b in range(3):
        for m in range(3):
            center[b, m] += (b - m) * 0.001
            shifted[b, m] += (b - m) * 0.001
    result = evaluate_stencil(center, shifted, **kw)
    assert np.all(result["forward_comparisons"]["joint"]["difference"] == 0)
    assert np.all(result["forward_status"] == "fail")
    assert np.all(result["common_slope_status"] != "pass")


def test_joint_change_cannot_hide_behind_small_adjacent_changes():
    center, shifted, kw = stencil()
    for b in range(3):
        center[b] += b * 0.00015
        shifted[b] += b * 0.00015
    result = evaluate_stencil(center, shifted, **kw)
    assert np.all(result["forward_comparisons"]["bearing"]["within_budget"])
    assert np.all(result["forward_status"] == "fail")


def test_step_instability_is_not_a_kink_or_common_slope():
    center, shifted, kw = stencil(lambda a, d: (a + 1) * d + 10 * d**2)
    result = evaluate_stencil(center, shifted, **kw)
    assert np.all(result["forward_status"] == "pass")
    assert np.all(result["behavior"] == "unresolved")


@pytest.mark.parametrize("problem", ["missing", "nan", "edge"])
def test_incomplete_nonfinite_and_edge_cells_never_pass(problem):
    center, shifted, kw = stencil()
    if problem == "missing":
        kw["displaced_present"][1, 2, 1, 0, 0, 0] = False
        # Even a finite stale value behind an absent flag must not count.
    elif problem == "nan":
        shifted[1, 2, 1, 0, 0, 0] = np.nan
    else:
        kw["edge_limited"][0] = True
    result = evaluate_stencil(center, shifted, **kw)
    expected = "not_run" if problem == "missing" else "unresolved"
    assert result["forward_status"][0] == expected
    assert np.all(result["common_slope_status"][:, 0] == expected)
    assert result["forward_status"][1] == "pass"
    assert np.all(result["common_slope_status"][:, 1] == "pass")


def test_axis_and_sign_permutation_uses_labels():
    center, shifted, kw = stencil()
    expected = evaluate_stencil(center, shifted, **kw)
    shifted = shifted[:, :, :, [2, 0, 1]][:, :, :, :, [1, 0]]
    kw["axes"] = ["U", "E", "N"]
    kw["signs"] = [1, -1]
    result = evaluate_stencil(center, shifted, **kw)
    for name in ("minus_k_per_m", "plus_k_per_m", "common_slope_status"):
        np.testing.assert_array_equal(result[name], expected[name])


def test_refinement_extension_assesses_finest_consecutive_triplet():
    center, shifted, kw = stencil()
    center = np.concatenate([center[:1] + 1, center], axis=0)
    shifted = np.concatenate([shifted[:1] + 1, shifted], axis=0)
    kw["bearing_spacing_deg"] = [0.2, 0.1, 0.05, 0.025]
    kw["center_present"] = np.ones(center.shape, bool)
    kw["displaced_present"] = np.ones(shifted.shape, bool)
    assert np.all(evaluate_stencil(center, shifted, **kw)["forward_status"] == "pass")
    kw["bearing_spacing_deg"] = kw["bearing_spacing_deg"][:3]
    kw["center_present"] = kw["center_present"][:3]
    kw["displaced_present"] = kw["displaced_present"][:3]
    assert np.all(
        evaluate_stencil(center[:3], shifted[:3], **kw)["forward_status"] == "fail"
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("axes", ["E", "E", "U"]),
        ("signs", [-1, True]),
        ("steps_m", [0.01, 0.001, 0.003]),
        ("mask_subsamples", [180, 360, 720.5]),
        ("bearing_spacing_deg", [0.1, 0.05]),
        ("bearing_spacing_deg", [0.1, 0.025, 0.0125]),
        ("mask_subsamples", [180, 360, 1440]),
        ("cell_coordinates", {"sample_id": ["same", "same"]}),
        ("edge_limited", [0, 0]),
    ],
)
def test_invalid_coordinates_rejected(key, value):
    center, shifted, kw = stencil()
    kw[key] = value
    with pytest.raises(ValueError):
        evaluate_stencil(center, shifted, **kw)


def test_worst_case_summary_retains_coordinates_and_missing_count():
    result = difference_summary(
        np.array([[1, np.nan], [-4, 2.0]]),
        {"axis": ["E", "N"], "frequency_mhz": [50, 100]},
    )
    assert result["worst_coordinates"] == {"axis": "N", "frequency_mhz": 50}
    assert result["max_abs"] == 4
    assert result["finite_count"] == 3
    assert result["nonfinite_count"] == 1
    empty = difference_summary(np.array([np.nan]), {"sample": [0]})
    assert empty["rms"] is None


@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan, True])
def test_bad_criteria(value):
    with pytest.raises(ValueError):
        Criteria(forward_atol_k=value)


def test_multiple_named_cell_dimensions_preserve_local_failures():
    center, shifted, kw = stencil(cells=6)
    center = center.reshape(3, 3, 2, 3)
    shifted = shifted.reshape(3, 3, 3, 3, 2, 2, 3)
    kw.update(
        cell_coordinates={"height_m": [30, 87.5], "frequency_mhz": [50, 100, 150]},
        center_present=np.ones(center.shape, bool),
        displaced_present=np.ones(shifted.shape, bool),
        edge_limited=np.zeros((2, 3), bool),
    )
    center[0, :, 0, 2] += 0.001
    result = evaluate_stencil(center, shifted, **kw)
    assert result["forward_status"].shape == (2, 3)
    assert result["forward_status"][0, 2] == "fail"
    assert np.count_nonzero(result["forward_status"] == "pass") == 5
    assert result["common_slope_status"].shape == (3, 2, 3)
    assert np.all(result["common_slope_status"][:, 1] == "pass")
