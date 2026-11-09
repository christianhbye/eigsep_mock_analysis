"""Known analytic pole slots never authorize unrelated NaN replacement."""

import numpy as np
import pytest
from fine_night_d5.sht_operands import checked_poles, raw_support


def test_raw_nonfinite_gate_distinguishes_boundary_from_interior_and_invalid():
    raw = np.zeros((7, 5, 4))
    theta = np.linspace(0, np.pi, 5)
    raw[3, 0, 1] = np.nan
    bad, summary = raw_support(raw, theta)
    assert bad.sum() == 1 and summary["nonfinite_total"] == 1
    raw[3, 1, 1] = np.inf
    with pytest.raises(ValueError, match=r"theta\[1\].*ell=1, m=0"):
        raw_support(raw, theta)
    raw[3, 1, 1] = 0
    raw[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match=r"ell=0, m=3"):
        raw_support(raw, theta)


def test_completed_kernel_must_match_analytic_poles_and_be_finite_everywhere():
    L = 4
    kernel = np.zeros((5, L, L))
    norm = np.sqrt((2 * np.arange(L) + 1) / (4 * np.pi))
    weights = np.array([0.2, 1, 1, 1, 0.3])
    kernel[0, :, 0] = norm * weights[0]
    kernel[-1, :, 0] = (-1.0) ** np.arange(L) * norm * weights[-1]
    checked_poles(kernel, weights)
    kernel[2, 1, 0] = np.nan
    with pytest.raises(ValueError, match="completed.*nonfinite"):
        checked_poles(kernel, weights)
    kernel[2, 1, 0] = 0
    kernel[0, 2, 1] = 1
    with pytest.raises(AssertionError):
        checked_poles(kernel, weights)
