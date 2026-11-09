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
