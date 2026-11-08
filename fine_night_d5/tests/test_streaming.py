"""Native-support and historical-source checks independent of heavy SHT."""

import json

import numpy as np
import pytest
from fine_night_d5.assets import calibration_stage, sha256
from fine_night_d5.streaming import orientation_groups, reduce_columns


def test_exact_provider_groups_and_rotation_rejection():
    el = np.array([0.0, 20.0, 0.0, 20.0])
    a = np.radians(el)
    rot = np.zeros((4, 3, 3))
    rot[:, 0, 0] = 1
    rot[:, 1, 1] = rot[:, 2, 2] = np.cos(a)
    rot[:, 1, 2] = -np.sin(a)
    rot[:, 2, 1] = np.sin(a)
    groups = orientation_groups(el, np.zeros(4), rot, np.eye(3))
    assert len(groups) == 2
    np.testing.assert_array_equal(groups[0][2], [0, 2])
    np.testing.assert_array_equal(groups[1][2], [1, 3])
    with pytest.raises(ValueError, match="stored native drive"):
        orientation_groups(el + 0.1, np.zeros(4), rot, np.eye(3))
    with pytest.raises(ValueError, match="proper rotations"):
        orientation_groups(el, np.zeros(4), rot, np.diag([-1.0, 1.0, 1.0]))


def test_actual_channel_support_not_random_or_exposure_approximation():
    values = np.arange(24.0).reshape(4, 3, 2)
    use = np.array(
        [
            [True, False, True],
            [False, True, False],
            [True, True, False],
            [True, False, False],
        ]
    )
    mean, count = reduce_columns(values, use, np.array([0, 0, 1, 1]), 3)
    np.testing.assert_array_equal(count, [[1, 1, 1], [2, 1, 0], [0, 0, 0]])
    np.testing.assert_array_equal(mean[0], [values[0, 0], values[1, 1], values[0, 2]])
    np.testing.assert_array_equal(mean[1, 0], (values[2, 0] + values[3, 0]) / 2)
    assert np.isnan(mean[1, 2]).all() and np.isnan(mean[2]).all()
    values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="support differ"):
        reduce_columns(values, use, np.array([0, 0, 1, 1]), 3)


def test_historical_stage_is_bound_to_actual_snapshot(tmp_path):
    (tmp_path / "sources").mkdir()
    source = tmp_path / "sources/fine_night.py"
    source.write_text("historical_actual_producer = 1\n")
    np.savez(tmp_path / "arrays.npz", raw=np.array([1, 2, 3]))
    record = {
        "schema": 1,
        "status": "COMPLETE",
        "label": "diagnostic",
        "arrays_sha256": sha256(tmp_path / "arrays.npz"),
        "producer_source_sha256": sha256(source),
        "producer_sources": {"fine_night.py": sha256(source)},
    }
    path = tmp_path / "STAGE.json"
    path.write_text(json.dumps(record))
    pin = {"path": str(path), "sha256": sha256(path)}
    actual, result = calibration_stage(pin)
    assert actual == tmp_path / "arrays.npz"
    assert result == record
    source.write_text("different_producer = 2\n")
    with pytest.raises(ValueError, match="snapshot differs"):
        calibration_stage(pin)
