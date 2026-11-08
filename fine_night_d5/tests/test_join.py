"""Independent sparse-mask means and fail-closed chunk identity checks."""

import copy

import numpy as np
import pytest
from fine_night_d5.join import join_arrays


def piece(channel):
    s = np.arange(4.0, dtype=float) + channel / 100
    fg = np.full(4, 0.6)
    a = s + 2.725 * 0.4
    mask = (
        np.array([True, False, True, True])
        if channel == 283
        else (np.array([True, True, False, False]))
    )
    if channel == 283:
        indices = [[0], [2, 3]]
    else:
        indices = [[0, 1], []]

    def means(x):
        return np.array([sum(x[k]) / len(k) if k else np.nan for k in indices])[:, None]

    return {
        "native_index": np.arange(4),
        "row": np.array([0, 1, 0, 1]),
        "source_file": np.array(["a", "a", "b", "b"]),
        "input": np.full(4, "4"),
        "time_best": np.array([0.0, 1.0, 10.0, 11.0]),
        "fine_bin_id": np.array([0, 0, 1, 1]),
        "channels": np.array([channel]),
        "frequencies_mhz": np.array([channel]) * 250 / 1024,
        "accepted": mask[:, None],
        "native_S": s[:, None],
        "native_fg": fg[:, None],
        "native_A": a[:, None],
        "bin_S": means(s),
        "bin_fg": means(fg),
        "bin_A": means(a),
        "bin_count": np.array([len(k) for k in indices])[:, None],
    }


def test_roundrobin_reorders_actual_channels_and_retains_empty_bins():
    a, b = piece(283), piece(603)
    out, _ = join_arrays([(b, {"beam": "old"}), (a, {"beam": "old"})], [283, 603])
    np.testing.assert_array_equal(out["channels"], [283, 603])
    np.testing.assert_array_equal(out["bin_count"], [[1, 2], [2, 0]])
    np.testing.assert_allclose(out["bin_S"][0], [2.83, 6.53])
    assert np.isnan(out["bin_A"][1, 1])
    np.testing.assert_array_equal(
        out["accepted"], [[True, True], [False, True], [True, False], [True, False]]
    )


@pytest.mark.parametrize(
    "key", ["time_best", "row", "native_index", "fine_bin_id", "source_file", "input"]
)
def test_no_different_native_coordinate_is_joined(key):
    a, b = piece(283), piece(603)
    b[key] = b[key].copy()
    if b[key].dtype.kind in "US":
        b[key][0] = "x"
    else:
        b[key][0] += 1
    with pytest.raises((ValueError, AssertionError)):
        join_arrays([(a, {}), (b, {})], [283, 603])


@pytest.mark.parametrize(
    "bad", ["overlap", "missing", "context", "mask", "count", "frequency", "nan"]
)
def test_no_coverage_runtime_or_weight_substitution(bad):
    a, b = piece(283), piece(603)
    context = {"runtime": "frozen", "beam": "old"}
    other = copy.deepcopy(context)
    expected = [283, 603]
    if bad == "overlap":
        b = piece(283)
    elif bad == "missing":
        expected = [283, 603, 843]
    elif bad == "context":
        other["runtime"] = "different"
    elif bad == "mask":
        b["accepted"][0] = False
    elif bad == "count":
        b["bin_count"][0] += 1
    elif bad == "frequency":
        b["frequencies_mhz"][0] += 0.001
    else:
        b["native_A"][2] = np.nan  # even excluded-row model identity persists
    with pytest.raises((ValueError, AssertionError)):
        join_arrays([(a, context), (b, other)], expected)
