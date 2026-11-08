"""Independent sparse-mask means and fail-closed chunk identity checks."""

import copy
import json

import numpy as np
import pytest
from fine_night_d5.assets import sha256
from fine_night_d5.join import PHYSICS_SOURCES, join_arrays, read_piece


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


def pinned_record_fixture(tmp_path):
    """Build a real on-disk producer record with independent payload hashes."""

    def save(name, obj):
        path = tmp_path / name
        path.write_text(json.dumps(obj))
        return {"path": str(path), "sha256": sha256(path)}

    runtime = {
        "dependency_sources": {
            name: {"sha256": "1" * 64}
            for name in ("eigsim.simulate", "eigsim.rotations", "croissant.simulator")
        },
        "jax_binaries": {"lib": {"sha256": "2" * 64}},
    }
    runtime_pin = save("runtime.json", runtime)
    source = tmp_path / "sources"
    source.mkdir()
    source_hashes = {}
    for name in PHYSICS_SOURCES:
        path = source / name
        path.write_text("# historical producer " + name)
        source_hashes[name] = sha256(path)
    np.savez(tmp_path / "columns.npz", **piece(283))
    config = {
        "channels": [283],
        "native_stage": {"sha256": "3" * 64},
        "pointing_stage": {"sha256": "4" * 64},
        "beam": {"role": "old_symmetric", "sha256": "5" * 64},
        "runtime_modules": {
            k: v["sha256"] for k, v in runtime["dependency_sources"].items()
        },
        "runtime_evidence": runtime_pin,
    }
    input_pin = save("inputs.json", config)
    record = {
        "status": "DONE",
        "source_commit": "6" * 40,
        "source_sha256": source_hashes,
        "columns_sha256": sha256(tmp_path / "columns.npz"),
        "input_manifest_sha256": input_pin["sha256"],
        "calibration_stage": "3" * 64,
        "pointing_stage": "4" * 64,
        "beam": {"role": "old_symmetric", "input_sha256": "5" * 64},
        "runtime": {"numpy": "test"},
        "iers": {},
        "horizon": {},
    }
    return {"done": save("RUN_DONE.json", record), "inputs": input_pin}


def test_read_piece_preserves_actual_historical_producer(tmp_path):
    pin = pinned_record_fixture(tmp_path)
    (arrays, context), proof, record = read_piece(pin)
    np.testing.assert_array_equal(arrays["channels"], [283])
    assert context["actual_model_producer_commit"] == "6" * 40
    assert proof["actual_producer_commit"] == record["source_commit"]


@pytest.mark.parametrize("bad", ["manifest_axis", "snapshot", "runtime"])
def test_pinned_read_rejects_wrong_scope_or_changed_historical_evidence(tmp_path, bad):
    pin = pinned_record_fixture(tmp_path)
    if bad == "manifest_axis":
        # Even consistently re-hashed JSON cannot substitute a different
        # frequency request for the immutable actual payload channel IDs.
        inputs = json.loads((tmp_path / "inputs.json").read_text())
        inputs["channels"] = [603]
        (tmp_path / "inputs.json").write_text(json.dumps(inputs))
        pin["inputs"]["sha256"] = sha256(tmp_path / "inputs.json")
        record = json.loads((tmp_path / "RUN_DONE.json").read_text())
        record["input_manifest_sha256"] = pin["inputs"]["sha256"]
        (tmp_path / "RUN_DONE.json").write_text(json.dumps(record))
        pin["done"]["sha256"] = sha256(tmp_path / "RUN_DONE.json")
    elif bad == "snapshot":
        (tmp_path / "sources/model.py").write_text("# changed historical producer")
    else:
        (tmp_path / "runtime.json").write_text("{}")
    with pytest.raises(ValueError):
        read_piece(pin)
