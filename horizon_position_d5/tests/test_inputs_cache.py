import json

import numpy as np
import pytest

from horizon_position_d5.cache import BlockCache, file_sha256, fingerprint
from horizon_position_d5.inputs import select_night_rows, verify_inputs


def test_verified_manifest_rejects_changed_input(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("a")
    manifest = {"input.txt": {"sha256": file_sha256(source), "bytes": 1}}
    (tmp_path / "INPUT_MANIFEST.json").write_text(json.dumps(manifest))
    assert (
        verify_inputs(tmp_path)["input.txt"]["sha256"]
        == manifest["input.txt"]["sha256"]
    )
    source.write_text("b")
    with pytest.raises(ValueError, match="input.txt"):
        verify_inputs(tmp_path)


def test_cache_invalidates_parameters_source_and_corruption(tmp_path):
    cache = BlockCache(tmp_path, {"source": "one"})
    cache.write("trace", {"step": 0.1}, {"alpha": np.arange(3.0)})
    assert np.array_equal(cache.load("trace", {"step": 0.1})["alpha"], np.arange(3.0))
    assert cache.load("trace", {"step": 0.2}) is None
    assert BlockCache(tmp_path, {"source": "two"}).load("trace", {"step": 0.1}) is None
    (tmp_path / "trace.npz").write_bytes(b"corrupt")
    assert cache.load("trace", {"step": 0.1}) is None


def test_cache_missing_commit_record_is_incomplete(tmp_path):
    cache = BlockCache(tmp_path, {"source": "one"})
    cache.write("trace", {}, {"x": np.zeros(2)})
    (tmp_path / "trace.json").unlink()
    assert cache.load("trace", {}) is None


def test_fingerprint_is_order_independent():
    assert fingerprint({"a": 1, "b": [2]}) == fingerprint({"b": [2], "a": 1})


def test_night_selection_checks_coordinate_alignment():
    binned = {
        "t": np.arange(10.0),
        "era": np.array(["~87.5m"] * 8 + ["~91m"] * 2),
        "az": np.zeros(10),
        "el": np.zeros(10),
    }
    fit = {**binned, "fitted": np.array([True] * 7 + [False] * 3)}
    assert np.array_equal(select_night_rows(binned, fit, 3), [0, 3, 6])
    fit = {**fit, "t": binned["t"] + 1}
    with pytest.raises(ValueError, match="t"):
        select_night_rows(binned, fit, 3)
