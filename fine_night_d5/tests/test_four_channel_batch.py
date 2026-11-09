import json

import numpy as np
import pytest
from fine_night_d5.four_channel_batch import (
    checkpoint,
    preserve_incomplete,
    tasks,
)

from fine_night_d5 import assets


def test_complete_schedule_has_explicit_phase_cost_and_physical_frequencies():
    native = np.arange(264, 984) * 250 / 1024
    centres = native.reshape(-1, 4).mean(1)
    roles = ["raster_hybrid", "represented_hfss_control", "strong"] + [
        f"delete{i}" for i in range(13)
    ]
    plan = tasks(
        native,
        centres,
        roles,
        ["primary_native", "centre_control", "secondary_native"],
    )
    assert len(plan) == 4200
    assert sum(t["phase"] == "primary_native" for t in plan) == 480
    centre = [t for t in plan if t["phase"] == "centre_control"]
    assert len(centre) == 120
    np.testing.assert_array_equal(centre[0]["frequencies_mhz"], centres[:3])
    assert centre[0]["frequency_operator"] == "four-channel-centres"
    with pytest.raises(ValueError, match="unique"):
        tasks(native, centres, roles, ["primary_native", "primary_native"])


def test_resume_checks_source_axes_context_and_actual_payload(tmp_path):
    f = np.array([65.0, 66.0, 67.0])
    v = np.ones((7, 3))
    assets.atomic_npz(
        tmp_path / "columns.npz",
        frequencies_mhz=f,
        native_S=v,
        native_fg=v,
        native_A=v,
    )
    expected = dict(context_sha256="frozen", native_rows=7, frequencies_mhz=f.tolist())
    source = dict(path="frozen-source.json", sha256="original")
    record = dict(
        **expected,
        source_snapshot=source,
        columns_sha256=assets.sha256(tmp_path / "columns.npz"),
    )
    assets.write_json(tmp_path / "RUN_DONE.json", record)
    assert checkpoint(tmp_path, expected, source) == record
    for changed in [
        {**expected, "context_sha256": "other"},
        {**expected, "native_rows": 8},
    ]:
        with pytest.raises(ValueError, match="differs"):
            checkpoint(tmp_path, changed, source)
    with pytest.raises(ValueError, match="source_snapshot"):
        checkpoint(tmp_path, expected, {**source, "sha256": "changed"})
    with (tmp_path / "columns.npz").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="payload"):
        checkpoint(tmp_path, expected, source)


def test_interrupted_target_is_preserved_without_becoming_completed(tmp_path):
    target = tmp_path / "role"
    target.mkdir()
    (target / "PROGRESS.json").write_text(json.dumps({"partial": True}))
    assert checkpoint(target, {}, {}) is None
    preserve_incomplete(target)
    assert not target.exists()
    saved = list(tmp_path.glob("role.incomplete-*"))
    assert len(saved) == 1
    assert json.loads((saved[0] / "PROGRESS.json").read_text())["partial"]
