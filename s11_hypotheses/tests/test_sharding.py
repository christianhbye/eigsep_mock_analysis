"""Execution partitions cannot change science parameters or weaken identity."""

import copy
import json

import numpy as np
import pytest
from eigsep_analysis import s11_hypothesis_io as io

from s11_hypotheses import checkpoint as ck
from s11_hypotheses import sharding
from s11_hypotheses.config import DEFAULTS
from s11_hypotheses.run import _unit_key, gate, physical_units


def test_exact_fixed_matrix_partition_and_native_key_invariance():
    config = dict(choices=copy.deepcopy(DEFAULTS))
    draws = physical_units(config)
    plan = sharding.plan(draws, 4)
    ids = [
        sharding.task_name(t)
        for stage in ["primary", "fine"]
        for i in range(4)
        for t in sharding.selected(plan, stage, i)
    ]
    assert len(ids) == len(set(ids)) == 96
    assert sum(len(p) for p in plan["partitions"]["primary"]) == 51
    assert sum(len(p) for p in plan["partitions"]["fine"]) == 45
    other = sharding.plan(draws, 2)
    t = sharding.selected(plan, "primary", 0)[0]
    assert t in other["tasks"]
    assert _unit_key({}, "input", config["choices"], t) == _unit_key(
        {}, "input", config["choices"], other["tasks"][other["tasks"].index(t)]
    )
    with pytest.raises(ValueError, match="duplicate"):
        sharding.plan(draws + draws[:1])


def make_shards(tmp_path):
    choices, identity, manifest = {}, {"source": "fixed"}, "input"
    draw = dict(
        configuration="tiny",
        epoch="night",
        member=0,
        radius_m=600.0,
        stride=4,
        bin_deg=1.0,
    )
    plan = sharding.plan([draw], 2)
    geometry = tmp_path / "geometry"
    cell_params = dict(draw, member="geometry")
    cell_key = _unit_key(identity, manifest, choices, cell_params)
    cell = ck.Unit(geometry, "cells_tiny_night", cell_key)
    ck.write_h5(cell.path / "cells.h5", dict(E=[1.0, 2.0]))
    cell.finish(dict(parameters=draw))
    roots = []
    for i in range(2):
        root = tmp_path / f"shard{i}"
        selected = sharding.selected(plan, "primary", i)
        for task in selected:
            f = np.array([50e6, 250e6])
            params = dict(
                task,
                frequency_sha256=ck.fingerprint(f.tolist()),
                cell_result_sha256=io.digest(cell.path / "cells.h5"),
            )
            key = _unit_key(identity, manifest, choices, params)
            unit = ck.Unit(root, sharding.task_name(task), key)
            ck.write_h5(unit.path / "spectra.h5", dict(frequency_hz=f, po=[1j, 2j]))
            unit.finish(dict(parameters=params))
        ck.write_json(
            root / "SHARD_DONE.json",
            dict(
                plan_sha256=plan["plan_sha256"],
                source_fingerprint=ck.fingerprint(identity),
                manifest_sha256=manifest,
                stage="primary",
                index=i,
                units=[sharding.task_name(t) for t in selected],
            ),
        )
        ck.write_json(root / "EXECUTION_PLAN.json", plan)
        ck.write_json(
            root / "RUN_RECORD_physical-shard.json",
            dict(
                source_identity=identity,
                input_manifest_sha256=manifest,
                configuration=dict(choices=choices),
                execution_selection=dict(
                    stage="primary", index=i, plan_sha256=plan["plan_sha256"]
                ),
            ),
        )
        roots.append(root)
    return plan, roots, geometry, identity, manifest, choices


def test_gather_verifies_original_bytes_and_rejects_missing_overlap(tmp_path):
    plan, roots, geometry, identity, manifest, choices = make_shards(tmp_path)
    output = tmp_path / "collection"
    with pytest.raises(ValueError, match="missing planned"):
        sharding.gather(plan, roots[:1], output, identity, manifest, choices, geometry)
    with pytest.raises(ValueError, match="overlapping"):
        sharding.gather(
            plan, [roots[0], roots[0]], output, identity, manifest, choices, geometry
        )
    result = sharding.gather(plan, roots, output, identity, manifest, choices, geometry)
    assert len(result["imports"]) == 2 and not result["pending_units"]
    for row in result["imports"]:
        src = roots[0] if (roots[0] / "units" / row["unit"]).exists() else roots[1]
        for name in ["DONE.json", "spectra.h5"]:
            assert (src / "units" / row["unit"] / name).read_bytes() == (
                output / "units" / row["unit"] / name
            ).read_bytes()


def test_gather_rejects_changed_payload_source_and_metadata(tmp_path):
    plan, roots, geometry, identity, manifest, choices = make_shards(tmp_path)
    path = next((roots[0] / "units").glob("*/spectra.h5"))
    path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        sharding.gather(
            plan, roots, tmp_path / "out", identity, manifest, choices, geometry
        )
    with pytest.raises(ValueError, match="source"):
        sharding.gather(
            plan,
            roots,
            tmp_path / "out",
            {"source": "new"},
            manifest,
            choices,
            geometry,
        )


def test_shard_gate_requires_fixed_plan_and_stage(tmp_path):
    config = dict(
        choices=copy.deepcopy(DEFAULTS), resource_gate=str(tmp_path / "gate.json")
    )
    identity, manifest = {}, "input"
    plan = sharding.plan(physical_units(config))
    approval = dict(
        allow=["physical-shard"],
        source_fingerprint=ck.fingerprint(identity),
        manifest_sha256=manifest,
        configuration_fingerprint=ck.fingerprint(
            dict(choices=config["choices"], units=physical_units(config))
        ),
        plan_sha256=plan["plan_sha256"],
        stages=["primary"],
    )
    (tmp_path / "gate.json").write_text(json.dumps(approval))
    gate(config, "physical-shard", identity, manifest, plan, "primary")
    with pytest.raises(ValueError, match="stage"):
        gate(config, "physical-shard", identity, manifest, plan, "fine")
    with pytest.raises(ValueError, match="plan"):
        gate(
            config,
            "physical-shard",
            identity,
            manifest,
            sharding.plan(physical_units(config), 2),
            "primary",
        )
