"""Planning controls: coordinate selection, identity, deduplication and I/O."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from horizon_position_d5.cache import BlockCache, atomic_json, file_sha256, fingerprint
from horizon_position_d5.convergence import Criteria
from horizon_position_d5.convergence_plan import (
    ConvergencePlan,
    build_inventory,
    load_upstreams,
    select_stress_position,
)
from horizon_position_d5.reference_convergence import main


def arrays():
    coords = {
        "nominal_enu_m": np.array([100, 200, 300.0]),
        "heights_m": np.array([91, 30, 87.5]),
        "times_jd": np.array([11, 10, 11.0]),
        "observed_coordinate": np.array([True, True, False]),
        "frequencies_mhz": np.array([100, 50.0]),
        "bearings_deg": np.arange(0, 360, 90.0),
    }
    jac = np.zeros((2, 2, 3, 3, 3, 2, 3))
    # Intentionally non-default grid/method/height/time/frequency orders.
    jac[0, 0, :, 2] = 0.1
    jac[0, 0, :, 2, 2] = 100  # synthetic sample must not win
    return {
        "coordinates": coords,
        "validation_coordinates": {
            "nominal_enu_m": coords["nominal_enu_m"].copy(),
            "offsets_m": np.array([[0, 0, 0], [-1, 0.2, 0.3], [2, 0.2, 0.3]]),
            "derivative_indices": np.array([2, 1]),
            "steps_m": np.array([0.003, 0.01]),
        },
        "interpolation": {
            "grid_points": np.array([5, 3]),
            "methods": np.array(["cubic", "linear"]),
            "jacobian_k_per_m": jac,
        },
        "heldout_derivatives": {"central_k_per_m": np.zeros((2, 2, 3, 3, 3, 2))},
    }


def write_bundle(path, phase, identity, blocks):
    cache = BlockCache(path / "blocks", identity)
    files = {}
    for name, value in blocks.items():
        cache.write(name, {"result": name}, value)
        files[name] = {
            kind: f"blocks/{name}.{extension}"
            for kind, extension in [("data", "npz"), ("metadata", "json")]
        }
        files[name].update(
            {
                f"{kind}_sha256": file_sha256(path / files[name][kind])
                for kind in ("data", "metadata")
            }
        )
    record = {
        "schema_version": 1,
        "label": "diagnostic",
        "phase": phase,
        "identity": identity,
        "identity_hash": fingerprint(identity),
    }
    atomic_json(path / "RUN_RECORD.json", record)
    manifest = dict(record, complete=True, files=files)
    atomic_json(path / "RESULTS.json", manifest)
    return manifest


@pytest.fixture
def bundles(tmp_path):
    blocks = arrays()
    inputs = {"dem.npz": "d" * 64}
    pilot_identity = {
        "input_hashes": inputs,
        "config": {
            "height_offsets_m": [0, -61, -3.5],
            "inner_cutoff_m": 0.25,
            "earth_radius_m": 6371000,
        },
    }
    pilot_dir, ensemble_dir = tmp_path / "pilot", tmp_path / "ensemble"
    pilot = write_bundle(
        pilot_dir,
        "temperature-pilot",
        pilot_identity,
        {"coordinates": blocks["coordinates"], "response": {"kernel_k": np.zeros(1)}},
    )
    phash = file_sha256(pilot_dir / "RESULTS.json")
    write_bundle(
        ensemble_dir,
        "position-ensemble",
        {
            "input_hashes": inputs,
            "config": {"pilot_identity": pilot_identity, "pilot_results_sha256": phash},
        },
        blocks,
    )
    plan = ConvergencePlan(phash, file_sha256(ensemble_dir / "RESULTS.json"))
    return plan, pilot_dir, ensemble_dir, pilot


def inventory(bundles, plan=None, source=None):
    default, pilot_dir, ensemble_dir, _ = bundles
    plan = default if plan is None else plan
    source = source or {
        "reference_sha256": "a" * 64,
        "temperature_sha256": "b" * 64,
        "mask_sha256": "c" * 64,
    }
    return build_inventory(
        plan, *load_upstreams(plan, pilot_dir, ensemble_dir), source=source
    )


def test_selection_uses_coordinate_values_and_deterministic_ties():
    selected = select_stress_position(arrays())
    assert selected["offset_m"] == [-1, 0.2, 0.3]
    assert selected["height_m"] == 87.5
    assert selected["axis"] == "E"
    assert selected["time_jd"] == 10
    assert selected["frequency_mhz"] == 50
    assert selected["absolute_discrepancy_k_per_m"] == 0.1


def test_selection_rejects_nonfinite_and_invalid_derivative_indices():
    blocks = arrays()
    blocks["interpolation"]["jacobian_k_per_m"][0, 0, 1, 2, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        select_stress_position(blocks)
    blocks = arrays()
    blocks["validation_coordinates"]["derivative_indices"] = np.array([-1, 1])
    with pytest.raises(ValueError, match="indices"):
        select_stress_position(blocks)


def test_plan_roundtrip_and_inventory_deduplicate_mask_work(bundles):
    plan = bundles[0]
    assert ConvergencePlan.from_dict(json.loads(json.dumps(plan.to_dict()))) == plan
    first = inventory(bundles)
    assert first == inventory(bundles)
    assert first["counts"] == {
        "positions": 2,
        "traces": 114,
        "evaluations": 342,
        "stencil_entries": 342,
    }
    assert not first["execution_started"]
    assert {x["status"] for x in first["traces"]} == {"not_run"}
    assert len({x["id"] for x in first["traces"]}) == 114
    # Every trace is referenced by all three mask evaluations, not retraced.
    links = [x["parameters"]["trace_id"] for x in first["evaluations"]]
    assert all(links.count(t["id"]) == 3 for t in first["traces"])


def test_numerical_criteria_do_not_change_traces_but_do_change_inventory(bundles):
    first = inventory(bundles)
    second = inventory(
        bundles, replace(bundles[0], criteria=Criteria(forward_atol_k=1e-4))
    )
    assert first["traces"] == second["traces"]
    assert first["inventory_hash"] != second["inventory_hash"]
    assert first["plan_hash"] != second["plan_hash"]


def test_mask_change_reuses_trace_identity_and_reference_change_does_not(bundles):
    first = inventory(bundles)
    more_masks = inventory(
        bundles, replace(bundles[0], mask_subsamples=(180, 360, 720, 1440))
    )
    assert first["traces"] == more_masks["traces"]
    assert len(more_masks["evaluations"]) == 456
    changed = inventory(
        bundles,
        source={
            "reference_sha256": "f" * 64,
            "temperature_sha256": "b" * 64,
            "mask_sha256": "c" * 64,
        },
    )
    assert {t["id"] for t in first["traces"]}.isdisjoint(
        t["id"] for t in changed["traces"]
    )


def test_nominal_stress_point_is_deduplicated_and_boundary_stencil_rejected(bundles):
    plan = bundles[0]
    p, e, pa, ea = load_upstreams(plan, bundles[1], bundles[2])
    ea["validation_coordinates"]["derivative_indices"] = np.array([2, 0])
    ea["interpolation"]["jacobian_k_per_m"][0, 0, 0] *= 2
    source = {
        "reference_sha256": "a" * 64,
        "temperature_sha256": "b" * 64,
        "mask_sha256": "c" * 64,
    }
    result = build_inventory(plan, p, e, pa, ea, source=source)
    assert result["counts"]["positions"] == 1
    assert result["counts"]["traces"] == 57
    ea["validation_coordinates"]["offsets_m"][0, 0] = 3.195
    with pytest.raises(ValueError, match="stencil leaves"):
        build_inventory(plan, p, e, pa, ea, source=source)


def test_corrupt_upstream_rejected_without_repair(bundles):
    plan, pilot_dir, ensemble_dir, _ = bundles
    path = ensemble_dir / "blocks/interpolation.npz"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_upstreams(plan, pilot_dir, ensemble_dir)
    assert path.read_bytes() == b"corrupt"


def test_missing_block_and_wrong_manifest_pin_rejected(bundles):
    plan, pilot_dir, ensemble_dir, _ = bundles
    with pytest.raises(ValueError, match="pinned"):
        load_upstreams(
            replace(plan, ensemble_results_sha256="0" * 64), pilot_dir, ensemble_dir
        )
    (ensemble_dir / "blocks/interpolation.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_upstreams(plan, pilot_dir, ensemble_dir)


def test_mixed_upstream_coordinates_rejected_even_with_valid_file_hashes(bundles):
    plan, pilot_dir, ensemble_dir, _ = bundles
    blocks = arrays()
    blocks["coordinates"]["frequencies_mhz"] = np.array([101, 50.0])
    manifest = json.loads((ensemble_dir / "RESULTS.json").read_text())
    write_bundle(ensemble_dir, "position-ensemble", manifest["identity"], blocks)
    plan = replace(
        plan, ensemble_results_sha256=file_sha256(ensemble_dir / "RESULTS.json")
    )
    with pytest.raises(ValueError, match="coordinate mismatch"):
        load_upstreams(plan, pilot_dir, ensemble_dir)


def test_cli_read_only_and_execution_not_available(bundles, tmp_path, capsys):
    plan, pilot_dir, ensemble_dir, _ = bundles
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan.to_dict()))
    before = {
        str(p): (file_sha256(p), p.stat().st_mtime_ns)
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    args = [
        "--plan",
        str(plan_path),
        "--pilot",
        str(pilot_dir),
        "--ensemble",
        str(ensemble_dir),
    ]
    with pytest.raises(SystemExit):
        main(args)
    main([*args, "--dry-run"])
    result = json.loads(capsys.readouterr().out)
    assert result["counts"]["traces"] == 114
    after = {
        str(p): (file_sha256(p), p.stat().st_mtime_ns)
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    assert before == after


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"pilot_results_sha256": "not-a-hash"},
        {"unrecognized": 1},
        {"steps_m": [0.01, 0.003, 0.003]},
        {"steps_m": [0.01, -0.003, 0.001]},
        {"mask_subsamples": [180, 360, 720.5]},
        {"mask_subsamples": [180, 360, 1440]},
        {"bearing_spacing_deg": [0.1, 0.025, 0.0125]},
        {"bearing_spacing_deg": [0.11, 0.055, 0.0275]},
        {"half_width_m": 0.001},
        {"criteria": {"typo": 1}},
    ],
)
def test_malformed_plan_is_rejected(change):
    value = ConvergencePlan("a" * 64, "b" * 64).to_dict()
    value.update(copy.deepcopy(change))
    with pytest.raises(ValueError):
        ConvergencePlan.from_dict(value)


def test_unrepresentable_absolute_displacement_is_rejected(bundles):
    plan = bundles[0]
    p, e, pa, ea = load_upstreams(plan, bundles[1], bundles[2])
    pa["coordinates"]["nominal_enu_m"][:] = 1e20
    with pytest.raises(ValueError, match="coordinate precision"):
        build_inventory(
            plan,
            p,
            e,
            pa,
            ea,
            source={
                "reference_sha256": "a" * 64,
                "temperature_sha256": "b" * 64,
                "mask_sha256": "c" * 64,
            },
        )
