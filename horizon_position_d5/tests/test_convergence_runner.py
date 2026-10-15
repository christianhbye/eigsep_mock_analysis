"""Small end-to-end runner controls; no D5 tracing or systemd launches."""

import copy
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from horizon_position_d5.cache import file_sha256, load_results
from horizon_position_d5.convergence_plan import build_inventory, load_upstreams
from horizon_position_d5.convergence_runner import (
    ConvergenceRunner,
    TemperatureEvaluator,
    _initialize_trace_worker,
    _trace_task,
)
from horizon_position_d5.tests.test_convergence_plan import bundles as bundles


def manufactured_trace(payload):
    params, settings = payload
    n = params["bearing_count"]
    shape = (len(settings["height_offsets_m"]), n)
    position = np.asarray(params["position_enu_m"])
    alpha = np.full(shape, position.sum() / 1000)
    result = {
        name: np.ones(shape)
        for name in ("distance_m", "surface_height_m", "curvature_height_m")
    }
    result.update(
        alpha_rad=alpha,
        gradient_en=np.zeros((*shape, 2)),
        edge_range_m=np.ones(n),
        edge_limited=np.zeros(shape, bool),
        bearings_deg=np.arange(n) * params["bearing_spacing_deg"],
        position_enu_m=position,
    )
    return result, {"seconds": 0.001, "peak_rss_kib": 10}


def manufactured_temperature(trace, sub):
    return np.broadcast_to(trace["alpha_rad"][:, :1, None] * 1000, (3, 3, 2)).copy()


@pytest.fixture
def runner_setup(bundles, tmp_path):
    plan = replace(
        bundles[0], bearing_spacing_deg=(90, 45, 22.5), mask_subsamples=(2, 4, 8)
    )
    upstreams = load_upstreams(plan, bundles[1], bundles[2])
    source = {
        "source_dirty": False,
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "lock_sha256": "c" * 64,
        "reference_sha256": "d" * 64,
        "temperature_sha256": "e" * 64,
        "mask_sha256": "f" * 64,
    }
    inventory = build_inventory(plan, *upstreams, source=source)
    runtime = {k: source[k] for k in ("source_commit", "source_tree", "lock_sha256")}
    runtime.update(versions={}, input_hashes={"dem.npz": "d" * 64})
    output = tmp_path / "run"

    def build(evaluator=manufactured_temperature, supplied_runtime=None):
        return ConvergenceRunner(
            output,
            inventory,
            runtime if supplied_runtime is None else supplied_runtime,
            workers=1,
            dem_path=tmp_path / "unused.npz",
            evaluator=evaluator,
        )

    return build, output, inventory, runtime


def test_probe_then_resume_exports_all_raw_values_and_verdicts(runner_setup):
    build, output, inventory, _ = runner_setup
    report = build().run(probe=True, trace_function=manufactured_trace)
    assert report["executed_scope"] == {"traces": 3, "evaluations": 9}
    assert not report["full_run_complete"]
    assert not (output / "RESULTS.json").exists()
    probe_hash = file_sha256(output / "PROBE_REPORT.json")
    report = build().run(trace_function=manufactured_trace)
    assert sum(t["reused"] for t in report["timings"] if t["kind"] == "trace") == 3
    assert sum(t["reused"] for t in report["timings"] if t["kind"] == "evaluation") == 9
    assert file_sha256(output / "PROBE_REPORT.json") == probe_hash
    manifest, blocks = load_results(output, expected_phase="reference-convergence")
    assert len(manifest["files"]) == 114 + 342 + 3
    raw = blocks["stencil_temperatures"]
    assert raw["center_k"].shape == (3, 3, 2, 3, 3, 2)
    assert raw["displaced_k"].shape == (3, 3, 3, 3, 2, 2, 3, 3, 2)
    verdict = blocks["convergence"]
    assert np.all(verdict["common_slope_status"] == "pass")
    np.testing.assert_allclose(verdict["central_k_per_m"], 1, atol=2e-10)
    assert len([key for key in blocks if key.startswith("trace-")]) == 114
    with pytest.raises(ValueError, match="already complete"):
        build().run(trace_function=manufactured_trace)


def test_interruption_retains_checkpoints_and_resume(runner_setup):
    build, output, *_ = runner_setup
    calls = []

    def interrupt(payload):
        if len(calls) == 5:
            raise RuntimeError("injected interruption")
        calls.append(payload)
        return manufactured_trace(payload)

    with pytest.raises(RuntimeError, match="injected"):
        build().run(trace_function=interrupt)
    failures = list(output.glob("failure-*.json"))
    assert len(failures) == 1
    assert not (output / "RESULTS.json").exists()
    resumed = build().run(trace_function=manufactured_trace)
    assert sum(t["reused"] for t in resumed["timings"] if t["kind"] == "trace") == 5
    assert failures[0].exists()


def test_changed_identity_rejected_without_overwriting_record(runner_setup):
    build, output, _, runtime = runner_setup
    build().run(probe=True, trace_function=manufactured_trace)
    before = (output / "RUN_RECORD.json").read_bytes()
    changed = copy.deepcopy(runtime)
    changed["input_hashes"]["dem.npz"] = "0" * 64
    with pytest.raises(ValueError, match="identity changed"):
        build(supplied_runtime=changed).run(trace_function=manufactured_trace)
    assert (output / "RUN_RECORD.json").read_bytes() == before


def test_corrupt_probe_trace_is_recomputed(runner_setup):
    build, output, *_ = runner_setup
    build().run(probe=True, trace_function=manufactured_trace)
    next((output / "traces").glob("*.npz")).write_bytes(b"corrupt")
    report = build().run(trace_function=manufactured_trace)
    assert sum(t["reused"] for t in report["timings"] if t["kind"] == "trace") == 2
    load_results(output, expected_phase="reference-convergence")


def test_numerical_failure_can_complete_but_invalid_output_cannot(runner_setup):
    build, output, *_ = runner_setup

    def angular_error(trace, sub):
        return manufactured_temperature(trace, sub) + sub * 0.01

    report = build(evaluator=angular_error).run(trace_function=manufactured_trace)
    assert report["full_run_complete"]
    assert report["outcomes"]["forward_status"] == {"fail": 36}
    assert (output / "RESULTS.json").is_file()


def test_nonfinite_temperature_never_gets_checkpointed(runner_setup):
    build, output, *_ = runner_setup

    def bad(trace, sub):
        return np.full((3, 3, 2), np.nan)

    with pytest.raises(ValueError, match="finiteness"):
        build(evaluator=bad).run(probe=True, trace_function=manufactured_trace)
    assert not list((output / "blocks").glob("temperature-*.npz"))
    assert not (output / "RESULTS.json").exists()


def test_real_tracer_serial_spawn_agree(tmp_path):
    from horizon_position_d5.cache import fingerprint

    path = tmp_path / "dem.npz"
    np.savez(path, dem=np.zeros((20, 20)), res=1, e0_px=0, n0_px=0)
    settings = {
        "height_offsets_m": [0, -1],
        "method": "continuous",
        "inner_cutoff_m": 0.25,
        "earth_radius_m": 6371000,
    }
    bearings = [0.0, 90.0, 180.0, 270.0]
    params = {
        "position_enu_m": [10, 10, 2],
        "bearing_count": 4,
        "bearing_spacing_deg": 90,
        "bearings_hash": fingerprint(bearings),
    }
    _initialize_trace_worker(path)
    serial, _ = _trace_task((params, settings))
    with ProcessPoolExecutor(
        max_workers=2,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_trace_worker,
        initargs=(path,),
    ) as pool:
        outputs = list(pool.map(_trace_task, [(params, settings)] * 2))
    for output, timing in outputs:
        for name in serial:
            np.testing.assert_array_equal(output[name], serial[name])
        assert timing["seconds"] > 0


def test_temperature_operator_all_sky_and_all_ground():
    import jax

    jax.config.update("jax_enable_x64", True)
    from eigsim.horizon import mwss_grid

    theta, phi = mwss_grid(3)
    response = {
        "offset_k": np.array([[1.0], [2.0]]),
        "kernel_k": np.ones((2, 1, len(theta), len(phi))),
    }
    evaluate = TemperatureEvaluator(response, 3)
    trace = {
        "bearings_deg": np.arange(0, 360, 45.0),
        "alpha_rad": np.stack([np.full(8, -np.pi / 2), np.full(8, np.pi / 2)]),
    }
    for sub in (2, 4):
        values = evaluate(trace, sub)
        np.testing.assert_allclose(
            values[0], response["offset_k"] + len(theta) * len(phi)
        )
        np.testing.assert_allclose(values[1], response["offset_k"])


def test_missing_or_corrupt_run_record_cannot_claim_checkpoints(runner_setup):
    build, output, *_ = runner_setup
    build().run(probe=True, trace_function=manufactured_trace)
    path = output / "RUN_RECORD.json"
    record = json.loads(path.read_text())
    record["identity_hash"] = "0" * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="identity changed"):
        build().run(trace_function=manufactured_trace)
    path.unlink()
    with pytest.raises(ValueError, match="unowned"):
        build().run(trace_function=manufactured_trace)


def test_second_writer_cannot_enter_run(runner_setup):
    import fcntl

    build, output, *_ = runner_setup
    output.mkdir()
    with (output / "RUN.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another runner"):
            build().run(probe=True, trace_function=manufactured_trace)
    assert not (output / "RUN_RECORD.json").exists()
