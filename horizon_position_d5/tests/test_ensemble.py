"""Tiny end-to-end check using analytic horizons and an independent mask kernel."""

# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from horizon_position_d5.cache import BlockCache, atomic_json, load_results
from horizon_position_d5.ensemble import Ensemble, validation_positions
from horizon_position_d5.pilot import temperature_model


def test_ensemble_preserves_axis_order_and_exact_affine_horizon_truth(tmp_path):
    study = Ensemble.__new__(Ensemble)
    study.args = SimpleNamespace(output=Path(tmp_path), grid_points=[3, 5], workers=1)
    import time

    study.started = time.monotonic()
    study.p = np.zeros(3)
    study.bearings = np.arange(0, 360, 45.0)
    study.offset = np.full((6, 2), 300.0)
    study.kernel = np.random.default_rng(971).normal(size=(6, 2, 4, 6))
    study.lmax = 2
    study.model = temperature_model(study.offset, study.kernel, study.bearings, 2)
    study.points = validation_positions()
    study.derivative_indices = np.array([0, 4, 8, 14])
    study.fd_steps = np.array([0.01, 0.003])
    study.coords = {"observed_coordinate": np.array([True] * 4 + [False] * 2)}
    study.config = {"declared_noise_scales_k": [0.001, 0.01, 0.1, 1.0]}
    study.identity = {"test": "analytic-horizon"}
    study.cache = BlockCache(tmp_path / "blocks", study.identity)
    study.record = {
        "schema_version": 1,
        "label": "diagnostic",
        "phase": "position-ensemble",
        "identity": study.identity,
        "identity_hash": study.cache.identity_hash,
    }
    atomic_json(tmp_path / "RUN_RECORD.json", study.record)
    study.exports, study.timings = {}, []
    beta = np.radians(study.bearings)
    horizon_jac = np.stack(
        [0.003 * np.cos(beta), 0.002 * np.sin(2 * beta), np.full(len(beta), 0.001)],
        axis=-1,
    )
    nominal_alpha = np.array([0.11, 0.13, 0.14])[:, None] + 0.01 * np.cos(beta)

    def trace(offsets):
        return (
            nominal_alpha[None]
            + np.einsum("bi,pi->pb", horizon_jac, np.asarray(offsets))[:, None]
        )

    study.traces = trace
    nominal_t = study.temperatures(nominal_alpha)
    exact_j = np.stack(
        [
            np.stack(
                [
                    np.asarray(
                        jax.jvp(
                            study.model,
                            (jnp.asarray(alpha),),
                            (jnp.asarray(horizon_jac[:, axis]),),
                        )[1]
                    )
                    for axis in range(3)
                ],
                axis=-1,
            )
            for alpha in nominal_alpha
        ]
    )
    study.arrays = {
        "nominal": {
            "temperature_k": np.stack([nominal_t] * 2),
            "candidate_jacobian_k_per_m": np.stack([exact_j] * 4),
            "candidate_names": np.array(
                ["sampled_e7", "sampled_e7prime", "continuous_e7", "continuous_e7prime"]
            ),
        }
    }
    study.run()
    manifest, arrays = load_results(tmp_path, expected_phase="position-ensemble")
    assert manifest["complete"]
    assert arrays["interpolation"]["temperature_k"].shape == (2, 2, 18, 3, 6, 2)
    expected = arrays["heldout_reference"]["temperature_k"]
    assert np.max(abs(arrays["interpolation"]["temperature_k"] - expected)) < 1e-10
    assert arrays["inference"]["delta_m"].shape == (24, 3)
    assert np.all(arrays["inference"]["success"])
    assert np.max(abs(arrays["inference"]["reference_residual_k"])) < 1e-8


def test_four_spawned_workers_preserve_trace_order_values_and_cache(
    tmp_path, monkeypatch
):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from dataclasses import asdict

    from horizon_position_d5.ensemble import _initialize_worker, _trace_worker
    from horizon_position_d5.pilot import HEIGHT_OFFSETS
    from horizon_position_d5.reference import BilinearDEM, trace_horizon

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    east, north = np.meshgrid(np.arange(16.0), np.arange(15.0))
    terrain = 0.03 * east * north + np.sin(east) - 0.2 * north
    np.savez(inputs / "dem.npz", dem=terrain, res=1.0, e0_px=0, n0_px=0)
    dem = BilinearDEM.from_npz(inputs / "dem.npz")
    bearings = np.array([0.0, 31.5, 90.0, 183.0, 270.0, 351.0])
    points = np.array(
        [
            [6.1, 5.2, 8.0],
            [7.3, 5.1, 8.2],
            [6.6, 6.2, 7.9],
            [5.9, 7.1, 8.1],
            [7.2, 7.0, 8.3],
            [5.5, 5.8, 8.4],
        ]
    )
    expected = [
        asdict(trace_horizon(dem, point, bearings, height_offsets_m=HEIGHT_OFFSETS))
        for point in points
    ]
    identity = {"test": "parallel-trace-equivalence", "workers": 4}
    output = tmp_path / "parallel"
    with ProcessPoolExecutor(
        max_workers=4,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(inputs, output, identity, bearings),
    ) as pool:
        first = list(pool.map(_trace_worker, points, chunksize=1))
        second = list(pool.map(_trace_worker, points[::-1], chunksize=1))
    for results, truths, reused in (
        (first, expected, False),
        (second, expected[::-1], True),
    ):
        for (arrays, checkpoint), truth in zip(results, truths, strict=True):
            assert checkpoint["reused"] is reused
            assert arrays.keys() == truth.keys()
            for name in truth:
                np.testing.assert_array_equal(arrays[name], truth[name])
    assert len(list((output / "traces").glob("*.npz"))) == len(points)
