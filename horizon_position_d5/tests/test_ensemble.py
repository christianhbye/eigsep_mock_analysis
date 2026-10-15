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
