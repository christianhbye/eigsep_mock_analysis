"""Held-out position interpolation and conditional surrogate-inference study.

The completed temperature pilot is a hashed upstream input. Independent
retracing supplies every training/validation horizon; validation coordinates
are frozen before training. Worker processes share the launcher's CPU affinity
and nice level. No source/environment mutation is permitted during the run.
"""
# ruff: noqa: E402

import argparse
import json
import multiprocessing
import resource
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from eigsim.horizon import grid_bearings_to_az, open_sky_weight

from eigsim import PositionHorizonGrid

from .cache import BlockCache, atomic_json, file_sha256, fingerprint, load_results
from .inference import fit_multistart, local_identifiability, reference_resolution
from .inputs import verify_inputs
from .pilot import HEIGHT_OFFSETS, summary, temperature_model
from .reference import BilinearDEM, trace_horizon
from .runtime import runtime_record
from .temperature import apply_response

_WORKER = None


def _initialize_worker(inputs, output, identity, bearings):
    global _WORKER
    _WORKER = (
        BilinearDEM.from_npz(Path(inputs) / "dem.npz"),
        BlockCache(Path(output) / "traces", identity),
        np.asarray(bearings),
    )


def _trace_worker(position):
    dem, cache, bearings = _WORKER
    params = {
        "position_enu_m": np.asarray(position).tolist(),
        "height_offsets_m": list(HEIGHT_OFFSETS),
        "bearings_hash": fingerprint(bearings.tolist()),
        "method": "continuous",
        "inner_cutoff_m": 0.25,
        "earth_radius_m": 6371e3,
    }
    name = "trace-" + fingerprint(params)[:20]
    start = time.monotonic()
    arrays = cache.load(name, params)
    reused = arrays is not None
    if arrays is None:
        arrays = asdict(
            trace_horizon(dem, position, bearings, height_offsets_m=HEIGHT_OFFSETS)
        )
        cache.write(name, params, arrays)
    return arrays, {
        "checkpoint": name,
        "seconds": time.monotonic() - start,
        "reused": reused,
    }


def validation_positions():
    """Two independent fixed seeds plus face/corner-near points, in metres."""
    random = [
        np.random.default_rng(seed).uniform(-3.0, 3.0, (4, 3))
        for seed in (192840, 716933)
    ]
    faces = np.array(
        [
            [3.19, 0.37, -0.53],
            [-3.19, 0.37, -0.53],
            [0.37, 3.19, -0.53],
            [0.37, -3.19, -0.53],
            [0.37, -0.53, 3.19],
            [0.37, -0.53, -3.19],
        ]
    )
    corners = np.array(
        [
            [3.19, 3.19, 3.19],
            [-3.19, 3.19, -3.19],
            [3.19, -3.19, -3.19],
            [-3.19, -3.19, 3.19],
        ]
    )
    return np.concatenate([*random, faces, corners])


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


class Ensemble:
    def __init__(self, args):
        self.args = args
        self.started = time.monotonic()
        self.pilot, self.arrays = load_results(
            args.pilot, expected_phase="temperature-pilot"
        )
        hashes = {k: v["sha256"] for k, v in verify_inputs(args.inputs).items()}
        if hashes != self.pilot["identity"]["input_hashes"]:
            raise ValueError("ensemble inputs differ from pilot")
        runtime = runtime_record()
        self.coords = self.arrays["coordinates"]
        self.p = self.coords["nominal_enu_m"]
        self.bearings = self.coords["bearings_deg"]
        self.offset = self.arrays["response"]["offset_k"]
        self.kernel = self.arrays["response"]["kernel_k"]
        self.lmax = self.pilot["identity"]["config"]["lmax"]
        self.model = temperature_model(
            self.offset, self.kernel, self.bearings, self.lmax
        )
        self.points = validation_positions()
        self.derivative_indices = np.array([0, 4, 8, 14])
        self.fd_steps = np.array([0.01, 0.003])
        self.config = {
            "grid_points": list(args.grid_points),
            "half_width_m": 3.2,
            "validation_offsets_m": self.points.tolist(),
            "random_seeds": [192840, 716933],
            "derivative_indices": self.derivative_indices.tolist(),
            "derivative_steps_m": self.fd_steps.tolist(),
            "inference_injection_indices": [0, 4],
            "optimizer_starts": "zero, known injection, and four signed 2m corners",
            "declared_noise_scales_k": [0.001, 0.01, 0.1, 1.0],
            "inference_scope": "noiseless injections, observed 87.5m coordinates only",
            "nuisance": "fixed or one unconstrained global gain plus global K offset",
            "position_prior": "uniform +/-3.2m box; diagnostic, not D5 posterior",
            "workers": args.workers,
            "pilot_identity": self.pilot["identity"],
            "pilot_rows": self.pilot["rows"],
            "reference_epoch_jd": self.pilot["reference_epoch_jd"],
            "pilot_results_sha256": file_sha256(args.pilot / "RESULTS.json"),
        }
        self.identity = {
            "source_commit": runtime["source_commit"],
            "source_tree": runtime["source_tree"],
            "lock_sha256": runtime["lock_sha256"],
            "input_hashes": hashes,
            "config": self.config,
            "versions": runtime["versions"],
        }
        self.record = {
            "schema_version": 1,
            "label": "diagnostic",
            "phase": "position-ensemble",
            "identity": self.identity,
            "identity_hash": fingerprint(self.identity),
            "runtime": runtime,
        }
        args.output.mkdir(parents=True, exist_ok=True)
        record_path = args.output / "RUN_RECORD.json"
        if record_path.exists():
            if (
                json.loads(record_path.read_text())["identity_hash"]
                != self.record["identity_hash"]
            ):
                raise ValueError("output belongs to another run identity")
        atomic_json(record_path, self.record)
        self.cache = BlockCache(args.output / "blocks", self.identity)
        self.exports = {}
        self.timings = []
        self.pool = ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(args.inputs, args.output, self.identity, self.bearings),
        )

    def traces(self, offsets):
        results = []
        for arrays, checkpoint in self.pool.map(
            _trace_worker, self.p + np.asarray(offsets), chunksize=1
        ):
            results.append(arrays["alpha_rad"])
            self.timings.append(checkpoint)
            print(json.dumps(checkpoint), flush=True)
        return np.stack(results)

    def temperatures(self, curves):
        shape = curves.shape[:-1]
        flat = curves.reshape(-1, curves.shape[-1])
        values = np.stack([np.asarray(self.model(a)) for a in flat])
        return values.reshape((*shape, *values.shape[1:]))

    def save(self, name, arrays):
        self.cache.write(name, {"result": name}, arrays)
        self.exports[name] = {
            "data": f"blocks/{name}.npz",
            "metadata": f"blocks/{name}.json",
            "data_sha256": file_sha256(self.args.output / f"blocks/{name}.npz"),
            "metadata_sha256": file_sha256(self.args.output / f"blocks/{name}.json"),
        }

    def interpolator(self, axes, values, era, method):
        az = grid_bearings_to_az(self.bearings)
        order = np.argsort(az)
        grid = PositionHorizonGrid(
            axes, az[order], values[:, :, :, era, order], method=method
        )

        @jax.jit
        def model(delta):
            return apply_response(
                self.offset,
                self.kernel,
                open_sky_weight(grid(delta), az[order], self.lmax),
            )

        return grid, model, jax.jit(jax.jacfwd(model))

    def run(self):
        self.save("coordinates", self.coords)
        # This immutable inventory is written before a single training trace.
        self.save(
            "validation_coordinates",
            {
                "offsets_m": self.points,
                "derivative_indices": self.derivative_indices,
                "steps_m": self.fd_steps,
                "nominal_enu_m": self.p,
            },
        )
        truth_alpha = self.traces(self.points)
        truth = self.temperatures(truth_alpha)
        base = self.arrays["nominal"]["temperature_k"]
        candidate_jac = self.arrays["nominal"]["candidate_jacobian_k_per_m"]
        candidate_values = np.stack(
            [
                base[method // 2] + np.einsum("etfi,pi->petf", jac, self.points)
                for method, jac in enumerate(candidate_jac)
            ]
        )
        self.save(
            "heldout_reference",
            {
                "alpha_rad": truth_alpha,
                "temperature_k": truth,
                "linear_candidate_temperature_k": candidate_values,
            },
        )
        training, predictions, gradients, overshoots = {}, [], [], []
        for count in self.args.grid_points:
            axis = np.linspace(-3.2, 3.2, count)
            axes = (axis, axis, axis)
            nodes = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
            if np.any(
                np.all(
                    np.isclose(self.points[:, None], nodes[None], atol=1e-12, rtol=0),
                    axis=-1,
                )
            ):
                raise ValueError("validation point is a training node")
            values = self.traces(nodes).reshape(
                count, count, count, 3, len(self.bearings)
            )
            training[count] = (axes, values)
            self.save(f"training_{count}", {"axis_m": axis, "alpha_rad": values})
            by_method, jac_method, over_method = [], [], []
            for method in ("linear", "cubic"):
                era_values, era_jac, era_over = [], [], []
                for era in range(3):
                    grid, model, jacobian = self.interpolator(axes, values, era, method)
                    curves = np.stack([np.asarray(grid(p)) for p in self.points])
                    overshoot = np.max(abs(curves), axis=-1) > np.pi / 2
                    if np.any(overshoot):
                        raise RuntimeError(
                            "emulator leaves valid horizon range; do not clip"
                        )
                    era_values.append(
                        np.stack([np.asarray(model(p)) for p in self.points])
                    )
                    era_jac.append(
                        np.stack([np.asarray(jacobian(p)) for p in self.points])
                    )
                    era_over.append(overshoot)
                by_method.append(np.stack(era_values, axis=1))
                jac_method.append(np.stack(era_jac, axis=1))
                over_method.append(np.stack(era_over, axis=1))
            predictions.append(np.stack(by_method))
            gradients.append(np.stack(jac_method))
            overshoots.append(np.stack(over_method))
        predictions, gradients = np.stack(predictions), np.stack(gradients)
        self.save(
            "interpolation",
            {
                "grid_points": np.asarray(self.args.grid_points),
                "methods": np.array(["linear", "cubic"]),
                "temperature_k": predictions,
                "jacobian_k_per_m": gradients,
                "outside_elevation_range": np.stack(overshoots),
            },
        )
        # Reserve the same two independent seed points plus a face and corner
        # for the expensive retraced derivative controls at both step sizes.
        offsets = self.points[self.derivative_indices, None, None, None, :] + (
            self.fd_steps[None, :, None, None, None]
            * np.eye(3)[None, None, :, None, :]
            * np.array([-1, 1])[None, None, None, :, None]
        )
        fd_alpha = self.traces(offsets.reshape(-1, 3)).reshape(
            len(self.derivative_indices),
            len(self.fd_steps),
            3,
            2,
            3,
            len(self.bearings),
        )
        fd_t = self.temperatures(fd_alpha)
        scale = self.fd_steps[None, :, None, None, None, None]
        center = truth[self.derivative_indices, None, None]
        left = (center - fd_t[:, :, :, 0]) / scale
        right = (fd_t[:, :, :, 1] - center) / scale
        self.save(
            "heldout_derivatives",
            {
                "temperature_k": fd_t,
                "central_k_per_m": (left + right) / 2,
                "left_k_per_m": left,
                "right_k_per_m": right,
            },
        )
        self.inference(
            training, truth, candidate_jac, base, (left + right) / 2, left, right
        )
        report = {
            "label": "diagnostic",
            "wall_seconds": time.monotonic() - self.started,
            "peak_parent_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024**2,
            "worker_count": self.args.workers,
            "trace_checkpoints": self.timings,
            "candidate_forward_error_k": [
                summary(value - truth) for value in candidate_values
            ],
            "emulator_forward_error_k": [
                [summary(value - truth) for value in resolution]
                for resolution in predictions
            ],
            "scope": (
                "18 held-out positions; four derivative positions; 4 frequencies; "
                "8 observed and 4 synthetic pointings; 3 height controls"
            ),
            "limits": [
                "no D5 physical posterior",
                "widths are local linear diagnostics",
                "no unrun grid resolution is validated",
                "finite differences can straddle terrain or mask kinks",
            ],
        }
        atomic_json(self.args.output / "ensemble_report.json", _plain(report))
        atomic_json(
            self.args.output / "RESULTS.json",
            {
                **self.record,
                "complete": True,
                "files": self.exports,
                "reports": {
                    name: file_sha256(self.args.output / name)
                    for name in ("ensemble_report.json", "inference_report.json")
                },
            },
        )
        print(json.dumps(_plain(report), indent=2), flush=True)

    def inference(
        self, training, truth, candidate_jac, base, reference_jac, left, right
    ):
        observed = self.coords["observed_coordinate"]
        nobs = int(observed.sum())
        models = []
        for method, jac in enumerate(candidate_jac):
            baseline = base[method // 2, 1, observed]
            j = jac[1, observed]

            def model(delta, baseline=baseline, j=j):
                return baseline + np.einsum("tfi,i->tf", j, delta)

            def derivative(delta, j=j):
                return j

            models.append(
                (
                    str(self.arrays["nominal"]["candidate_names"][method]),
                    model,
                    derivative,
                )
            )
        axes, values = training[max(training)]
        for method in ("linear", "cubic"):
            _, whole, whole_jac = self.interpolator(axes, values, 1, method)

            def model(delta, whole=whole):
                return np.asarray(whole(jnp.asarray(delta)))[:nobs]

            def derivative(delta, whole_jac=whole_jac):
                return np.asarray(whole_jac(jnp.asarray(delta)))[:nobs]

            models.append((f"grid{max(training)}_{method}", model, derivative))
        fits = []
        for injection in (0, 4):
            for nuisance in (False, True):
                for name, model, jacobian in models:
                    fit = fit_multistart(
                        model,
                        jacobian,
                        truth[injection, 1, observed],
                        self.points[injection],
                        nuisance=nuisance,
                    )
                    fit.update(
                        {
                            "model": name,
                            "injection_index": injection,
                            "true_delta_m": self.points[injection],
                            "nuisance": nuisance,
                            "recovered_minus_injected_m": fit["delta_m"]
                            - self.points[injection],
                        }
                    )
                    fits.append(fit)
                    print(
                        json.dumps(
                            {
                                "fit": name,
                                "injection": injection,
                                "nuisance": nuisance,
                                "success": fit["success"],
                                "rank": fit["rank"],
                            }
                        ),
                        flush=True,
                    )
        direct_t = self.temperatures(self.traces([f["delta_m"] for f in fits]))
        reference = []
        for slot, injection in enumerate((0, 4)):
            # reference_jac dimensions: heldout, step, axis, era, time, frequency
            j = np.moveaxis(reference_jac[slot, -1, :, 1, :nobs], 0, -1)
            for nuisance in (False, True):

                def select(value):
                    return np.moveaxis(value[slot, -1, :, 1, :nobs], 0, -1)

                ident = local_identifiability(
                    j, truth[injection, 1, observed], nuisance=nuisance
                )
                resolution = reference_resolution(
                    j,
                    np.moveaxis(reference_jac[slot, 0, :, 1, :nobs], 0, -1),
                    select(left),
                    select(right),
                    truth[injection, 1, observed],
                    nuisance=nuisance,
                )
                reference.append(
                    {
                        "injection_index": injection,
                        "nuisance": nuisance,
                        **ident,
                        **resolution,
                        "reference_width_interpretable": bool(
                            ident["linear_width_valid"]
                            and resolution["derivative_resolved"]
                        ),
                    }
                )
        for fit, direct in zip(fits, direct_t, strict=True):
            direct = fit["gain"] * direct[1, observed] + fit["offset_k"]
            fit["direct_reference_prediction_k"] = direct
            fit["direct_reference_residual_k"] = (
                direct - truth[fit["injection_index"], 1, observed]
            )
            fit["surrogate_minus_reference_k"] = fit["prediction_k"] - direct
        self.save(
            "inference",
            {
                "delta_m": np.stack([f["delta_m"] for f in fits]),
                "true_delta_m": np.stack([f["true_delta_m"] for f in fits]),
                "model": np.array([f["model"] for f in fits]),
                "nuisance": np.array([f["nuisance"] for f in fits]),
                "success": np.array([f["success"] for f in fits]),
                "rank": np.array([f["rank"] for f in fits]),
                "surrogate_residual_k": np.stack([f["residual_k"] for f in fits]),
                "reference_residual_k": np.stack(
                    [f["direct_reference_residual_k"] for f in fits]
                ),
                "linear_sigma_m_per_k": np.stack(
                    [f["linear_sigma_m_per_k"] for f in fits]
                ),
                "linear_width_valid": np.array([f["linear_width_valid"] for f in fits]),
                "noise_scales_k": np.array(self.config["declared_noise_scales_k"]),
            },
        )
        atomic_json(
            self.args.output / "inference_report.json",
            _plain(
                {
                    "label": "diagnostic",
                    "fits": fits,
                    "reference_local_identifiability": reference,
                    "noise_scales_k": self.config["declared_noise_scales_k"],
                    "interpretation": (
                        "Noiseless synthetic injections; local linear widths, "
                        "not posterior intervals. Noise scales are declared, "
                        "not estimated from D5. Bound contact or weak rank "
                        "prevents a physical position claim."
                    ),
                }
            ),
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--grid-points", nargs="+", type=int, default=[3, 5])
    args = parser.parse_args(argv)
    if args.grid_points != sorted(set(args.grid_points)) or min(args.grid_points) < 2:
        raise ValueError("grid point counts must be distinct, ascending and >=2")
    study = Ensemble(args)
    try:
        study.run()
    finally:
        study.pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
