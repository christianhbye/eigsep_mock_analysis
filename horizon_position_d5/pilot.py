"""Checkpointed D5 temperature derivative pilot; all products are diagnostic.

Run on Limbo with --inputs DIR --output DIR. The first eight time/pointing
pairs are observed night coordinates; the last four are synthetic tilts.
The 30 m and 91 m horizons at these coordinates are synthetic height controls.
"""
# ruff: noqa: E402

import argparse
import json
import resource
import time
from dataclasses import asdict
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import croissant as cro
import jax.numpy as jnp
import numpy as np
from eigsim.horizon import grid_bearings_to_az, open_sky_weight

from .cache import BlockCache, atomic_json, file_sha256, fingerprint
from .candidates import position_tangents
from .inputs import SITE, load_inputs, setup_iers
from .reference import BilinearDEM, trace_horizon
from .runtime import runtime_record
from .temperature import affine_response, apply_response, make_temperature_functions

HEIGHT_OFFSETS = (-61.0, -3.5, 0.0)
STEPS_M = (1.0, 0.3, 0.1, 0.03, 0.01, 0.004, 0.002, 0.001)


def summary(values):
    values = np.asarray(values)
    return {
        "signed_mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "p95_abs": float(np.percentile(abs(values), 95)),
        "max_abs": float(np.max(abs(values))),
    }


def pointings(inp):
    """Append explicitly synthetic drive angles without changing sky epoch."""
    take = np.array([0, 2, 4, 7])
    return {
        "times_jd": np.r_[inp["times_jd"], inp["times_jd"][take]],
        "elevations_deg": np.r_[inp["elevations_deg"], [45.0, 60.0, 75.0, 30.0]],
        "azimuths_deg": np.r_[inp["azimuths_deg"], [0.0, 90.0, 180.0, 270.0]],
        "observed_coordinate": np.r_[np.ones(8, bool), np.zeros(4, bool)],
    }


class Study:
    """One source/input/config identity and atomic trace/response checkpoints."""

    def __init__(
        self, inputs, output, *, angular_control=True, split_orientation=False
    ):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.inp = load_inputs(inputs)
        self.pointing = pointings(self.inp)
        self.runtime = runtime_record()
        if not self.runtime["jax_enable_x64"]:
            raise RuntimeError("x64 is required")
        self.iers = setup_iers(inputs)
        self.config = {
            "n_observed_times": 8,
            "angular_control": bool(angular_control),
            "fuse_orientation": not split_orientation,
            "jit_response_adjoint": not split_orientation,
            "synthetic_tilts": 4,
            "frequencies_mhz": self.inp["frequencies_mhz"].tolist(),
            "lmax": self.inp["lmax"],
            "height_offsets_m": list(HEIGHT_OFFSETS),
            "steps_m": list(STEPS_M),
            "ground_temperature_k": 300.0,
            "sky_nside": 64,
            "sky_niter": 3,
            "site": SITE,
            "inner_cutoff_m": 0.25,
            "earth_radius_m": 6371e3,
            "iers": self.iers,
            "scope": self.inp["scope"],
        }
        self.identity = {
            "source_commit": self.runtime["source_commit"],
            "source_tree": self.runtime["source_tree"],
            "lock_sha256": self.runtime["lock_sha256"],
            "input_hashes": self.inp["input_hashes"],
            "config": self.config,
            "versions": self.runtime["versions"],
        }
        self.cache = BlockCache(self.output / "blocks", self.identity)
        existing = self.output / "RUN_RECORD.json"
        if existing.exists():
            previous = json.loads(existing.read_text())
            if previous["identity_hash"] != fingerprint(self.identity):
                raise ValueError("output directory belongs to a different run identity")
        self.record = {
            "schema_version": 1,
            "label": "diagnostic",
            "phase": "temperature-pilot",
            "identity": self.identity,
            "identity_hash": fingerprint(self.identity),
            "runtime": self.runtime,
            "rows": self.inp["rows"].tolist(),
            "reference_epoch_jd": float(self.pointing["times_jd"][0]),
            "sky_source": self.inp["sky_source"],
        }
        atomic_json(existing, self.record)
        self.inputs_path = Path(inputs)
        self.dem = None
        self.trace_timings = []
        self.exports = {}

    def block(self, name, parameters, calculate):
        start = time.monotonic()
        result = self.cache.load(name, parameters)
        reused = result is not None
        if result is None:
            result = calculate()
            self.cache.write(name, parameters, result)
        print(
            json.dumps(
                {
                    "checkpoint": name,
                    "seconds": time.monotonic() - start,
                    "reused": reused,
                }
            ),
            flush=True,
        )
        return result

    def bearings(self, spacing_deg=0.1):
        if spacing_deg == 0.1:
            return self.inp["team_horizon"]["bearings_deg"].copy()
        return np.round(np.arange(0, 360, spacing_deg), 6)

    def trace(self, position, spacing_deg=0.1, method="continuous"):
        parameters = {
            "position_enu_m": np.asarray(position).tolist(),
            "bearing_spacing_deg": spacing_deg,
            "bearings_hash": fingerprint(self.bearings(spacing_deg).tolist()),
            "method": method,
            "height_offsets_m": list(HEIGHT_OFFSETS),
            "inner_cutoff_m": 0.25,
            "earth_radius_m": 6371e3,
        }
        name = "trace-" + fingerprint(parameters)[:20]

        def calculate():
            if self.dem is None:
                self.dem = BilinearDEM.from_npz(self.inputs_path / "dem.npz")
            start = time.monotonic()
            result = asdict(
                trace_horizon(
                    self.dem,
                    position,
                    self.bearings(spacing_deg),
                    height_offsets_m=HEIGHT_OFFSETS,
                    method=method,
                )
            )
            self.trace_timings.append(time.monotonic() - start)
            return result

        return self.block(name, parameters, calculate)

    def save(self, name, arrays):
        self.cache.write(name, {"result": name}, arrays)
        self.exports[name] = {
            "data": f"blocks/{name}.npz",
            "metadata": f"blocks/{name}.json",
            "data_sha256": file_sha256(self.output / f"blocks/{name}.npz"),
            "metadata_sha256": file_sha256(self.output / f"blocks/{name}.json"),
        }

    def response(self):
        inp = self.inp
        blocks = []
        errors = []
        durations = []
        for f, frequency in enumerate(inp["frequencies_mhz"]):
            parameters = {
                "frequency_mhz": float(frequency),
                "pointing": {k: v.tolist() for k, v in self.pointing.items()},
            }

            def calculate(f=f):
                start = time.monotonic()
                sky = cro.Sky(
                    inp["sky_maps"][f : f + 1],
                    inp["frequencies_mhz"][f : f + 1],
                    sampling="healpix",
                    coord="galactic",
                    niter=3,
                )
                fast, direct = make_temperature_functions(
                    inp["beam"][f : f + 1],
                    inp["frequencies_mhz"][f : f + 1],
                    sky,
                    self.pointing["times_jd"],
                    self.pointing["elevations_deg"],
                    self.pointing["azimuths_deg"],
                    misalignment=inp["misalignment"],
                    site=SITE,
                    fuse_orientation=self.config["fuse_orientation"],
                )
                offset, kernel = affine_response(
                    fast,
                    inp["beam"].shape[1:],
                    sequential=True,
                    jit_adjoint=self.config["jit_response_adjoint"],
                )
                print(f"RESPONSE COMPUTED {frequency} MHz", flush=True)
                rng = np.random.default_rng(1807)
                az = np.linspace(0, 2 * np.pi, 360, endpoint=False)
                weights = [
                    rng.uniform(0.1, 0.9, inp["beam"].shape[1:]),
                    np.asarray(
                        open_sky_weight(0.15 + 0.04 * np.sin(az), az, inp["lmax"])
                    ),
                ]
                exact, cached = [], []
                for weight in weights:
                    exact.append(np.asarray(direct(weight)))
                    cached.append(np.asarray(apply_response(offset, kernel, weight)))
                exact, cached = np.stack(exact), np.stack(cached)
                if not np.allclose(exact, cached, atol=1e-8, rtol=1e-10):
                    raise RuntimeError("affine cache failed independent public calls")
                return {
                    "offset_k": offset,
                    "kernel_k": kernel,
                    "control_exact_k": exact,
                    "control_cached_k": cached,
                    "seconds": np.array(time.monotonic() - start),
                }

            block = self.block(f"response-{f}", parameters, calculate)
            blocks.append(block)
            errors.append(block["control_cached_k"] - block["control_exact_k"])
            durations.append(float(block["seconds"]))
            # Python/JAX transform closures can retain substantial compilations.
            jax.clear_caches()
        offset = np.concatenate([b["offset_k"] for b in blocks], axis=1)
        kernel = np.concatenate([b["kernel_k"] for b in blocks], axis=1)
        self.save(
            "response_controls",
            {
                "exact_k": np.concatenate(
                    [b["control_exact_k"] for b in blocks], axis=2
                ),
                "cached_k": np.concatenate(
                    [b["control_cached_k"] for b in blocks], axis=2
                ),
                "seconds": np.asarray(durations),
            },
        )
        self.save("response", {"offset_k": offset, "kernel_k": kernel})
        return offset, kernel


def temperature_model(offset, kernel, bearings_deg, lmax, sub=180):
    az = grid_bearings_to_az(bearings_deg)
    order = np.argsort(az)

    @jax.jit
    def model(alpha):
        return apply_response(
            offset, kernel, open_sky_weight(alpha[order], az[order], lmax, sub=sub)
        )

    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-angular",
        action="store_true",
        help="record angular control as not run; no convergence claim",
    )
    parser.add_argument(
        "--response-only",
        action="store_true",
        help="checkpoint response controls for the timing gate, then stop",
    )
    parser.add_argument(
        "--split-orientation",
        action="store_true",
        help="separate transform JITs; unchanged model/resolution, smaller graphs",
    )
    args = parser.parse_args(argv)
    study = Study(
        args.inputs,
        args.output,
        angular_control=not args.skip_angular,
        split_orientation=args.split_orientation,
    )
    study.inputs_path = args.inputs
    inp, p = study.inp, study.inp["nominal_enu_m"]
    bearings = study.bearings()
    study.save(
        "coordinates",
        {
            **study.pointing,
            "rows": inp["rows"],
            "frequencies_mhz": inp["frequencies_mhz"],
            "nominal_enu_m": p,
            "heights_m": np.array([30.0, 87.5, 91.0]),
            "bearings_deg": bearings,
        },
    )
    nominal = study.trace(p)
    sampled = study.trace(p, method="sampled")
    team = np.stack(
        [inp["team_horizon"][f"elev_rad_{e}"] for e in ("30m", "87.5m", "91m")]
    )
    reproduction = float(np.max(abs(team - sampled["alpha_rad"])))
    if reproduction > 1e-14:
        raise RuntimeError(
            f"sampled reference does not reproduce product: {reproduction}"
        )
    offset, kernel = study.response()
    if args.response_only:
        atomic_json(
            study.output / "RESPONSE_DONE.json",
            {
                **study.record,
                "files": study.exports,
                "response_wall_seconds": time.monotonic() - study.started,
            },
        )
        return
    model = temperature_model(offset, kernel, bearings, inp["lmax"])

    def evaluate(alpha):
        return np.stack([np.asarray(model(a)) for a in alpha])

    base_t = np.stack([evaluate(sampled["alpha_rad"]), evaluate(nominal["alpha_rad"])])
    candidates, horizon_candidates = [], []
    for trace in (sampled, nominal):
        for method in ("e7", "e7prime"):
            tangent = position_tangents(
                trace["alpha_rad"],
                trace["distance_m"],
                trace["gradient_en"],
                bearings,
                method,
            )
            horizon_candidates.append(tangent)
            candidates.append(
                np.stack(
                    [
                        np.stack(
                            [
                                np.asarray(
                                    jax.jvp(
                                        model,
                                        (jnp.asarray(alpha),),
                                        (jnp.asarray(t[:, i]),),
                                    )[1]
                                )
                                for i in range(3)
                            ],
                            axis=-1,
                        )
                        for alpha, t in zip(trace["alpha_rad"], tangent, strict=True)
                    ]
                )
            )
    candidates = np.stack(candidates)
    study.save(
        "nominal",
        {
            "sampled_alpha_rad": sampled["alpha_rad"],
            "continuous_alpha_rad": nominal["alpha_rad"],
            "distance_m": nominal["distance_m"],
            "edge_limited": nominal["edge_limited"],
            "temperature_k": base_t,
            "candidate_jacobian_k_per_m": candidates,
            "candidate_horizon_jacobian_rad_per_m": np.stack(horizon_candidates),
            "candidate_names": np.array(
                ["sampled_e7", "sampled_e7prime", "continuous_e7", "continuous_e7prime"]
            ),
        },
    )
    displaced, alpha_displaced = [], []
    for h in STEPS_M:
        by_axis, alpha_axis = [], []
        for axis in range(3):
            traces = [study.trace(p + sign * h * np.eye(3)[axis]) for sign in (-1, 1)]
            alpha_axis.append(np.stack([t["alpha_rad"] for t in traces]))
            by_axis.append(np.stack([evaluate(t["alpha_rad"]) for t in traces]))
        displaced.append(np.stack(by_axis))
        alpha_displaced.append(np.stack(alpha_axis))
    displaced, alpha_displaced = np.stack(displaced), np.stack(alpha_displaced)
    hscale = np.asarray(STEPS_M).reshape(-1, 1, 1, 1, 1)
    left = (base_t[1] - displaced[:, :, 0]) / hscale
    right = (displaced[:, :, 1] - base_t[1]) / hscale
    derivative = (left + right) / 2
    study.save(
        "displacements",
        {
            "steps_m": np.asarray(STEPS_M),
            "temperature_k": displaced,
            "alpha_rad": alpha_displaced,
            "central_k_per_m": derivative,
            "left_k_per_m": left,
            "right_k_per_m": right,
        },
    )
    # Independent alpha-chain perturbations, with both sides retained.
    tangent = 0.7 + 0.2 * np.cos(2 * np.radians(bearings))
    alpha_steps = np.array([1e-4, 1e-5, 1e-6, 1e-7])
    chain_ad = np.stack(
        [
            np.asarray(jax.jvp(model, (jnp.asarray(a),), (jnp.asarray(tangent),))[1])
            for a in nominal["alpha_rad"]
        ]
    )
    chain_fd, chain_left, chain_right = [], [], []
    for step in alpha_steps:
        a, b = (
            evaluate(nominal["alpha_rad"] - step * tangent),
            evaluate(nominal["alpha_rad"] + step * tangent),
        )
        chain_left.append((base_t[1] - a) / step)
        chain_right.append((b - base_t[1]) / step)
        chain_fd.append((b - a) / (2 * step))
    study.save(
        "alpha_chain",
        {
            "steps_rad": alpha_steps,
            "ad_k_per_rad": chain_ad,
            "central_k_per_rad": np.stack(chain_fd),
            "left_k_per_rad": np.stack(chain_left),
            "right_k_per_rad": np.stack(chain_right),
        },
    )
    # Resolve phi integration separately, using exactly the same horizon samples.
    sub_t, sub_d = [], []
    index = STEPS_M.index(0.01)
    for sub in (180, 360, 720):
        f = temperature_model(offset, kernel, bearings, inp["lmax"], sub=sub)
        sub_t.append(np.stack([np.asarray(f(a)) for a in nominal["alpha_rad"]]))
        sub_d.append(
            np.stack(
                [
                    np.stack(
                        [
                            (np.asarray(f(pair[1, era])) - np.asarray(f(pair[0, era])))
                            / 0.02
                            for era in range(3)
                        ]
                    )
                    for pair in alpha_displaced[index]
                ]
            )
        )
    study.save(
        "phi_resolution",
        {
            "subsamples": np.array([180, 360, 720]),
            "temperature_k": np.stack(sub_t),
            "central_k_per_m": np.stack(sub_d),
            "position_step_m": np.array(0.01),
        },
    )
    angular_t, angular_d = [base_t[1]], [derivative[index]]
    angular_spacings = [0.1]
    if not args.skip_angular:
        for spacing in (0.05, 0.025):
            beta = study.bearings(spacing)
            f = temperature_model(offset, kernel, beta, inp["lmax"])
            angular_t.append(
                np.stack(
                    [np.asarray(f(a)) for a in study.trace(p, spacing)["alpha_rad"]]
                )
            )
            axis_d = []
            for axis in range(3):
                a, b = [
                    study.trace(p + sign * 0.01 * np.eye(3)[axis], spacing)["alpha_rad"]
                    for sign in (-1, 1)
                ]
                axis_d.append(
                    np.stack(
                        [
                            (np.asarray(f(y)) - np.asarray(f(x))) / 0.02
                            for x, y in zip(a, b, strict=True)
                        ]
                    )
                )
            angular_d.append(np.stack(axis_d))
            angular_spacings.append(spacing)
    study.save(
        "angular_resolution",
        {
            "spacing_deg": np.asarray(angular_spacings),
            "temperature_k": np.stack(angular_t),
            "central_k_per_m": np.stack(angular_d),
            "position_step_m": np.array(0.01),
        },
    )
    # Differences between successive steps are diagnostic, not certified error bars.
    reference = np.moveaxis(derivative[-1], 0, -1)
    report = {
        "label": "diagnostic",
        "team_reproduction_max_abs_rad": reproduction,
        "sampled_vs_continuous_temperature_k": summary(base_t[0] - base_t[1]),
        "candidate_errors_vs_1mm_k_per_m": [summary(c - reference) for c in candidates],
        "fd_last_two_steps_difference_k_per_m": summary(
            derivative[-1] - derivative[-2]
        ),
        "one_sided_1mm_difference_k_per_m": summary(left[-1] - right[-1]),
        "trace_seconds": study.trace_timings,
        "wall_seconds": time.monotonic() - study.started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "angular_control_run": not args.skip_angular,
        "limits": [
            "fixed beam/sky/site/pointing and receiver-free antenna plane",
            "four frequencies, eight parked night coordinates, four synthetic tilts",
            "30m/91m are synthetic height controls",
            "differences of finite steps are not certified derivative error bounds",
            "no physical position posterior or D5 noise model",
        ],
    }
    atomic_json(study.output / "pilot_report.json", report)
    atomic_json(
        study.output / "RESULTS.json",
        {
            **study.record,
            "complete": True,
            "files": study.exports,
            "reports": {
                "pilot_report.json": file_sha256(study.output / "pilot_report.json")
            },
        },
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
