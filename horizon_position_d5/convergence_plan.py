"""Validated plans and deterministic work inventories for reference convergence.

Planning reads completed, pinned artifacts. It neither traces terrain nor
instantiates BlockCache (whose constructor creates a directory).
"""

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from numbers import Real
from pathlib import Path

import numpy as np

from .cache import file_sha256, fingerprint, load_results
from .convergence import Criteria, ordered_levels, real_array

FRAME = "DEM grid East, grid North, Up; existing DEM vertical datum"


def _digest(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


@dataclass(frozen=True)
class ConvergencePlan:
    """Versioned numerical criteria and pinned upstream result manifests."""

    pilot_results_sha256: str
    ensemble_results_sha256: str
    schema_version: int = 1
    dem_sha256: str | None = None
    bearing_spacing_deg: tuple = (0.1, 0.05, 0.025)
    mask_subsamples: tuple = (180, 360, 720)
    steps_m: tuple = (0.01, 0.003, 0.001)
    half_width_m: float = 3.2
    criteria: Criteria = field(default_factory=Criteria)

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported plan schema_version")
        if self.dem_sha256 is not None:
            _digest(self.dem_sha256, "dem_sha256")
        _digest(self.pilot_results_sha256, "pilot_results_sha256")
        _digest(self.ensemble_results_sha256, "ensemble_results_sha256")
        for name, increasing in (
            ("bearing_spacing_deg", False),
            ("mask_subsamples", True),
            ("steps_m", False),
        ):
            values = ordered_levels(getattr(self, name), name, increasing=increasing)
            if name == "mask_subsamples" and np.any(values != np.floor(values)):
                raise ValueError("mask_subsamples must be integers")
            object.__setattr__(self, name, tuple(values.tolist()))
        for value in self.bearing_spacing_deg:
            count = Decimal("360") / Decimal(str(value))
            if count != count.to_integral_value():
                raise ValueError("bearing spacing must exactly divide 360 degrees")
        if not np.allclose(
            np.asarray(self.bearing_spacing_deg[:-1]) / self.bearing_spacing_deg[1:],
            2,
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError("bearing levels must be consecutive halvings")
        if not np.array_equal(
            np.asarray(self.mask_subsamples[1:]),
            2 * np.asarray(self.mask_subsamples[:-1]),
        ):
            raise ValueError("mask levels must be consecutive doublings")
        if (
            not isinstance(self.half_width_m, Real)
            or isinstance(self.half_width_m, (bool, np.bool_))
            or not np.isfinite(self.half_width_m)
            or self.half_width_m <= max(self.steps_m)
        ):
            raise ValueError("half_width_m must contain the stencil")
        if not isinstance(self.criteria, Criteria):
            raise ValueError("criteria must be Criteria")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("plan must be a JSON object")
        supplied = dict(value)
        try:
            if "criteria" in supplied:
                supplied["criteria"] = Criteria(**supplied["criteria"])
            return cls(**supplied)
        except (TypeError, OverflowError) as exc:
            raise ValueError(f"invalid plan: {exc}") from exc

    def to_dict(self):
        # JSON-normalize tuples so a serialized round trip has one identity.
        result = {
            **asdict(self),
            "bearing_spacing_deg": list(self.bearing_spacing_deg),
            "mask_subsamples": [int(x) for x in self.mask_subsamples],
            "steps_m": list(self.steps_m),
        }
        if self.dem_sha256 is None:
            result.pop("dem_sha256")
        return result


def _index(values, target, name):
    matches = np.flatnonzero(np.asarray(values) == target)
    if len(matches) != 1:
        raise ValueError(f"upstream must contain exactly one {name}={target}")
    return int(matches[0])


def select_stress_position(blocks):
    """Select the old worst 5-cubic vs 3-mm slope, never a new holdout."""
    coords = blocks["coordinates"]
    validation = blocks["validation_coordinates"]
    interp = blocks["interpolation"]
    points = real_array(validation["offsets_m"], "offsets_m")
    indices = np.asarray(validation["derivative_indices"])
    observed = np.asarray(coords["observed_coordinate"])
    heights = real_array(coords["heights_m"], "heights_m")
    times = real_array(coords["times_jd"], "times_jd")
    frequencies = real_array(coords["frequencies_mhz"], "frequencies_mhz")
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not np.isfinite(points).all()
        or len(np.unique(points, axis=0)) != len(points)
    ):
        raise ValueError("invalid upstream position coordinates")
    if (
        indices.ndim != 1
        or indices.dtype.kind not in "iu"
        or not indices.size
        or np.any(indices < 0)
        or np.any(indices >= len(points))
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("invalid derivative position indices")
    if (
        observed.dtype.kind != "b"
        or observed.ndim != 1
        or not observed.any()
        or times.shape != observed.shape
        or frequencies.ndim != 1
        or not np.isfinite(times).all()
        or not np.isfinite(frequencies).all()
        or len(np.unique(frequencies)) != len(frequencies)
    ):
        raise ValueError("invalid time/frequency/observed coordinates")
    height = _index(heights, 87.5, "height_m")
    grid = _index(interp["grid_points"], 5, "grid_points")
    method = _index(interp["methods"], "cubic", "method")
    step = _index(validation["steps_m"], 0.003, "step_m")
    jac = real_array(interp["jacobian_k_per_m"], "jacobian")
    direct = real_array(blocks["heldout_derivatives"]["central_k_per_m"], "direct")
    if jac.shape != (
        len(interp["grid_points"]),
        len(interp["methods"]),
        len(points),
        len(heights),
        len(times),
        len(frequencies),
        3,
    ):
        raise ValueError("upstream interpolator Jacobian shape mismatch")
    if direct.shape != (
        len(indices),
        len(validation["steps_m"]),
        3,
        len(heights),
        len(times),
        len(frequencies),
    ):
        raise ValueError("upstream direct derivative shape mismatch")
    candidate = np.moveaxis(jac[grid, method][indices, height][:, observed], -1, 1)
    reference = direct[:, step, :, height][:, :, observed]
    errors = candidate - reference
    if not np.isfinite(errors).all():
        raise ValueError("cannot select worst point from nonfinite upstream errors")
    worst = np.argwhere(abs(errors) == np.max(abs(errors)))
    observed_indices = np.flatnonzero(observed)

    def tie_key(item):
        p, axis, t, f = item
        return (
            *points[indices[p]],
            int(axis),
            times[observed_indices[t]],
            frequencies[f],
            int(observed_indices[t]),
        )

    p, axis, t, f = min(worst, key=tie_key)
    index = int(indices[p])
    return {
        "offset_m": points[index].tolist(),
        "validation_index": index,
        "axis": ("E", "N", "U")[axis],
        "height_m": 87.5,
        "time_jd": float(times[observed_indices[t]]),
        "sample_index": int(observed_indices[t]),
        "frequency_mhz": float(frequencies[f]),
        "signed_discrepancy_k_per_m": float(errors[p, axis, t, f]),
        "absolute_discrepancy_k_per_m": float(abs(errors[p, axis, t, f])),
        "role": "development stress point selected from previous results",
        "rule": "max abs(5-cubic Jacobian minus 3-mm central retrace), observed 87.5m",
        "tie_break": "lexicographic offset, E/N/U axis, time, frequency, sample index",
    }


def load_upstreams(plan, pilot_dir, ensemble_dir):
    """Verify pinned exports and their cross-phase coordinate/input identity."""
    manifests, blocks = [], []
    for directory, digest, phase in (
        (pilot_dir, plan.pilot_results_sha256, "temperature-pilot"),
        (ensemble_dir, plan.ensemble_results_sha256, "position-ensemble"),
    ):
        directory = Path(directory)
        if file_sha256(directory / "RESULTS.json") != digest:
            raise ValueError(f"pinned {phase} RESULTS hash mismatch")
        manifest, arrays = load_results(directory, expected_phase=phase)
        for relative, expected in manifest.get("reports", {}).items():
            path = (directory / relative).resolve()
            if (
                not path.is_relative_to(directory.resolve())
                or file_sha256(path) != expected
            ):
                raise ValueError("upstream report path/hash mismatch")
        manifests.append(manifest)
        blocks.append(arrays)
    pilot, ensemble = manifests
    config = ensemble["identity"]["config"]
    if (
        config["pilot_results_sha256"] != plan.pilot_results_sha256
        or config["pilot_identity"] != pilot["identity"]
        or ensemble["identity"]["input_hashes"] != pilot["identity"]["input_hashes"]
    ):
        raise ValueError("ensemble does not descend from the pinned pilot/inputs")
    pcoords, ecoords = blocks[0]["coordinates"], blocks[1]["coordinates"]
    if pcoords.keys() != ecoords.keys() or any(
        not np.array_equal(pcoords[key], ecoords[key]) for key in pcoords
    ):
        raise ValueError("upstream coordinate mismatch")
    if not np.array_equal(
        ecoords["nominal_enu_m"], blocks[1]["validation_coordinates"]["nominal_enu_m"]
    ):
        raise ValueError("upstream nominal position mismatch")
    return pilot, ensemble, blocks[0], blocks[1]


def expected_input_hashes(plan, pilot):
    """Keep frozen response inputs; allow only an explicitly pinned DEM.

    Response coefficients precede the horizon mask and retain pilot provenance.
    New terrain invalidates every trace/evaluation through its new identity.
    """
    hashes = dict(pilot["identity"]["input_hashes"])
    if plan.dem_sha256 is not None:
        hashes["dem.npz"] = plan.dem_sha256
    return hashes


def build_inventory(plan, pilot, ensemble, pilot_arrays, ensemble_arrays, *, source):
    """Build content-keyed trace/evaluation tasks without touching any cache.

    ``source`` records planner/source file hashes and lock identity. Old traces
    are never called reused here: actual cache integrity and source compatibility
    must be checked by the future runner. Every task starts as not_run.
    """
    selection = select_stress_position(ensemble_arrays)
    centers = [[0.0, 0.0, 0.0]]
    if selection["offset_m"] not in centers:
        centers.append(selection["offset_m"])
    coords = pilot_arrays["coordinates"]
    nominal = real_array(coords["nominal_enu_m"], "nominal")
    if nominal.shape != (3,) or not np.isfinite(nominal).all():
        raise ValueError("invalid nominal position")
    config = pilot["identity"]["config"]
    height_offsets = config["height_offsets_m"]
    trace_identity = {
        "reference_sha256": source["reference_sha256"],
        "numpy_version": np.__version__,
        "dem_sha256": plan.dem_sha256 or pilot["identity"]["input_hashes"]["dem.npz"],
        "method": "continuous",
        "coordinate_frame": FRAME,
        "height_offsets_m": height_offsets,
        "inner_cutoff_m": config["inner_cutoff_m"],
        "earth_radius_m": config["earth_radius_m"],
    }
    response_identity = {
        "pilot_results_sha256": plan.pilot_results_sha256,
        "response_data_sha256": pilot["files"]["response"]["data_sha256"],
        "response_metadata_sha256": pilot["files"]["response"]["metadata_sha256"],
        "pilot_identity_hash": pilot["identity_hash"],
        "coordinate_hash": fingerprint({k: v.tolist() for k, v in coords.items()}),
    }
    traces, evaluations, stencil = {}, {}, []
    for position_id, center in enumerate(centers):
        locations = [(None, None, 0, center)]
        for h in plan.steps_m:
            for axis in range(3):
                for sign in (-1, 1):
                    offset = list(center)
                    offset[axis] = float(
                        Decimal(str(center[axis])) + sign * Decimal(str(h))
                    )
                    locations.append((h, axis, sign, offset))
        for h, axis, sign, offset in locations:
            if any(abs(x) > plan.half_width_m for x in offset):
                raise ValueError("stencil leaves declared exploration box; do not clip")
            absolute = (nominal + offset).tolist()
            if h is not None and absolute == (nominal + center).tolist():
                raise ValueError(
                    "displacement is lost at absolute-coordinate precision"
                )
            if any(not np.isfinite(x) for x in absolute):
                raise ValueError("nonfinite absolute trace coordinate")
            for spacing in plan.bearing_spacing_deg:
                count = int(Decimal("360") / Decimal(str(spacing)))
                bearings = (np.arange(count) * spacing).tolist()
                params = {
                    "position_enu_m": absolute,
                    "bearings_hash": fingerprint(bearings),
                    "bearing_spacing_deg": spacing,
                    "bearing_count": count,
                }
                trace_id = "trace-" + fingerprint(
                    {"identity": trace_identity, "parameters": params}
                )
                traces.setdefault(
                    trace_id,
                    {"id": trace_id, "parameters": params, "status": "not_run"},
                )
                for sub in plan.mask_subsamples:
                    evaluation = {
                        "trace_id": trace_id,
                        "mask_subsamples": int(sub),
                        "response_identity": response_identity,
                        "temperature_source": source["temperature_sha256"],
                        "evaluation_source": source,
                        "numpy_version": np.__version__,
                    }
                    eval_id = "temperature-" + fingerprint(evaluation)
                    evaluations.setdefault(
                        eval_id,
                        {"id": eval_id, "parameters": evaluation, "status": "not_run"},
                    )
                    stencil.append(
                        {
                            "position_id": position_id,
                            "step_m": h,
                            "axis": None if axis is None else ("E", "N", "U")[axis],
                            "sign": sign,
                            "bearing_spacing_deg": spacing,
                            "mask_subsamples": int(sub),
                            "trace_id": trace_id,
                            "evaluation_id": eval_id,
                        }
                    )
    inventory = {
        "schema_version": 1,
        "phase": "reference-convergence-plan",
        "label": "diagnostic",
        "execution_started": False,
        "plan": plan.to_dict(),
        "plan_hash": fingerprint(plan.to_dict()),
        "source": source,
        "upstream_identity_hashes": {
            "pilot": pilot["identity_hash"],
            "ensemble": ensemble["identity_hash"],
        },
        "coordinate_frame": FRAME,
        "nominal_enu_m": nominal.tolist(),
        "center_offsets_m": centers,
        "selection": selection,
        "coordinates": {
            k: v.tolist() for k, v in coords.items() if k != "bearings_deg"
        },
        "trace_identity": trace_identity,
        "response_identity": response_identity,
        "traces": list(traces.values()),
        "evaluations": list(evaluations.values()),
        "stencil": stencil,
        "counts": {
            "positions": len(centers),
            "traces": len(traces),
            "evaluations": len(evaluations),
            "stencil_entries": len(stencil),
        },
        "limits": [
            "inventory only; no resource gate or scientific verdict",
            "upstream stress selection is not independent validation",
            "checkpoint reuse must be verified by the runner",
        ],
    }
    inventory["inventory_hash"] = fingerprint(inventory)
    return inventory
