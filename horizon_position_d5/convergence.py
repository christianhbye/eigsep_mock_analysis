"""Pure NumPy finite-change diagnostics; no tracing, fitting or file writes.

Array coordinates are explicit. A common slope is supported only on the tested
scales; no outcome here certifies a continuum derivative or physical accuracy.
"""

from dataclasses import asdict, dataclass
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class Criteria:
    """Version-1 engineering criteria from the numerical feasibility spec."""

    forward_atol_k: float = 0.00025
    slope_atol_k_per_m: float = 0.001
    slope_rtol: float = 0.01
    angular_slope_fraction: float = 0.25

    def __post_init__(self):
        for name, value in asdict(self).items():
            if (
                not isinstance(value, Real)
                or isinstance(value, (bool, np.bool_))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.angular_slope_fraction > 1 or self.slope_rtol >= 1:
            raise ValueError("relative criteria must be fractions below unity")

    def slope_budget(self, a, b):
        return self.slope_atol_k_per_m + self.slope_rtol * np.maximum(abs(a), abs(b))


def ordered_levels(values, name, *, increasing, minimum=3):
    """Validate rather than silently reorder a refinement coordinate."""
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf" or raw.ndim != 1 or len(raw) < minimum:
        raise ValueError(f"{name} must be a real vector of at least {minimum} levels")
    values = np.array(raw, dtype=float, copy=True)
    differences = np.diff(values) * (1 if increasing else -1)
    if not np.isfinite(values).all() or np.any(values <= 0) or np.any(differences <= 0):
        raise ValueError(f"{name} must be positive, finite and strictly ordered")
    return values


def real_array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be real numeric data")
    return np.asarray(raw, dtype=np.float64)


def boolean_array(value, shape, name):
    array = np.asarray(value)
    if array.dtype.kind != "b" or array.shape != shape:
        raise ValueError(f"{name} must be boolean with shape {shape}")
    return array


def difference_summary(delta, coordinates):
    """Summarize signed differences and retain the worst cell's coordinates.

    Coordinates map dimension names to their 1-D labels in array order.
    Missing/nonfinite values are counted and never replaced by zero.
    """
    delta = real_array(delta, "delta")
    if delta.shape != tuple(len(v) for v in coordinates.values()):
        raise ValueError("difference coordinates do not match array shape")
    finite = np.isfinite(delta)
    values = delta[finite]
    result = {"count": int(delta.size), "finite_count": int(values.size)}
    result["nonfinite_count"] = result["count"] - result["finite_count"]
    if not values.size:
        return dict(
            result,
            signed_mean=None,
            rms=None,
            p95_abs=None,
            max_abs=None,
            worst_coordinates=None,
        )
    worst = np.unravel_index(
        np.argmax(np.where(finite, abs(delta), -np.inf)), delta.shape
    )
    result.update(
        signed_mean=float(np.mean(values)),
        rms=float(np.sqrt(np.mean(values**2))),
        p95_abs=float(np.percentile(abs(values), 95)),
        max_abs=float(np.max(abs(values))),
        worst_coordinates={
            name: np.asarray(labels)[i].item()
            for (name, labels), i in zip(coordinates.items(), worst, strict=True)
        },
    )
    return result


def _angular_comparisons(values, valid, budget):
    """Keep both adjacent comparisons and the joint coarse/fine comparison."""
    pairs = {
        "bearing": (values[:-1], values[1:], valid[:-1] & valid[1:]),
        "mask": (values[:, :-1], values[:, 1:], valid[:, :-1] & valid[:, 1:]),
        "joint": (values[0, 0], values[-1, -1], valid[0, 0] & valid[-1, -1]),
    }
    reports = {}
    for name, (a, b, resolved) in pairs.items():
        with np.errstate(invalid="ignore", over="ignore"):
            delta = b - a
            limit = budget(a, b)
            passed = resolved & np.isfinite(delta) & (abs(delta) <= limit)
        reports[name] = {
            "difference": delta,
            "valid": resolved,
            "within_budget": passed,
        }
    return reports


def _all_angular(reports, trailing_ndim):
    passed = []
    for report in reports.values():
        array = report["within_budget"]
        axes = tuple(range(array.ndim - trailing_ndim))
        passed.append(np.all(array, axis=axes))
    return np.logical_and.reduce(passed)


def evaluate_stencil(
    center_k,
    displaced_k,
    *,
    bearing_spacing_deg,
    mask_subsamples,
    steps_m,
    axes,
    signs,
    cell_coordinates,
    center_present,
    displaced_present,
    edge_limited,
    criteria=Criteria(),
):
    """Classify a complete bearing/mask/position-step cross product.

    ``center_k`` has shape (bearing, mask, *cells), ``displaced_k`` has
    (bearing, mask, step, axis, sign, *cells). ``cell_coordinates`` is an
    ordered mapping of cell dimension names to unique labels (typically
    position ID, height, sample ID, frequency). Sample IDs distinguish times
    that recur at synthetic pointings. Axes/signs may be permuted but must
    contain exactly E/N/U and -1/+1. Returned slopes use E/N/U, minus/plus.

    Presence masks have the corresponding temperature shapes. ``edge_limited``
    has (*cells) shape and must conservatively flag any relevant trace in the
    assessed finest angular triplet. Present nonfinite data are unresolved;
    absent required data are not_run. Edge-limited cells never pass.

    The finest three consecutive angular levels are assessed. All h levels
    contribute angular comparisons; the finest three h levels set step
    stability. Earlier angular/step verdicts can be evaluated with prefixes.
    """
    bearing = ordered_levels(bearing_spacing_deg, "bearing", increasing=False)
    mask = ordered_levels(mask_subsamples, "mask", increasing=True)
    if np.any(mask != np.floor(mask)):
        raise ValueError("mask_subsamples must be integers")
    if not np.allclose(bearing[:-1] / bearing[1:], 2, rtol=0, atol=1e-12):
        raise ValueError("bearing levels must be consecutive halvings")
    if not np.array_equal(mask[1:], 2 * mask[:-1]):
        raise ValueError("mask levels must be consecutive doublings")
    steps = ordered_levels(steps_m, "steps", increasing=False)
    if tuple(sorted(axes)) != ("E", "N", "U") or len(axes) != 3:
        raise ValueError("axes must contain E, N, U once")
    if (
        len(signs) != 2
        or set(signs) != {-1, 1}
        or any(isinstance(s, (bool, np.bool_)) for s in signs)
    ):
        raise ValueError("signs must contain -1 and +1 once")
    if not cell_coordinates:
        raise ValueError("cell_coordinates must name every cell dimension")
    coordinates = {}
    for name, labels in cell_coordinates.items():
        labels = np.asarray(labels)
        if not isinstance(name, str) or labels.ndim != 1 or not labels.size:
            raise ValueError("cell coordinates must be named nonempty vectors")
        if labels.dtype.kind not in "iufUS" or len(np.unique(labels)) != len(labels):
            raise ValueError(
                f"coordinate {name} must contain unique real/string labels"
            )
        if labels.dtype.kind in "iuf" and not np.isfinite(labels).all():
            raise ValueError(f"coordinate {name} is nonfinite")
        coordinates[name] = labels.copy()
    shape = tuple(len(v) for v in coordinates.values())
    base = real_array(center_k, "center_k")
    shifted = real_array(displaced_k, "displaced_k")
    if base.shape != (len(bearing), len(mask), *shape) or shifted.shape != (
        len(bearing),
        len(mask),
        len(steps),
        3,
        2,
        *shape,
    ):
        raise ValueError("temperature arrays do not match explicit coordinates")
    base_present = boolean_array(center_present, base.shape, "center_present")
    shifted_present = boolean_array(
        displaced_present, shifted.shape, "displaced_present"
    )
    edge = boolean_array(edge_limited, shape, "edge_limited")
    axis_order = [list(axes).index(axis) for axis in ("E", "N", "U")]
    sign_order = [list(signs).index(sign) for sign in (-1, 1)]
    shifted = np.take(np.take(shifted, axis_order, axis=3), sign_order, axis=4)
    shifted_present = np.take(
        np.take(shifted_present, axis_order, axis=3), sign_order, axis=4
    )
    base, shifted = base[-3:, -3:], shifted[-3:, -3:]
    base_present = base_present[-3:, -3:]
    shifted_present = shifted_present[-3:, -3:]
    base = np.where(base_present, base, np.nan)
    shifted = np.where(shifted_present, shifted, np.nan)
    scale = steps.reshape((1, 1, len(steps), 1, *([1] * len(shape))))
    with np.errstate(invalid="ignore", over="ignore"):
        minus = (base[:, :, None, None] - shifted[:, :, :, :, 0]) / scale
        plus = (shifted[:, :, :, :, 1] - base[:, :, None, None]) / scale
        central = (shifted[:, :, :, :, 1] - shifted[:, :, :, :, 0]) / (2 * scale)
    slopes = np.stack([minus, plus], axis=4)
    temperatures = np.concatenate(
        [base[:, :, None], shifted.reshape(3, 3, -1, *shape)], axis=2
    )
    presence = np.concatenate(
        [base_present[:, :, None], shifted_present.reshape(3, 3, -1, *shape)], axis=2
    )
    forward = _angular_comparisons(
        temperatures, np.isfinite(temperatures), lambda a, b: criteria.forward_atol_k
    )
    forward_ok = _all_angular(forward, len(shape))
    complete = np.all(presence, axis=(0, 1, 2))
    finite = np.all(np.isfinite(temperatures), axis=(0, 1, 2))
    forward_status = np.where(forward_ok, "pass", "fail")
    forward_status = np.where(finite & ~edge, forward_status, "unresolved")
    forward_status = np.where(complete, forward_status, "not_run")

    angular = _angular_comparisons(
        slopes,
        np.isfinite(slopes),
        lambda a, b: criteria.angular_slope_fraction * criteria.slope_budget(a, b),
    )
    # Retain step, axis and side when reducing only bearing/mask comparisons.
    angular_each_side = _all_angular(angular, 3 + len(shape))
    angular_ok = np.all(angular_each_side, axis=(0, 2))  # axis, *cells
    finest = slopes[-1, -1, -3:]  # step, axis, side, *cells
    with np.errstate(invalid="ignore", over="ignore"):
        step_delta = finest[1:] - finest[:-1]
        step_pass = np.isfinite(step_delta) & (
            abs(step_delta) <= criteria.slope_budget(finest[1:], finest[:-1])
        )
        gap = finest[-2:, :, 1] - finest[-2:, :, 0]
        gap_limit = criteria.slope_budget(finest[-2:, :, 1], finest[-2:, :, 0])
    step_ok = np.all(step_pass, axis=(0, 2))
    same = np.all(np.isfinite(gap) & (abs(gap) <= gap_limit), axis=0)
    distinct = np.all(np.isfinite(gap) & (abs(gap) > gap_limit), axis=0)
    side_ok = angular_ok & step_ok & (forward_status == "pass")[None]
    behavior = np.where(side_ok & same, "common_slope", "unresolved")
    behavior = np.where(side_ok & distinct, "distinct_one_sided", behavior)
    behavior = np.where(complete[None], behavior, "not_run")
    one_sided_status = np.where(side_ok, "pass", "unresolved")
    one_sided_status = np.where(complete[None], one_sided_status, "not_run")
    common_status = np.where(behavior == "common_slope", "pass", "unresolved")
    common_status = np.where(behavior == "distinct_one_sided", "fail", common_status)
    common_status = np.where(complete[None], common_status, "not_run")
    return {
        "criteria": asdict(criteria),
        "cell_coordinates": coordinates,
        "bearing_spacing_deg": bearing[-3:],
        "mask_subsamples": mask[-3:],
        "steps_m": steps,
        "axes": np.array(["E", "N", "U"]),
        "sides": np.array(["minus", "plus"]),
        "minus_k_per_m": minus,
        "plus_k_per_m": plus,
        "central_k_per_m": central,
        "forward_comparisons": forward,
        "slope_comparisons": angular,
        "step_difference_k_per_m": step_delta,
        "step_within_budget": step_pass,
        "side_gap_k_per_m": gap,
        "side_gap_budget_k_per_m": gap_limit,
        "angular_slope_within_budget": angular_each_side,
        "forward_status": forward_status,
        "one_sided_status": one_sided_status,
        "common_slope_status": common_status,
        "behavior": behavior,
        "edge_limited": edge.copy(),
    }
