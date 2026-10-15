"""Conditional noiseless-injection diagnostics, not a D5 position posterior."""

import numpy as np
from scipy.optimize import least_squares


def local_identifiability(position_jacobian, temperature, *, nuisance=False):
    """Position rank/width per kelvin, optionally removing global gain/offset.

    Widths use the local linear Gaussian approximation with unit independent
    noise in each cell. A bounded prior, nonlinear likelihood or a physical
    D5 covariance is not represented by these widths. Rank-deficient outputs
    carry an explicit false validity flag and zero placeholders, never a
    claimed finite uncertainty.
    """
    jac = np.asarray(position_jacobian).reshape(-1, 3)
    reference_scale = np.linalg.norm(jac, ord=2)
    if nuisance:
        columns = np.column_stack(
            [np.asarray(temperature).reshape(-1), np.ones(jac.shape[0])]
        )
        jac = jac - columns @ np.linalg.lstsq(columns, jac, rcond=None)[0]
    _, singular, vh = np.linalg.svd(jac, full_matrices=False)
    # Scale to the original position columns: projection roundoff cannot
    # turn a wholly degenerate response into a seemingly full-rank one.
    cutoff = 1e-8 * reference_scale
    rank = int(np.sum(singular > cutoff))
    valid = rank == 3
    widths = np.zeros(3)
    if valid:
        covariance = (vh.T / singular**2) @ vh
        widths = np.sqrt(np.diag(covariance))
    return {
        "singular_values_k_per_m": singular,
        "rank": rank,
        "rank_cutoff_k_per_m": float(cutoff),
        "condition": float(singular[0] / singular[-1]) if valid else None,
        "linear_sigma_m_per_k": widths,
        "linear_width_valid": valid,
    }


def fit_surrogate(
    model, jacobian, truth_k, *, half_width_m=3.2, nuisance=False, initial_delta=None
):
    """Fit a bounded position offset, with optional unconstrained global g,b.

    ``model(delta)`` and ``jacobian(delta)`` return temperatures and an appended
    E/N/U derivative axis. Synthetic data are noiseless; noise scales enter
    only subsequent likelihood/error and local-width diagnostics. The nuisance
    case fits one gain and one additive K offset over all time/frequency cells.
    Unphysical gain and bound contact are reported, not interpreted away.
    """
    truth = np.asarray(truth_k)

    def residual(parameters):
        prediction = np.asarray(model(parameters[:3]))
        if nuisance:
            prediction = parameters[3] * prediction + parameters[4]
        return (prediction - truth).ravel()

    def derivative(parameters):
        jac = np.asarray(jacobian(parameters[:3])).reshape(-1, 3)
        if nuisance:
            jac = np.column_stack(
                [
                    parameters[3] * jac,
                    np.asarray(model(parameters[:3])).ravel(),
                    np.ones(truth.size),
                ]
            )
        return jac

    delta0 = np.zeros(3) if initial_delta is None else np.asarray(initial_delta)
    initial = np.r_[delta0, [1.0, 0.0]] if nuisance else delta0
    low = (
        np.r_[np.full(3, -half_width_m), [-np.inf, -np.inf]]
        if nuisance
        else (np.full(3, -half_width_m))
    )
    high = (
        np.r_[np.full(3, half_width_m), [np.inf, np.inf]]
        if nuisance
        else (np.full(3, half_width_m))
    )
    fit = least_squares(
        residual,
        initial,
        jac=derivative,
        bounds=(low, high),
        x_scale="jac",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=500,
    )
    gain, offset = fit.x[3:] if nuisance else (1.0, 0.0)
    raw = np.asarray(model(fit.x[:3]))
    prediction = gain * raw + offset
    jac = gain * np.asarray(jacobian(fit.x[:3]))
    ident = local_identifiability(jac, raw, nuisance=nuisance)
    return {
        "delta_m": fit.x[:3],
        "gain": float(gain),
        "offset_k": float(offset),
        "prediction_k": prediction,
        "residual_k": prediction - truth,
        "position_jacobian_k_per_m": jac,
        **ident,
        "success": bool(fit.success),
        "status": int(fit.status),
        "message": fit.message,
        "optimality": float(fit.optimality),
        "nfev": int(fit.nfev),
        "bound_contact": bool(np.any(fit.active_mask[:3])),
        "gain_positive": bool(gain > 0),
    }


def reference_resolution(
    small_step, large_step, left, right, temperature, *, nuisance=False
):
    """Assess numerical stability before interpreting a reference Fisher width.

    This is an engineering check, not a certified derivative error bound.
    Require each position column's step and one-sided RMS differences below
    1e-6 K/m + 1e-3 relative, and their operator norm below 0.1 of the smallest
    resolved singular value. Nuisance projection is applied before checking,
    so near cancellation cannot hide poor conditioning of the position fit.
    """
    jacobians = [
        np.asarray(j).reshape(-1, 3) for j in (small_step, large_step, left, right)
    ]
    if nuisance:
        columns = np.column_stack(
            [np.asarray(temperature).ravel(), np.ones(jacobians[0].shape[0])]
        )
        jacobians = [
            j - columns @ np.linalg.lstsq(columns, j, rcond=None)[0] for j in jacobians
        ]
    small, large, minus, plus = jacobians
    step_change = small - large
    half_sided = (plus - minus) / 2
    rms = np.sqrt(np.mean(small**2, axis=0))
    threshold = 1e-6 + 1e-3 * rms
    step_rms = np.sqrt(np.mean(step_change**2, axis=0))
    sided_rms = np.sqrt(np.mean(half_sided**2, axis=0))
    error_proxy = max(
        np.linalg.norm(step_change, ord=2), np.linalg.norm(half_sided, ord=2)
    )
    least_singular = np.linalg.svd(small, compute_uv=False)[-1]
    stable = bool(
        np.all(step_rms <= threshold)
        and np.all(sided_rms <= threshold)
        and error_proxy < 0.1 * least_singular
    )
    return {
        "derivative_resolved": stable,
        "step_change_rms_k_per_m": step_rms,
        "half_one_sided_rms_k_per_m": sided_rms,
        "column_threshold_k_per_m": threshold,
        "operator_difference_proxy_k_per_m": float(error_proxy),
        "smallest_singular_k_per_m": float(least_singular),
    }


def fit_multistart(model, jacobian, truth_k, injected_delta, *, nuisance=False):
    """Compare six deterministic starts, including truth as an optimizer control.

    The known injection is available only in this synthetic experiment. These
    restarts probe ambiguity/local minima; they do not prove global uniqueness.
    Keep all costs and solutions, including failures, for inspection.
    """
    starts = np.array(
        [np.zeros(3), injected_delta, [-2, -2, -2], [2, 2, 2], [-2, 2, -2], [2, -2, 2]]
    )
    fits = [
        fit_surrogate(model, jacobian, truth_k, nuisance=nuisance, initial_delta=start)
        for start in starts
    ]
    eligible = [i for i, fit in enumerate(fits) if fit["success"]]
    if not eligible:
        eligible = list(range(len(fits)))
    costs = np.array([np.sum(fit["residual_k"] ** 2) for fit in fits])
    selected = min(eligible, key=lambda i: costs[i])
    result = fits[selected].copy()
    result["selected_start_index"] = selected
    result["restarts"] = [
        {
            "initial_delta_m": start,
            "delta_m": fit["delta_m"],
            "sum_squared_residual_k2": float(cost),
            "success": fit["success"],
            "status": fit["status"],
            "optimality": fit["optimality"],
            "gain": fit["gain"],
            "offset_k": fit["offset_k"],
        }
        for start, fit, cost in zip(starts, fits, costs, strict=True)
    ]
    result["restart_position_spread_m"] = np.ptp(
        np.stack([fits[i]["delta_m"] for i in eligible]), axis=0
    )
    return result
