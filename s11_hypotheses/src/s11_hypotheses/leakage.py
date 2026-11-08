"""Upstream causal/anti-causal controls, operator and covariance propagation."""

import numpy as np
from eigsep_analysis import s11_diagnostics as sd
from eigsep_analysis import s11_hypotheses as obs


def delayed_lines(freq_hz, delays_s, amplitudes):
    """Generate physical lines on stored coordinates; retain coherent sums."""
    f = np.asarray(freq_hz)
    return np.exp(-2j * np.pi * f[:, None] * np.asarray(delays_s)) @ amplitudes


def controls(freq_hz):
    """Finite-band delayed-line controls; no presumed empty window."""
    f = np.asarray(freq_hz)
    period = (len(f) - 1) / (f[-1] - f[0])
    far = 2 * 450 / sd.C_LIGHT
    return {
        "zero": delayed_lines(f, [0.0], [1.0]),
        "early": delayed_lines(f, [22.9e-9], [1.0]),
        "far450m": delayed_lines(f, [far], [1.0]),
        "far_plus_period": delayed_lines(f, [far + period], [1.0]),
        "coherent_pair": delayed_lines(f, [22.9e-9, far], [1.0, 0.3j]),
        "negative_candidate": delayed_lines(f, [-0.2e-6], [1.0]),
    }


def influence(freq_hz, basis, train_mask, ridge_fraction, mask, taper):
    """Actual residual operator from measured channels to signed delays."""
    f = np.asarray(freq_hz)
    fit = obs.fit_basis(
        np.zeros(len(f), complex),
        basis,
        train_mask,
        ridge_fraction,
        return_operator=True,
    )
    wt = np.where(mask, sd.make_taper(taper, len(f)), 0.0)
    tau = sd.native_delays(len(f), (f[-1] - f[0]) / (len(f) - 1))
    selected = (tau > -1.3e-6) & (tau < 0)
    tau = tau[selected]
    transform = np.exp(2j * np.pi * tau[:, None] * (f - f[0]))
    transform *= wt[None] / wt.sum()
    measured = np.diag(np.asarray(mask, float))
    residual_operator = transform @ (measured - fit.operator @ measured)
    return dict(
        tau_s=tau,
        measured_mask=np.asarray(mask, bool),
        train_mask=fit.train_mask,
        fit_rank=fit.rank,
        residual_operator=residual_operator,
        singular_values=np.linalg.svd(residual_operator, compute_uv=False),
    )


def covariance_through(operator, covariance):
    """Propagate a declared complex covariance; do not infer its sigma."""
    a = np.asarray(operator)
    return a @ covariance @ a.conj().T


def ensemble_power(coefficients):
    """Mean power and Gaussian-speckle MC error, not a deterministic field."""
    x = np.asarray(coefficients)
    power = np.abs(x) ** 2
    return dict(
        mean=power.mean(axis=0),
        mc_se=power.std(axis=0, ddof=1) / np.sqrt(len(x))
        if len(x) > 1
        else np.full(x.shape[1:], np.nan),
        n_members=len(x),
    )
