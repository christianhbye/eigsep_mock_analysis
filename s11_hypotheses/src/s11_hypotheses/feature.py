"""Source-normalized sampled candidates and upstream injections."""

import numpy as np
from eigsep_analysis import s11_hypotheses as obs


def width_template(freq_hz, source_values, width_hz, shift_hz=0.0):
    """Add half-open top hats, normalized to reproduce BOTH source samples.

    The returned source sampling matrix and coefficients define overlapping
    excursions explicitly. A zero D5 norm is UNOBSERVED, not absence.
    """
    source = np.array([46.75e6, 47e6]) + shift_hz
    f = np.asarray(freq_hz)

    def sample(at):
        return (
            (at[:, None] >= source - width_hz / 2)
            & (at[:, None] < source + width_hz / 2)
        ).astype(float)

    matrix = sample(source)
    if np.linalg.matrix_rank(matrix) != 2:
        raise ValueError("source-sampling matrix does not identify widths")
    coeff = np.linalg.solve(matrix, np.asarray(source_values, complex))
    template = sample(f) @ coeff
    return dict(
        template=template,
        source_operator=matrix,
        coefficients=coeff,
        support=np.any(sample(f) > 0, axis=1),
        norm2=float(np.vdot(template, template).real),
        status="sampled" if np.any(template != 0) else "UNOBSERVED",
    )


def candidates(freq_hz, source_values, widths_hz, shift_hz=0.0, baseline=None):
    """Translated family; all controls repeat fitting with identical freedom."""
    f = np.asarray(freq_hz)
    out = {}
    for width in widths_hz:
        t = width_template(f, source_values, width, shift_hz)["template"]
        out[f"width_{width:g}_additive"] = t[:, None]
        if baseline is not None:
            out[f"width_{width:g}_multiplicative"] = (
                np.asarray(baseline)[:, :, None] * t[None, :, None]
            )
    out["step_additive"] = (f >= 47e6 + shift_hz)[:, None].astype(complex)
    pair = np.zeros((len(f), 2), complex)
    for j, center in enumerate([46861861.862, 47111111.111]):
        i = int(np.argmin(abs(f - center - shift_hz)))
        pair[i, j] = 1
    out["pair_additive"] = pair
    # Equal-dimensional translated pair/step candidates are part of the family.
    return out


def source_template(freq_hz, residual, source_ids):
    """Estimate from named source captures only; retain leave-one-source-out.

    Uses absolute complex values, never divides by a matched-load reflection.
    A separate multiplicative candidate is evaluated through its baseline.
    """
    f = np.asarray(freq_hz)
    idx = []
    for center in [46.75e6, 47e6]:
        hit = np.flatnonzero(abs(f - center) < 0.01)
        if len(hit) != 1:
            raise ValueError("MISTIC exact feature coordinate unavailable")
        idx.append(int(hit[0]))
    r = np.asarray(residual, complex)[source_ids][:, idx]
    if len(r) < 2 or not np.isfinite(r).all():
        raise ValueError("insufficient source-only residual captures")
    mean = r.mean(axis=0)
    loo = (r.sum(axis=0)[None] - r) / (len(r) - 1)
    return dict(
        values_re_im=np.column_stack([mean.real, mean.imag]).tolist(),
        leave_one_source_out_re_im=np.stack([loo.real, loo.imag], axis=-1).tolist(),
        source_indices=list(map(int, source_ids)),
        source_residual_re_im=np.stack([r.real, r.imag], axis=-1).tolist(),
        source_coordinates_hz=f[idx].tolist(),
        uncertainty_scope="source-capture repeatability; not absolute VNA sigma",
    )


def inject_and_recover(
    freq_hz, values, selection, window, template, amplitude, phase_rad
):
    """Inject before fitting; measure the actual baseline absorption."""
    signal = amplitude * np.exp(1j * phase_rad) * np.asarray(template)
    before = obs.baseline(values, freq_hz, selection, window)
    after = obs.baseline(np.asarray(values) + signal, freq_hz, selection, window)
    recovered = signal - (after.prediction - before.prediction)
    target = obs.interval_mask(freq_hz, window)
    result = obs.recovery_ratio(signal[target], recovered[target])
    result.update(
        amplitude=float(amplitude),
        phase_rad=float(phase_rad),
        measured_samples=int(np.count_nonzero(signal[target])),
        original_measured_mask=np.isfinite(values),
        injected=signal,
        recovered=recovered,
    )
    return result
