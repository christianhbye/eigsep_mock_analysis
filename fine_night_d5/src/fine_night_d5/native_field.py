"""Shape-only native harmonic-field export for the C57 matched models.

Angular harmonics and spectral continuation are explicit caller inputs,
using the fitted producer's convention. No field interpolation or new
beam-frequency smoothing is introduced by this export.
"""

import numpy as np


def transverse_power(fields, directions, solid_angle_weights):
    """Project Cartesian fields transverse and normalize spherical power.

    Returns normalized scalar power and the pre-normalization spherical
    mean. A transmitter gain must multiply by that mean to preserve its
    prediction when the corresponding field is divided by its square root.
    Radial subtraction preserves coupling to a transverse transmitter arm.
    """
    e = np.asarray(fields)
    r, w = np.asarray(directions, float), np.asarray(solid_angle_weights, float)
    if (
        e.ndim != 3
        or e.shape[1] != 3
        or r.shape != e.shape[1:]
        or w.shape != (e.shape[2],)
        or not all(np.isfinite(a).all() for a in [e, r, w])
        or np.any(w < 0)
        or not np.isclose(w.sum(), 4 * np.pi, rtol=1e-12, atol=1e-12)
        or not np.allclose(np.sum(r * r, axis=0), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("finite Cartesian fields/unit directions/4pi weights required")
    transverse = e - np.sum(e * r[None], axis=1)[:, None] * r[None]
    power = np.sum(abs(transverse) ** 2, axis=1)
    norm = power @ w / (4 * np.pi)
    if not np.isfinite(norm).all() or np.any(norm <= 0):
        raise ValueError("positive finite transverse power required")
    power /= norm[:, None]
    np.testing.assert_allclose(power @ w, 4 * np.pi, rtol=1e-12, atol=1e-12)
    return power, norm


def harmonic_power(coefficients, angular_basis, directions, weights):
    """Evaluate existing component-major harmonic coefficients then project.

    coefficients are already evaluated at the requested native frequencies.
    The caller retains actual coordinates and quadrature in its product.
    """
    c, y = np.asarray(coefficients), np.asarray(angular_basis)
    if c.ndim != 2 or y.ndim != 2 or c.shape[1] != 3 * y.shape[1]:
        raise ValueError("component-major harmonic coefficient layout required")
    fields = c.reshape(len(c), 3, y.shape[1]) @ y.T
    return transverse_power(fields, directions, weights)
