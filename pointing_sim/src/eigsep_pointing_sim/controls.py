"""Small declared perturbations for detector power on observed geometry."""

import numpy as np


def inject_motion_loss(
    y, residual, mask, direction, *, start_deg, loss_deg, width_deg=0
):
    """Inject loss against motion on one leg, retaining observed residuals.

    A zero width is a discrete loss; positive width spreads it uniformly
    over drive advance. No noise, sampling, or hardware state is inferred.
    """
    y, residual, mask = (
        np.asarray(y, float),
        np.asarray(residual, float),
        np.asarray(mask, bool),
    )
    if y.ndim != 1 or residual.shape != y.shape or mask.shape != y.shape:
        raise ValueError("aligned one-dimensional loss inputs required")
    if (
        direction not in (-1, 1)
        or not np.isfinite([start_deg, loss_deg, width_deg]).all()
        or loss_deg < 0
        or width_deg < 0
    ):
        raise ValueError("signed direction and nonnegative loss/width required")
    advance = direction * (y - start_deg)
    fraction = (
        (advance >= 0).astype(float)
        if width_deg == 0
        else np.clip(advance / width_deg, 0, 1)
    )
    result = residual.copy()
    result[mask] -= direction * loss_deg * fraction[mask]
    return result


def inject_pot_angle_shift(voltage, shift_deg, *, scale_v_per_deg):
    """Perturb supplied voltage by a declared angle shift, preserving noise."""
    voltage, shift = np.broadcast_arrays(
        np.asarray(voltage, float), np.asarray(shift_deg, float)
    )
    if not np.isfinite(scale_v_per_deg) or scale_v_per_deg == 0:
        raise ValueError("finite nonzero voltage scale required")
    return voltage + scale_v_per_deg * shift


def simulate_pot_voltage(
    az_deg, walk_deg, *, scale_v_per_deg, offset_v, noise_sd_deg, rng
):
    """Declared affine voltage/walk benchmark; noise sd is in angle units."""
    az, walk = np.broadcast_arrays(
        np.asarray(az_deg, float), np.asarray(walk_deg, float)
    )
    if not np.isfinite([offset_v, noise_sd_deg]).all() or noise_sd_deg < 0:
        raise ValueError("finite offset and nonnegative angle noise required")
    noise = noise_sd_deg * rng.normal(size=az.shape)
    voltage = inject_pot_angle_shift(
        np.zeros(az.shape) + offset_v,
        az + walk + noise,
        scale_v_per_deg=scale_v_per_deg,
    )
    return dict(
        voltage=voltage, noise_deg=noise, walk_deg=walk.copy(), az_deg=az.copy()
    )
