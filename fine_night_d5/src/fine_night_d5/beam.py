"""Common MOD24 beam symmetry/normalization/truncation/frequency operator.

The symmetry is 055's complex Cartesian body-x operation. The native MWSS
quadrature and scalar SHT are v002's recipe, not 055's cell-area weights.
Raw fields are separate controls, with their own explicit artifact identity.
"""

import numpy as np
from scipy.interpolate import CubicSpline


def native_indices(count, requested=None):
    """Explicit ordered native-slice control selection; full grid by default."""
    if not isinstance(count, int) or count <= 0:
        raise ValueError("positive integer native frequency count required")
    if requested is None:
        return np.arange(count)
    chosen = np.asarray(requested)
    if (
        chosen.ndim != 1
        or not len(chosen)
        or chosen.dtype.kind not in "iu"
        or np.any(chosen < 0)
        or np.any(chosen >= count)
        or np.any(np.diff(chosen.astype(np.int64)) <= 0)
    ):
        raise ValueError("ordered unique in-range integer native indices required")
    return chosen


def body_x_symmetry(field):
    """Apply [E(n)-P_x E(P_x n)]/2 before squaring, x axis only."""
    e = np.asarray(field)
    if e.ndim not in (3, 4) or e.shape[-3] != 3:
        raise ValueError("complex Cartesian (...,3,theta,phi) field required")
    nt, nphi = e.shape[-2:]
    if nphi != 2 * (nt - 1) or nphi % 2 or not np.isfinite(e).all():
        raise ValueError("finite native MWSS field required")
    if e.dtype.kind != "c":
        raise ValueError("preserve complex fields through symmetry")
    mirror = (nphi // 2 - np.arange(nphi)) % nphi
    parity = np.array([1, -1, -1])[:, None, None]
    return 0.5 * (e + parity * e[..., mirror])


def normalize_native(field, theta_weights, symmetry):
    """Normalize scalar power to integral4pi on the exact native MWSS grid."""
    if symmetry not in ("body-x", "raw"):
        raise ValueError("explicit body-x or raw control required")
    e = np.asarray(field)
    if e.ndim != 3 or e.shape[-1] != 2 * (e.shape[-2] - 1):
        raise ValueError("single-frequency native MWSS field required")
    if symmetry == "body-x":
        e = body_x_symmetry(e)
    elif (
        e.ndim != 3
        or e.shape[0] != 3
        or e.dtype.kind != "c"
        or not np.isfinite(e).all()
    ):
        raise ValueError("finite complex single-frequency Cartesian field required")
    if e.ndim != 3:
        raise ValueError("normalize one frequency at a time")
    weights = np.asarray(theta_weights)
    if weights.shape != (e.shape[1],) or not np.isfinite(weights).all():
        raise ValueError("native theta quadrature axis differs")
    if not np.isclose(weights.sum() * e.shape[-1], 4 * np.pi, rtol=1e-12, atol=1e-12):
        raise ValueError("weights do not integrate the full MWSS sphere")
    power = np.sum(np.abs(e) ** 2, axis=0)
    integral = float(np.einsum("tp,t->", power, weights))
    if not np.isfinite(integral) or integral <= 0:
        raise ValueError("zero/nonfinite projected field power")
    return power * (4 * np.pi / integral), integral


def scalar_truncate(power, L_in, lmax_out=128):
    """Exact v002 MWSS scalar-power transform; lazy heavy runtime import.

    A real invocation is a separately gated science job. No JAX flag/JIT
    override or alternate transform is selected here. This is the same
    forward/crop/inverse operator as eigsim/scripts/make_bowtie_v002.py.
    """
    if lmax_out + 1 > L_in or lmax_out < 0:
        raise ValueError("invalid scalar truncation")
    if np.shape(power) != (L_in + 1, 2 * L_in):
        raise ValueError("power grid is not native MWSS")
    import jax
    import jax.numpy as jnp
    import s2fft

    if not jax.config.x64_enabled:
        raise ValueError("run requires existing JAX x64 runtime; no silent override")
    alm = np.asarray(
        s2fft.forward(
            jnp.asarray(power),
            L=L_in,
            spin=0,
            sampling="mwss",
            method="jax",
            reality=True,
        )
    )
    L_out = lmax_out + 1
    centre = L_in - 1
    kept = alm[:L_out, centre - (L_out - 1) : centre + L_out]
    total = np.sum(np.abs(alm) ** 2)
    dropped = float(1 - np.sum(np.abs(kept) ** 2) / total)
    result = s2fft.inverse(
        jnp.asarray(kept),
        L=L_out,
        spin=0,
        sampling="mwss",
        method="jax",
        reality=True,
    )
    return np.asarray(result).real, dropped


def frequency_operator(native_mhz, maps, requested_mhz):
    """Direct not-a-knot scalar-power spline, no extrapolation or clipping."""
    f, target, bm = (
        np.asarray(native_mhz),
        np.asarray(requested_mhz),
        np.asarray(maps),
    )
    if (
        f.ndim != 1
        or len(f) < 4
        or not np.isfinite(f).all()
        or np.any(np.diff(f) <= 0)
        or bm.shape[0] != len(f)
        or not np.isfinite(bm).all()
    ):
        raise ValueError("finite strictly increasing native frequency axis required")
    if (
        target.ndim != 1
        or not np.isfinite(target).all()
        or np.any(target < f[0])
        or np.any(target > f[-1])
    ):
        raise ValueError("requested frequency outside native support")
    result = CubicSpline(f, bm, axis=0, bc_type="not-a-knot", extrapolate=False)(target)
    return result


def interpolation_qc(native_mhz, maps, requested_mhz, theta_weights):
    """Report withheld-slice error/negative power; never tune or clip it."""
    f, bm = np.asarray(native_mhz), np.asarray(maps)
    result = frequency_operator(f, bm, requested_mhz)
    weights = np.asarray(theta_weights)
    if weights.shape != bm.shape[1:2]:
        raise ValueError("output quadrature axis differs")
    holdout = np.arange(1, len(f) - 1, 2)
    if len(f) - len(holdout) < 4:
        raise ValueError("need four retained native slices for withheld QC")
    keep = np.ones(len(f), bool)
    keep[holdout] = False
    predicted = frequency_operator(f[keep], bm[keep], f[holdout])
    error = predicted - bm[holdout]
    qc = {
        "withheld_indices": holdout,
        "withheld_max_abs_power_error": np.max(abs(error), axis=(1, 2)),
        "withheld_rms_power_error": np.sqrt(np.mean(error**2, axis=(1, 2))),
        "requested_integral_over_4pi": np.einsum("ftp,t->f", result, weights)
        / (4 * np.pi),
        "requested_minimum_power": np.min(result, axis=(1, 2)),
        "requested_negative_pixel_count": (result < 0).sum(axis=(1, 2)),
    }
    return result, qc
