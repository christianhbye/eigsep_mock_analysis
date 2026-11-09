"""Checkpointed native-time columns with unchanged pinned eigsim physics.

The private adapter uses eigsim's existing setup/orientation/convolution
operators once per frequency chunk. It never concatenates model tensors
for all orientations. Source fingerprints of these internal APIs are a
mandatory run preflight; changing them requires a new conformance review.
"""

import importlib
import time
from pathlib import Path

import numpy as np

from .assets import atomic_npz, write_json


def orientation_groups(elevation, azimuth, rotations, misalignment):
    """Group exact provider drive coordinates after checking their rotations.

    Stored rounded drive values are authoritative. Recovering angles from
    matrices and then uniquing their floating point roundoff would invent
    hundreds of orientations with the same physical pointing.
    """
    el, az, rot, mis = map(np.asarray, (elevation, azimuth, rotations, misalignment))
    n = len(el)
    if (
        el.shape != (n,)
        or az.shape != (n,)
        or rot.shape != (n, 3, 3)
        or mis.shape != (3, 3)
        or n == 0
        or not all(np.isfinite(a).all() for a in (el, az, rot, mis))
    ):
        raise ValueError("complete exact native drive/rotation coordinates required")
    er, ar = np.radians(el), np.radians(az)
    ce, se, ca, sa = np.cos(er), np.sin(er), np.cos(ar), np.sin(ar)
    zero = np.zeros(n)
    drive = np.stack(
        [
            np.stack([ca, -sa, zero], -1),
            np.stack([ce * sa, ce * ca, -se], -1),
            np.stack([se * sa, se * ca, ce], -1),
        ],
        -2,
    )
    if (
        not np.allclose(mis @ mis.T, np.eye(3), rtol=0, atol=1e-10)
        or not np.isclose(np.linalg.det(mis), 1, rtol=0, atol=1e-10)
        or not np.allclose(mis @ drive, rot, rtol=0, atol=1e-10)
    ):
        raise ValueError("stored native drive coordinates differ from proper rotations")
    values, inverse = np.unique(np.column_stack([el, az]), axis=0, return_inverse=True)
    return [
        (float(e), float(a), np.flatnonzero(inverse == i))
        for i, (e, a) in enumerate(values)
    ]


def reduce_columns(values, accepted, mapping, n_bins):
    """Count-weighted exact measured support, retaining empty cells as NaN."""
    values, accepted, mapping = map(np.asarray, (values, accepted, mapping))
    if (
        values.ndim != 3
        or accepted.dtype.kind != "b"
        or values.shape[:2] != accepted.shape
        or mapping.shape != (len(values),)
        or mapping.dtype.kind not in "iu"
        or not isinstance(n_bins, int)
        or n_bins <= 0
        or np.any((mapping < 0) | (mapping >= n_bins))
        or np.any(accepted[..., None] & ~np.isfinite(values))
    ):
        raise ValueError("measured native model axes/support differ")
    total = np.zeros((n_bins, *values.shape[1:]))
    count = np.zeros((n_bins, values.shape[1]), np.int64)
    np.add.at(total, mapping, np.where(accepted[..., None], values, 0))
    np.add.at(count, mapping, accepted.astype(np.int64))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count[..., None]
    return mean, count


def simulate_native(
    beam,
    frequencies,
    sky,
    horizon,
    times,
    elevation,
    azimuth,
    rotations,
    misalignment,
    site,
    out,
    chunk_rows=128,
    sky_alm=None,
    closure=True,
):
    """One global frame epoch, sequential orientations, bounded convolution.

    Checkpoints retain original row positions and physical S/fg columns.
    The first chunk is compared to unchanged eigsim._run_orientation using
    the SAME setup/frame. No JIT, normalization or resolution overrides.
    """
    if not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise ValueError("positive integer convolution chunk required")
    times, f, bm, horizon = map(np.asarray, (times, frequencies, beam, horizon))
    if (
        times.ndim != 1
        or not np.isfinite(times).all()
        or f.ndim != 1
        or bm.shape != (len(f), 130, 258)
        or horizon.shape != (130, 258)
        or not np.isfinite(bm).all()
        or not np.isfinite(horizon).all()
        or set(site) != {"lat", "lon", "alt"}
    ):
        raise ValueError("native model time/frequency/asset/site axes differ")
    groups = orientation_groups(elevation, azimuth, rotations, misalignment)
    if sum(len(k) for _, _, k in groups) != len(times):
        raise ValueError("native drive and time row counts differ")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    sim = importlib.import_module("eigsim.simulate")
    start = time.monotonic()
    setup = sim._setup(
        bm,
        f,
        sky,
        times / 86400 + 2440587.5,
        None,
        "mwss",
        {"horizon": horizon},
        sky_alm,
        {**site, "Tgnd": 0.0},
        fuse_orientation=True,
    )
    return simulate_setup(
        setup,
        f,
        times,
        elevation,
        azimuth,
        rotations,
        misalignment,
        out,
        chunk_rows=chunk_rows,
        closure=closure,
        prepared_output=True,
        started=start,
    )


def simulate_setup(
    setup,
    frequencies,
    times,
    elevation,
    azimuth,
    rotations,
    misalignment,
    out,
    chunk_rows=128,
    closure=True,
    prepared_output=False,
    started=None,
):
    """Existing native convolution/checkpoints with an explicit fixed setup.

    Callers reusing a frame must independently validate actual beam/sky
    replacement and normalization; this helper performs no such inference.
    Actual physical frequency labels remain in every checkpoint.
    """
    if not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise ValueError("positive integer convolution chunk required")
    times, f = np.asarray(times), np.asarray(frequencies)
    if (
        times.ndim != 1
        or f.ndim != 1
        or not len(f)
        or not np.isfinite(times).all()
        or not np.isfinite(f).all()
        or np.shape(setup.phases)[0] != len(times)
        or np.shape(setup.alm)[0] != len(f)
        or np.shape(setup.sky_alm)[0] != len(f)
        or np.shape(setup.beam_norm) != (len(f),)
    ):
        raise ValueError("shared setup native time/frequency axes differ")
    groups = orientation_groups(elevation, azimuth, rotations, misalignment)
    out = Path(out)
    if prepared_output:
        if not out.is_dir() or any(out.iterdir()):
            raise ValueError("prepared output must be a fresh empty directory")
    else:
        out.mkdir(parents=True, exist_ok=False)
    import jax.numpy as jnp
    from croissant.rotations import rotmat_to_eulerZYZ
    from croissant.simulator import convolve
    from eigsim.rotations import drive_rotation_matrix

    sim = importlib.import_module("eigsim.simulate")
    start = time.monotonic() if started is None else started
    values = np.full((len(times), len(f), 2), np.nan)
    completed, first_closure = [], None
    for group, (el, az, rows) in enumerate(groups):
        matrix = drive_rotation_matrix(el, az, misalignment=misalignment)
        euler = jnp.asarray(rotmat_to_eulerZYZ(matrix), dtype=jnp.float64)
        beam_eq, fg = setup.orient(
            setup.alm, euler, setup.sim, setup.horizon, setup.quad_weights
        )
        for offset in range(0, len(rows), chunk_rows):
            k = rows[offset : offset + chunk_rows]
            phases = setup.phases[k]
            s = np.asarray(
                (
                    convolve(beam_eq, setup.sky_alm, phases) / setup.beam_norm[None, :]
                ).real
            )
            ground = np.broadcast_to(np.asarray(fg), s.shape)
            if (
                s.shape != (len(k), len(f))
                or not np.isfinite(s).all()
                or not np.isfinite(ground).all()
            ):
                raise ValueError("nonfinite or wrong-axis native model")
            if first_closure is None and closure:
                reference = np.asarray(
                    sim._run_orientation(
                        setup, el, az, phases, misalignment=misalignment
                    )
                )
                first_closure = float(np.max(abs(s - reference)))
                np.testing.assert_allclose(s, reference, rtol=1e-12, atol=1e-9)
            value = np.stack([s, ground], axis=-1)
            values[k] = value
            path = out / f"orientation_{group:04}_chunk_{offset:06}.npz"
            atomic_npz(
                path,
                native_position=k,
                times_unix=times[k],
                frequencies_mhz=f,
                columns=value,
                elevation_deg=el,
                azimuth_deg=az,
            )
            completed.append({"file": path.name, "rows": len(k), "orientation": group})
            write_json(
                out / "PROGRESS.json",
                {
                    "completed": completed,
                    "orientations_total": len(groups),
                    "elapsed_s": time.monotonic() - start,
                    "first_unchanged_operator_max_abs_K": first_closure,
                },
            )
    if not np.isfinite(values).all():
        raise ValueError("native model row coverage incomplete")
    return values, {
        "orientations": len(groups),
        "rows": len(times),
        "frequencies_mhz": f.tolist(),
        "elapsed_s": time.monotonic() - start,
        "first_unchanged_operator_max_abs_K": first_closure,
    }
