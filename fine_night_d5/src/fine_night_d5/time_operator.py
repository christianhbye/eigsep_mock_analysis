"""Exact measured native-row averaging around the existing sky evaluator."""

import numpy as np


def join_pointing(native, table):
    """Join native rotations by exact file,row,input, rejecting missing rows.

    A producer of the table must record the existing v5/mount/IMU recipe,
    source hashes, and any fill identities. This function does no pointing
    interpolation or invented nearest-parent replacement.
    """

    def keys(source):
        n = len(source["row"])
        if any(np.shape(source[k]) != (n,) for k in ("source_file", "input")):
            raise ValueError("pointing identity axes differ")
        return list(
            zip(
                np.asarray(source["source_file"]).tolist(),
                np.asarray(source["row"]).tolist(),
                np.asarray(source["input"]).tolist(),
            )
        )

    nk, tk = keys(native), keys(table)
    if len(set(nk)) != len(nk) or len(set(tk)) != len(tk):
        raise ValueError("duplicate native/pointing identities")
    lookup = {key: i for i, key in enumerate(tk)}
    try:
        indices = np.array([lookup[key] for key in nk])
    except KeyError as exc:
        raise ValueError("native pointing row absent") from exc
    rot = np.asarray(table["rotation"])[indices]
    if rot.shape != (len(nk), 3, 3) or not np.isfinite(rot).all():
        raise ValueError("invalid native rotation matrices")
    if not np.allclose(
        rot @ np.swapaxes(rot, -1, -2), np.eye(3), rtol=0, atol=1e-10
    ) or not np.allclose(np.linalg.det(rot), 1, rtol=0, atol=1e-10):
        raise ValueError("pointing matrices must be proper rotations")
    if not np.array_equal(
        np.asarray(table["time_best"])[indices],
        np.asarray(native["time_best"]),
    ):
        raise ValueError("pointing/native time_best differs")
    return indices, rot


def average_native(
    times,
    frequencies_mhz,
    rotations,
    accepted,
    mapping,
    n_bins,
    evaluate,
    chunk_rows=128,
):
    """Average actual model columns with the same binary native data mask.

    ``evaluate(times,frequencies_mhz,rotations)`` is the pinned existing
    eigsim sky/horizon/loss column evaluator, returning(row,channel,term).
    Batches do not change the sampling or add independent observations.
    Counts, not exposure, are the weights of this count-mean estimator.
    """
    t, f = np.asarray(times), np.asarray(frequencies_mhz)
    rot, use, index = (
        np.asarray(rotations),
        np.asarray(accepted),
        np.asarray(mapping),
    )
    n = len(t)
    if (
        t.shape != (n,)
        or f.ndim != 1
        or not np.isfinite(t).all()
        or not np.isfinite(f).all()
        or rot.shape != (n, 3, 3)
        or not np.isfinite(rot).all()
        or use.shape != (n, len(f))
        or use.dtype.kind != "b"
        or index.shape != (n,)
        or index.dtype.kind not in "iu"
        or n_bins <= 0
        or np.any((index < -1) | (index >= n_bins))
    ):
        raise ValueError("native model axes/coordinates differ")
    if not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise ValueError("positive integer chunk_rows required")
    count = np.zeros((n_bins, len(f)), np.int64)
    accum = None
    assigned = np.flatnonzero(index >= 0)
    for start in range(0, len(assigned), chunk_rows):
        k = assigned[start : start + chunk_rows]
        value = np.asarray(evaluate(t[k], f, rot[k]))
        if value.ndim != 3 or value.shape[:2] != (len(k), len(f)):
            raise ValueError("model evaluator must return row,channel,term")
        if np.any(use[k, :, None] & ~np.isfinite(value)):
            raise ValueError("missing model on a measured native cell")
        if accum is None:
            accum = np.zeros((n_bins, len(f), value.shape[-1]))
        elif accum.shape[-1] != value.shape[-1]:
            raise ValueError("model term axis changed between chunks")
        np.add.at(accum, index[k], np.where(use[k, :, None], value, 0))
        np.add.at(count, index[k], use[k].astype(np.int64))
    if accum is None:
        raise ValueError("no assigned native rows to evaluate")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = accum / count[:, :, None]
    return mean, count
