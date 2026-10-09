"""Idealized drive paths and exact integration moments for pointing tests.

These model assumptions do not reconstruct timestamps of sparse telemetry
updates. In particular, a saved arithmetic mean of updates is not guaranteed
to equal a continuous-time window mean. Estimator users must test/report that
precondition against their observations rather than smoothing it away.
"""

import numpy as np


def _moves(moves):
    table = np.asarray(moves, dtype=float)
    if table.size == 0:
        return np.empty((0, 3))
    if table.ndim != 2 or table.shape[1] != 3 or not np.isfinite(table).all():
        raise ValueError("moves must be finite (start_s, stop_s, rate_deg_s) rows")
    if np.any(table[:, 1] <= table[:, 0]) or np.any(table[:, 2] == 0):
        raise ValueError("each move needs positive duration and nonzero rate")
    if np.any(table[1:, 0] < table[:-1, 1]):
        raise ValueError("moves must be ordered and cannot overlap")
    return table


def uniform_drive(time, moves, *, y0=0.0):
    """Evaluate a continuous, constant-rate-per-move idealized drive.

    ``moves`` is a sequence of (start seconds, stop seconds, signed
    degrees/second), ordered without overlap. ``y0`` is the angle before
    the first move; the drive holds between moves. Times may be Unix or
    relative seconds. Angles are unwrapped degrees.
    """
    time = np.asarray(time, float)
    table = _moves(moves)
    if time.ndim != 1 or not np.isfinite(time).all() or not np.isfinite(y0):
        raise ValueError("finite one-dimensional times and initial angle required")
    angle = np.full(time.shape, y0, float)
    for start, stop, rate in table:
        angle += rate * np.clip(time - start, 0, stop - start)
    return angle


def uniform_window_means(edges, moves, *, y0=0.0):
    """Exact continuous-time means, including partial start/stop windows.

    The integral is evaluated over each linear overlap and each following
    plateau. Subtracting large squared primitives would be unstable at
    Unix epochs, so all time arithmetic is first made relative and the
    interval contributions are evaluated directly.
    """
    edges = np.asarray(edges, float)
    table = _moves(moves).copy()
    if (
        edges.ndim != 1
        or len(edges) < 2
        or not np.isfinite(edges).all()
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("finite strictly increasing window edges required")
    if not np.isfinite(y0):
        raise ValueError("finite initial angle required")
    origin = edges[0]
    edges = edges - origin
    table[:, :2] -= origin
    left, right = edges[:-1], edges[1:]
    width = right - left
    mean = np.full(left.shape, y0, float)
    for start, stop, rate in table:
        lo, hi = np.maximum(left, start), np.minimum(right, stop)
        moving_length = np.maximum(hi - lo, 0)
        # The average of a linear ramp on its overlap is its midpoint.
        ramp_area = moving_length * ((lo + hi) / 2 - start)
        plateau_length = np.maximum(right - np.maximum(left, stop), 0)
        mean += rate * (ramp_area + (stop - start) * plateau_length) / width
    return mean


def uniform_micro_means(edges, moves, *, y0=0.0, n_sub=64):
    """Return exact sub-window means, with midpoint representative times.

    These values are **microbin means**, not instantaneous midpoint samples.
    Their equal-weight per-row means reproduce continuous window integrals
    exactly even next to a start or stop. A fixed midpoint quadrature cannot
    satisfy a 1e-6-degree identity at arbitrary break phases with 32/64 bins.
    Nonlinear pointing effects applied to microbin means still need resolution
    checks; exact drive moments do not make those transformations exact.
    """
    edges = np.asarray(edges, float)
    # Validate before constructing a potentially large grid.
    uniform_window_means(edges, moves, y0=y0)
    if n_sub < 1 or int(n_sub) != n_sub:
        raise ValueError("positive integer n_sub required")
    n_sub = int(n_sub)
    origin = edges[0]
    edges = edges - origin
    table = _moves(moves).copy()
    table[:, :2] -= origin
    fraction = np.arange(n_sub) / n_sub
    micro_left = (edges[:-1, None] + np.diff(edges)[:, None] * fraction).ravel()
    micro_edges = np.r_[micro_left, edges[-1]]
    return dict(
        time=origin + (micro_edges[:-1] + micro_edges[1:]) / 2,
        time_relative=(micro_edges[:-1] + micro_edges[1:]) / 2,
        time_origin=float(origin),
        y_mean=uniform_window_means(micro_edges, table, y0=y0),
        edges=origin + micro_edges,
        edges_relative=micro_edges,
        n_sub=n_sub,
        representation="exact microbin means; midpoint representative times",
    )
