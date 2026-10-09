"""Declared duty-cycled drive sensitivity, not a firmware replay."""

import numpy as np

from .drive import _moves


def batch_moves(moves, *, period_s=0.179, on_s=0.072):
    """Expand each macro motion into active ramps and idle plateaus.

    Macro rate is the cycle-averaged speed. Each batch travels rate*period
    at active speed rate*period/on; the final batch is shortened to retain
    the declared total angle. The first batch begins at the macro onset.
    The final physical arrival can precede the macro's continuous arrival.
    This idealized 72/179-ms timing is a sensitivity convention; sparse
    D5 telemetry cannot establish the batch phase or its exact durations.
    """
    if not np.isfinite([period_s, on_s]).all() or not 0 < on_s <= period_s:
        raise ValueError("positive active duration no longer than period required")
    result = []
    for start, stop, rate in _moves(moves):
        total = abs(rate) * (stop - start)
        step = abs(rate) * period_s
        complete = int(np.floor(total / step))
        for batch in range(complete):
            onset = start + batch * period_s
            result.append((onset, onset + on_s, rate * period_s / on_s))
        remaining = total - complete * step
        if remaining > 1e-10:
            onset = start + complete * period_s
            result.append(
                (onset, onset + remaining / step * on_s, rate * period_s / on_s)
            )
    return np.asarray(result, float).reshape(-1, 3)
