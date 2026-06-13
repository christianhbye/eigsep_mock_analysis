"""Antenna-temperature differences, ground-loss modes, summary stats.

All functions are pure NumPy. ``t_sys`` has shape ``(P, n_times,
n_freqs)`` and ``fgnd`` has shape ``(P, n_freqs)``; index 0 is the
nominal position.
"""

import numpy as np

MODES = ("uncorrected", "oracle", "miscorrected")


def glc(t_sys, fgnd, t_gnd, t_rcvr):
    """Ground-loss correction ``(t_sys - t_rcvr - fgnd*t_gnd)/(1 - fgnd)``.

    ``fgnd`` is broadcast over the time axis when it has one fewer
    dimension than ``t_sys``.
    """
    fgnd = np.asarray(fgnd)
    if fgnd.ndim == np.ndim(t_sys) - 1:
        fgnd = np.expand_dims(fgnd, axis=-2)  # insert time axis
    return (t_sys - t_rcvr - fgnd * t_gnd) / (1.0 - fgnd)


def delta_waterfall(t_sys, fgnd, mode, t_gnd, t_rcvr, nominal=0):
    """ΔT vs the nominal position for every position, for ``mode``.

    Returns an array shaped like ``t_sys`` (``(P, n_times, n_freqs)``).
    """
    if mode == "uncorrected":
        field = t_sys
    elif mode == "oracle":
        field = glc(t_sys, fgnd, t_gnd, t_rcvr)
    elif mode == "miscorrected":
        f0 = np.broadcast_to(fgnd[nominal], fgnd.shape)
        field = glc(t_sys, f0, t_gnd, t_rcvr)
    else:
        raise ValueError(f"unknown mode {mode!r}; choose one of {MODES}")
    return field - field[nominal][None]


def summary_stats(delta):
    """Per-position RMS and max|ΔT| over the (time, freq) plane."""
    return {
        "rms": np.sqrt(np.mean(delta**2, axis=(1, 2))),
        "max": np.max(np.abs(delta), axis=(1, 2)),
    }


def rms_over_time(delta):
    """RMS over LST -> spectrum per position, shape ``(P, n_freqs)``."""
    return np.sqrt(np.mean(delta**2, axis=1))


def rms_over_freq(delta):
    """RMS over frequency -> time series per position, shape ``(P, n_times)``."""
    return np.sqrt(np.mean(delta**2, axis=2))
