"""Data loading utilities for default EIGSEP data files."""

from pathlib import Path

import numpy as np

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def load_beam(path=None):
    """Load the default EIGSEP beam data in MWSS sampling.

    Parameters
    ----------
    path : str, Path, or None
        Path to the beam .npz file.  If None, loads the default
        MWSS beam from the package data directory.

    Returns
    -------
    freqs : np.ndarray
        Frequencies in Hz, shape ``(N_freqs,)``.
    beam_data : np.ndarray
        Beam power pattern in MWSS sampling,
        shape ``(N_freqs, N_theta, N_phi)``.
    lmax : int
        Maximum spherical harmonic degree.

    """
    if path is None:
        path = _DATA_DIR / "eigsep_bowtie_v000_mwss.npz"
    d = np.load(path)
    return d["freqs"], d["bm"], int(d["lmax"])


def load_horizon(path=None):
    """Load the default EIGSEP horizon model in MWSS sampling.

    Parameters
    ----------
    path : str, Path, or None
        Path to the horizon .npz file.  If None, loads the default
        MWSS horizon from the package data directory.

    Returns
    -------
    horizon : np.ndarray
        Distance to nearest horizon in MWSS sampling,
        shape ``(N_theta, N_phi)``.  NaN indicates open sky.
    lmax : int
        Maximum spherical harmonic degree.

    """
    if path is None:
        path = _DATA_DIR / "horizon_mwss.npz"
    d = np.load(path)
    return d["horizon"], int(d["lmax"])
