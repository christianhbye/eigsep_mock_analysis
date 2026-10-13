"""Data loading utilities for default EIGSEP data files."""

from pathlib import Path

import numpy as np

from .config import load_config

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def load_beam(path=None, config=None):
    """Load an EIGSEP beam in MWSS sampling.

    Parameters
    ----------
    path : str, Path, or None
        Path to the beam .npz file.  If None, loads the file named by
        the config's ``beam.file`` from the package data directory.
    config : str, Path, or None
        Config for :func:`~eigsim.config.load_config`, read only when
        *path* is None.  ``None`` uses the default config (beam
        v002).  Pin ``"eigsep_v001"`` or ``"eigsep_v000"`` to reproduce
        studies made with those beams.

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
        path = _DATA_DIR / load_config(config)["beam"]["file"]
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


#: Height eras of the team's D5 horizon and their keys in the built file.
TEAM_HORIZON_ERAS = {
    "30m": "alpha_h_30m",
    "87.5m": "alpha_h_87p5m",
    "91m": "alpha_h_91m",
}


def load_horizon_curve(era="91m", path=None):
    """Load the team's D5 horizon for one height era, ready for
    :func:`~eigsim.horizon.open_sky_weight`.

    The file is built by ``scripts/make_horizon_d5_v0003.py`` from the
    team's ``curation/horizon_profiles_v0003`` (antenna at geometry
    release v0004, DEM v0003), converted from grid bearings to true
    compass azimuth.

    Parameters
    ----------
    era : {"30m", "87.5m", "91m"}
        Height era.  ``"91m"`` is the final height.
    path : str, Path, or None
        The built ``.npz``; default ``data/horizon_d5_v0003.npz``.

    Returns
    -------
    az_grid : np.ndarray
        Compass azimuth ``atan2(E, N)`` [rad], ascending on ``[0, 2 pi)``.
    alpha_h : np.ndarray
        Horizon elevation [rad] at each azimuth.
    info : dict
        ``era``, ``stamp`` (the team product's version stamp),
        ``source_sha256`` and ``convergence_deg``.

    """
    if era not in TEAM_HORIZON_ERAS:
        raise ValueError(f"era must be one of {sorted(TEAM_HORIZON_ERAS)}, got {era!r}")
    if path is None:
        path = _DATA_DIR / "horizon_d5_v0003.npz"
    d = np.load(path)
    info = {
        "era": era,
        "stamp": str(d["stamp"]),
        "source_sha256": str(d["source_sha256"]),
        "convergence_deg": float(d["convergence_deg"]),
    }
    return d["az"], d[TEAM_HORIZON_ERAS[era]], info
