"""Verified D5 inputs; time/pointing conventions follow the reviewed S1 adapter."""

import json
from pathlib import Path

import numpy as np

from .cache import file_sha256

SITE = {"lat": 39.247699, "lon": -113.402660, "alt": 1800.0}
PILOT_FREQUENCIES_MHZ = (50.78125, 97.65625, 148.4375, 246.09375)


def verify_inputs(directory):
    directory = Path(directory)
    records = json.loads((directory / "INPUT_MANIFEST.json").read_text())
    for name, record in records.items():
        if Path(name).name != name:
            raise ValueError(f"input name must be a filename: {name}")
        path = directory / name
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or file_sha256(path) != record["sha256"]
        ):
            raise ValueError(f"input missing or hash mismatch: {name}")
    return records


def select_night_rows(binned, fit, n_times):
    for key in ("t", "era", "az", "el"):
        if not np.array_equal(binned[key], fit[key]):
            raise ValueError(f"fit selection coordinate mismatch: {key}")
    eligible = np.flatnonzero(fit["fitted"] & (binned["era"] == "~87.5m"))
    eligible = eligible[np.argsort(binned["t"][eligible], kind="stable")]
    if not 1 <= n_times <= len(eligible):
        raise ValueError("n_times must fit the available selected night rows")
    return eligible[np.rint(np.linspace(0, len(eligible) - 1, n_times)).astype(int)]


def _load(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def load_inputs(directory, n_times=8, frequencies_mhz=PILOT_FREQUENCIES_MHZ):
    directory = Path(directory)
    records = verify_inputs(directory)
    binned = _load(directory / "binned.npz")
    fit = _load(directory / "fit_selection.npz")
    beam = _load(directory / "beam.npz")
    sky = _load(directory / "gsm_maps.npz")
    horizon = _load(directory / "team_horizon.npz")
    geo = json.loads((directory / "fit_manifest.json").read_text())["geometry"]
    shared = json.loads((directory / "geometry_shared.json").read_text())
    nominal = np.asarray(shared["antenna_91m_era"]["position_enu_m"])
    if not np.array_equal(nominal, horizon["antenna_enu_m"]):
        raise ValueError("geometry release and horizon position disagree")
    rows = select_night_rows(binned, fit, n_times)
    available = np.asarray(binned["freqs_mhz"])
    wanted = available if frequencies_mhz is None else np.asarray(frequencies_mhz)
    indices = np.argmin(abs(available[:, None] - wanted[None, :]), axis=0)
    if (
        not np.allclose(available[indices], wanted, rtol=0, atol=1e-8)
        or np.any(wanted >= 250)
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("frequencies must be distinct supported native D5 frequencies")
    beam_freqs = np.asarray(beam["freqs"]) / 1e6
    beam_indices = np.argmin(abs(beam_freqs[:, None] - wanted[None, :]), axis=0)
    if not np.allclose(
        beam_freqs[beam_indices], wanted, rtol=0, atol=1e-8
    ) or not np.array_equal(sky["freqs_mhz"], available):
        raise ValueError("beam/sky frequency coordinates disagree")
    psi = np.radians(geo["psi_deg"])
    c, s = np.cos(psi), np.sin(psi)
    return {
        "input_hashes": {key: rec["sha256"] for key, rec in records.items()},
        "rows": rows,
        "times_unix": binned["t"][rows],
        "times_jd": binned["t"][rows] / 86400 + 2440587.5,
        "azimuths_deg": np.round(binned["az"][rows], 1) + geo["az_offset_deg"],
        "elevations_deg": np.round(binned["el"][rows], 1) + geo["el_offset_deg"],
        "misalignment": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]),
        "frequencies_mhz": wanted,
        "beam": beam["bm"][beam_indices],
        "lmax": int(beam["lmax"]),
        "sky_maps": sky["M64"][indices],
        "nominal_enu_m": nominal,
        "team_horizon": horizon,
        "offsets": geo,
        "sky_source": json.loads((directory / "gsm_source.json").read_text()),
        "scope": (
            "87.5m selected night coordinates; 30m/91m are synthetic height controls"
        ),
    }


def setup_iers(directory):
    from astropy.utils import iers

    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    iers.conf.iers_degraded_accuracy = "error"
    table = iers.IERS_A.open(str(Path(directory) / "finals2000A.all"))
    iers.earth_orientation_table.set(table)
    return {
        "auto_download": False,
        "auto_max_age": None,
        "iers_degraded_accuracy": "error",
        "table": "IERS_A",
    }
