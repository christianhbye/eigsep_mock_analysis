"""C57 immutable context shared by numerical conformance and batched models."""

import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path

import numpy as np

from . import assets
from .four_channel_model import checked_stage


def load(config):
    """Load exact original coordinates, horizon and frame authority once."""
    if set(os.sched_getaffinity(0)) != set(range(6)):
        raise ValueError("native model requires recorded sixCPU0–5 allocation")
    required = {
        "numpy",
        "scipy",
        "jax",
        "jaxlib",
        "s2fft",
        "croissant-sim",
        "healpy",
        "pygdsm",
        "astropy",
    }
    modules = {"eigsim.simulate", "eigsim.rotations", "croissant.simulator"}
    if not required.issubset(config["runtime_versions"]) or not modules.issubset(
        config["runtime_modules"]
    ):
        raise ValueError("complete pinned physical runtime required")
    for name, version in config["runtime_versions"].items():
        if metadata.version(name) != version:
            raise ValueError("physical runtime version changed: " + name)
    for name, digest in config["runtime_modules"].items():
        if assets.sha256(importlib.import_module(name).__file__) != digest:
            raise ValueError("physical source changed: " + name)
    snapshot = json.loads(assets.checked_file(config["source_snapshot"]).read_text())
    source = Path(snapshot["root"])
    if not Path(__file__).resolve().is_relative_to(source):
        raise ValueError("executed producer is outside pinned source snapshot")
    for name, digest in snapshot["files_sha256"].items():
        if assets.sha256(source / name) != digest:
            raise ValueError("actual producer snapshot changed")
    directory, _ = checked_stage(config["native_stage"])
    point_done = assets.checked_file(config["pointing_stage"])
    point_record = json.loads(point_done.read_text())
    if point_record["full_pivot_done_sha256"] != config["native_stage"]["sha256"]:
        raise ValueError("pointing/calibration binding changed")
    point_npz = point_done.parent / "pointing.npz"
    if assets.sha256(point_npz) != point_record["pointing_sha256"]:
        raise ValueError("native pointing array changed")
    with np.load(directory / "cells.npz", allow_pickle=False) as z:
        cells = {
            k: z[k]
            for k in ["source_file", "row", "input", "time_best", "accepted", "chans"]
        }
    with np.load(directory / "grid.npz", allow_pickle=False) as z:
        mapping = z["native_to_bin"]
        nb = len(z["fine_bin_id"])
    with np.load(point_npz, allow_pickle=False) as z:
        pointing = {
            k: z[k]
            for k in [
                "el_drive_deg",
                "az_drive_deg",
                "rotations",
                "misalignment",
                "time_best",
            ]
        }
    np.testing.assert_array_equal(cells["time_best"], pointing["time_best"])
    if not np.all(cells["input"] == "4") or np.any(mapping < 0):
        raise ValueError("complete input4native coordinate assignment required")
    nominal = json.loads(assets.checked_file(config["nominal034"]).read_text())
    if (
        config["horizon"]["sha256"] != nominal["input_sha256"]["horizon"]
        or config["iers"]["finals2000A.all"]["sha256"]
        != nominal["iers"]["finals2000A_sha256"]
        or config["iers"]["eopc04.1962-now"]["sha256"]
        != nominal["iers"]["eopc04_sha256"]
        or point_record["night_tau_deg"] != nominal["tau_imu"]["tau_deg"]
    ):
        raise ValueError("frozen034 horizon/IERS/night tilt changed")
    assets.install_iers(config["iers"])
    horizon, _ = assets.horizon_weight(config["horizon"])
    site = {
        key: nominal["site"][value]
        for key, value in [("lat", "lat_deg"), ("lon", "lon_deg"), ("alt", "height_m")]
    }
    provenance = json.loads(assets.checked_file(config["provenance034"]).read_text())
    identity = dict(
        native_stage=config["native_stage"]["sha256"],
        pointing_stage=config["pointing_stage"]["sha256"],
        horizon=config["horizon"]["sha256"],
        iers={key: value["sha256"] for key, value in config["iers"].items()},
        site=site,
        world="earth",
        beam_sampling="mwss",
        beam_lmax=128,
        sky_sampling="healpix",
        sky_nside=64,
        sky_niter=3,
        sky_coord="galactic",
        runtime_versions=config["runtime_versions"],
        runtime_modules=config["runtime_modules"],
    )
    return dict(
        cells=cells,
        mapping=mapping,
        n_bins=nb,
        pointing=pointing,
        horizon=horizon,
        site=site,
        identity=identity,
        cmb=float(provenance["params"]["model"]["t_cmb_k"]),
        source_commit=snapshot["commit"],
    )
