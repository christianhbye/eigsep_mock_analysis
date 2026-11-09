"""C57 pinned native-frequency columns through unchanged eigsim physics.

This adapter reads the explicit C57 calibration/pointing schema. It calls
existing sky, horizon, frame and streaming operators without lower angular
resolution, a backend change or altered normalization.
"""

import argparse
import importlib
import importlib.metadata as metadata
import json
import resource
import time
from pathlib import Path

import numpy as np

from . import assets
from .streaming import reduce_columns, simulate_native


def checked_stage(pin):
    path = assets.checked_file(pin)
    record = json.loads(path.read_text())
    for name, digest in record["output_hashes"].items():
        if assets.sha256(path.parent / name) != digest:
            raise ValueError("C57 stage member changed")
    return path.parent, record


def run(config, out):
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    for name, version in config["runtime_versions"].items():
        if metadata.version(name) != version:
            raise ValueError("model runtime version changed: " + name)
    for name, digest in config["runtime_modules"].items():
        module = importlib.import_module(name)
        if assets.sha256(module.__file__) != digest:
            raise ValueError("model implementation changed: " + name)
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
    if not required.issubset(config["runtime_versions"]):
        raise ValueError("incomplete physical runtime pins")
    source = json.loads(assets.checked_file(config["source_snapshot"]).read_text())
    source_root = Path(source["root"])
    for name, digest in source["files_sha256"].items():
        if assets.sha256(source_root / name) != digest:
            raise ValueError("actual source snapshot changed")
    # The recorded commit comes from the verified snapshot, not a parent repo.
    directory, stage = checked_stage(config["native_stage"])
    point_path = assets.checked_file(config["pointing_stage"])
    point_record = json.loads(point_path.read_text())
    if point_record["full_pivot_done_sha256"] != config["native_stage"]["sha256"]:
        raise ValueError("pointing/calibration stage binding changed")
    point_npz = point_path.parent / "pointing.npz"
    if assets.sha256(point_npz) != point_record["pointing_sha256"]:
        raise ValueError("pointing array changed")
    channels = np.asarray(config["channels"])
    if (
        channels.ndim != 1
        or channels.dtype.kind not in "iu"
        or not len(channels)
        or np.any(np.diff(channels) <= 0)
    ):
        raise ValueError("exact increasing native data channels required")
    with np.load(directory / "cells.npz", allow_pickle=False) as z:
        axis = z["chans"]
        index = np.searchsorted(axis, channels)
        if np.any(index >= len(axis)) or not np.array_equal(axis[index], channels):
            raise ValueError("model channels absent from C57 data")
        native = {k: z[k] for k in ["source_file", "row", "input", "time_best"]}
        accepted = z["accepted"][:, index]
    with np.load(directory / "grid.npz", allow_pickle=False) as z:
        mapping = z["native_to_bin"]
        n_bins = len(z["fine_bin_id"])
    with np.load(point_npz, allow_pickle=False) as z:
        point = {
            k: z[k]
            for k in [
                "el_drive_deg",
                "az_drive_deg",
                "rotations",
                "misalignment",
                "time_best",
            ]
        }
    np.testing.assert_array_equal(native["time_best"], point["time_best"])
    mode = config["frequency_operator"]
    if mode == "native":
        frequency = channels * 250 / 1024
        expansion = np.arange(len(channels))
    elif mode == "four-channel-centres":
        members = channels.reshape(-1, 4)
        if np.any(members[:, 0] % 4) or not np.array_equal(
            members, members[:, :1] + np.arange(4)
        ):
            raise ValueError("complete aligned groups required for centre control")
        frequency = members.mean(1) * 250 / 1024
        expansion = np.repeat(np.arange(len(members)), 4)
    else:
        raise ValueError("explicit native or centre-control operator required")
    beam_pin = config["beam"]
    if beam_pin["role"] in ["raster_hybrid", "represented_hfss_control"]:
        beam_path = assets.checked_file(beam_pin)
        with np.load(beam_path, allow_pickle=False) as z:
            if str(z["role"]) != beam_pin["role"] or int(z["lmax"]) != 128:
                raise ValueError("explicit C57 beam role/grid required")
            f = z["freqs"] / 1e6
            idx = np.searchsorted(f, frequency)
            if np.any(idx >= len(f)) or not np.array_equal(f[idx], frequency):
                raise ValueError(
                    "C57 field must be evaluated at actual model frequencies"
                )
            bm = z["bm"][idx]
    else:
        bm, _, qc = assets.scalar_beam(beam_pin, frequency)
        assets.atomic_npz(out / "beam_interpolation_qc.npz", **qc)
    nominal = json.loads(assets.checked_file(config["nominal034"]).read_text())
    if (
        config["horizon"]["sha256"] != nominal["input_sha256"]["horizon"]
        or config["iers"]["finals2000A.all"]["sha256"]
        != nominal["iers"]["finals2000A_sha256"]
        or config["iers"]["eopc04.1962-now"]["sha256"]
        != nominal["iers"]["eopc04_sha256"]
    ):
        raise ValueError("034horizon or Earth-orientation identity changed")
    if point_record["night_tau_deg"] != nominal["tau_imu"]["tau_deg"]:
        raise ValueError("night tilt authority changed")
    assets.install_iers(config["iers"])
    maps, _ = assets.gsm_maps(config["gsm"], frequency, out / "gsm")
    import croissant as cro

    sky = cro.Sky(maps, frequency, sampling="healpix", coord="galactic", niter=3)
    horizon, _ = assets.horizon_weight(config["horizon"])
    site = {
        k: nominal["site"][v]
        for k, v in [("lat", "lat_deg"), ("lon", "lon_deg"), ("alt", "height_m")]
    }
    columns, qc = simulate_native(
        bm,
        frequency,
        sky,
        horizon,
        native["time_best"],
        point["el_drive_deg"],
        point["az_drive_deg"],
        point["rotations"],
        point["misalignment"],
        site,
        out / "native-checkpoints",
        chunk_rows=128,
    )
    columns = columns[:, expansion]
    means, counts = reduce_columns(columns, accepted, mapping, n_bins)
    provenance = json.loads(assets.checked_file(config["provenance034"]).read_text())
    cmb = float(provenance["params"]["model"]["t_cmb_k"])
    assets.atomic_npz(
        out / "columns.npz",
        channels=channels,
        native_A=columns[..., 0] + cmb * (1 - columns[..., 1]),
        native_S=columns[..., 0],
        native_fg=columns[..., 1],
        bin_A=means[..., 0] + cmb * (1 - means[..., 1]),
        bin_count=counts,
        native_index=np.arange(len(mapping)),
        fine_bin_id=mapping,
        accepted=accepted,
        model_evaluation_frequencies_mhz=frequency,
        data_to_evaluation_index=expansion,
        **native,
    )
    assets.write_json(
        out / "RUN_DONE.json",
        dict(
            label="diagnostic",
            source_commit=source["commit"],
            source_snapshot_sha256=config["source_snapshot"]["sha256"],
            input_config=config,
            model_qc=qc,
            columns_sha256=assets.sha256(out / "columns.npz"),
            elapsed_s=time.monotonic() - start,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    run(json.loads(a.inputs.read_text()), a.out)
