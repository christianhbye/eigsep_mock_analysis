"""Explicit gated native-time model producer; no default products are changed.

python -m fine_night_d5.model --inputs PINNED_JSON --out NEW_RUN
The manifest selects exact channels and existing calibration/pointing
stages. Its three-channel cost probe is a diagnostic, never a full fit.
"""

import argparse
import importlib
import importlib.metadata as metadata
import json
import resource
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import assets
from .streaming import reduce_columns, simulate_native
from .time_operator import join_pointing


def run(config, out):
    """Load immutable inputs and checkpoint actual physical model columns."""
    if config["schema"] != 1 or config["label"] != "diagnostic":
        raise ValueError("explicit diagnostic native-model schema required")
    required_versions = {
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
    required_modules = {"eigsim.simulate", "eigsim.rotations", "croissant.simulator"}
    if not required_versions.issubset(
        config["runtime_versions"]
    ) or not required_modules.issubset(config["runtime_modules"]):
        raise ValueError("explicit historical model runtime/source pins required")
    for name, expected in config["runtime_versions"].items():
        if metadata.version(name) != expected:
            raise ValueError("model runtime version differs: " + name)
    for name, digest in config["runtime_modules"].items():
        module = importlib.import_module(name)
        if assets.sha256(module.__file__) != digest:
            raise ValueError("model implementation differs: " + name)
    native_path, native_record = assets.calibration_stage(config["native_stage"])
    point_path, point_record = assets.calibration_stage(config["pointing_stage"])
    if (
        point_record["metadata"]["native_stage_sha256"]
        != config["native_stage"]["sha256"]
        or point_record["metadata"]["input_manifest_sha256"]
        != native_record["metadata"]["input_manifest_sha256"]
    ):
        raise ValueError("pointing is not bound to this exact calibration stage")
    # Read the small assignment axis FIRST, then keep only assigned rows.
    # This historical pilot archive still contains unassigned rows; future
    # full-night materialization subsets by parent before archive assembly.
    with np.load(native_path, allow_pickle=False) as z:
        mapping_all = z["grid__native_to_bin"]
        rows = np.flatnonzero(mapping_all >= 0)
        all_channels = z["native__chans"]
        channels = np.asarray(config["channels"])
        if (
            channels.ndim != 1
            or channels.dtype.kind not in "iu"
            or len(channels) == 0
            or np.any(np.diff(channels) <= 0)
        ):
            raise ValueError("unique increasing exact channel IDs required")
        index = np.searchsorted(all_channels, channels)
        if np.any(index >= len(all_channels)) or not np.array_equal(
            all_channels[index], channels
        ):
            raise ValueError("native model channels absent")
        native = {
            name: z["native__" + name][rows]
            for name in ("source_file", "row", "input", "time_best")
        }
        accepted = z["native__accepted"][np.ix_(rows, index)]
        mapping = mapping_all[rows]
        n_bins = len(z["grid__fine_bin_id"])
        expected_count = z["bin__count"][:, index]
    with np.load(point_path, allow_pickle=False) as z:
        point = {
            name: z[name]
            for name in ("source_file", "row", "input", "time_best", "rotation")
        }
        point["native_index"] = z["native_index"]
        point["az_drive_deg"], point["el_drive_deg"] = (
            z["az_drive_deg"],
            z["el_drive_deg"],
        )
    joined, rotations = join_pointing(native, point)
    np.testing.assert_array_equal(point["native_index"][joined], rows)
    frequency = channels * 250 / 1024
    bm, beam_identity, beam_qc = assets.scalar_beam(config["beam"], frequency)
    assets.atomic_npz(out / "beam_interpolation_qc.npz", **beam_qc)
    nominal_path = assets.checked_file(config["nominal034"])
    nominal = json.loads(nominal_path.read_text())
    if (
        config["horizon"]["sha256"] != nominal["input_sha256"]["horizon"]
        or config["iers"]["finals2000A.all"]["sha256"]
        != nominal["iers"]["finals2000A_sha256"]
        or config["iers"]["eopc04.1962-now"]["sha256"]
        != nominal["iers"]["eopc04_sha256"]
    ):
        raise ValueError("original034 horizon/IERS identity changed")
    iers = assets.install_iers(config["iers"])
    maps, gsm_checks = assets.gsm_maps(config["gsm"], frequency, out / "gsm")
    import croissant as cro
    from eigsim.rotations import misalignment_matrix

    sky = cro.Sky(maps, frequency, sampling="healpix", coord="galactic", niter=3)
    horizon, horizon_identity = assets.horizon_weight(config["horizon"])
    pointing = point_record["metadata"]
    mis = misalignment_matrix(
        tilt_y_deg=pointing["night_tau_deg"], tilt_z_deg=pointing["mount"]["psi_deg"]
    )
    if (
        assets.sha256(nominal_path)
        != pointing["tilt_authority"]["consumed_scalar_source_sha256"]
        or nominal["tau_imu"]["tau_deg"] != pointing["night_tau_deg"]
    ):
        raise ValueError("model changed the actual frozen034 consumed tilt")
    site = {
        k: nominal["site"][v]
        for k, v in (("lat", "lat_deg"), ("lon", "lon_deg"), ("alt", "height_m"))
    }
    columns, qc = simulate_native(
        bm,
        frequency,
        sky,
        horizon,
        native["time_best"],
        point["el_drive_deg"][joined],
        point["az_drive_deg"][joined],
        rotations,
        mis,
        site,
        out / "native-checkpoints",
        chunk_rows=int(config["chunk_rows"]),
    )
    means, counts = reduce_columns(columns, accepted, mapping, n_bins)
    np.testing.assert_array_equal(counts, expected_count)
    # Sky excludes the CMB; restore the SAME034 analytic unocculted CMB.
    provenance = json.loads(assets.checked_file(config["provenance034"]).read_text())
    tcmb = float(provenance["params"]["model"]["t_cmb_k"])
    native_a = columns[..., 0] + tcmb * (1 - columns[..., 1])
    mean_a = means[..., 0] + tcmb * (1 - means[..., 1])
    assets.atomic_npz(
        out / "columns.npz",
        channels=channels,
        frequencies_mhz=frequency,
        native_index=rows,
        fine_bin_id=mapping,
        accepted=accepted,
        **native,
        native_S=columns[..., 0],
        native_fg=columns[..., 1],
        native_A=native_a,
        bin_S=means[..., 0],
        bin_fg=means[..., 1],
        bin_A=mean_a,
        bin_count=counts,
    )
    return {
        "model_qc": qc,
        "beam": beam_identity,
        "iers": iers,
        "gsm_checks": gsm_checks,
        "horizon": horizon_identity,
        "calibration_stage": config["native_stage"]["sha256"],
        "pointing_stage": config["pointing_stage"]["sha256"],
        "columns_sha256": assets.sha256(out / "columns.npz"),
        "scope": "diagnostic native-time columns; no fit/adoption",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        config = json.loads(args.inputs.read_text())
        root = Path(__file__).resolve().parents[3]
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        if subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True
        ).strip():
            raise ValueError("native model requires a clean frozen source checkout")
        source_dir = args.out / "sources"
        source_dir.mkdir()
        sources = {}
        for source in Path(__file__).parent.glob("*.py"):
            shutil.copyfile(source, source_dir / source.name)
            sources[source.name] = assets.sha256(source)
        identity = {
            "source_commit": head,
            "source_sha256": sources,
            "input_manifest_sha256": assets.sha256(args.inputs),
            "python": sys.version,
            "runtime": {
                d.metadata["Name"]: d.version for d in metadata.distributions()
            },
        }
        assets.write_json(args.out / "RUN_START.json", identity)
        qc = run(config, args.out)
        assets.write_json(
            args.out / "RUN_DONE.json",
            {
                **identity,
                **qc,
                "status": "DONE",
                "elapsed_s": time.monotonic() - start,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
        )
    except BaseException as exc:
        assets.write_json(
            args.out / "RUN_FAILED.json",
            {
                "status": "FAILED",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "elapsed_s": time.monotonic() - start,
            },
        )
        raise


if __name__ == "__main__":
    main()
