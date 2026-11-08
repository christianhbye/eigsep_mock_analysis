"""Fail-closed frequency-chunk assembly; no maps, SHT or fits are run.

python -m fine_night_d5.join --inputs PINNED_JOIN_JSON --out NEW_STAGE
Every source chunk retains its actual producer snapshot and runtime proof.
"""

import argparse
import importlib.metadata as metadata
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

from .assets import atomic_npz, checked_file, sha256, write_json
from .streaming import reduce_columns

IDENTITY = ("native_index", "fine_bin_id", "source_file", "row", "input", "time_best")
NATIVE = ("native_S", "native_fg", "native_A")
BINNED = ("bin_S", "bin_fg", "bin_A", "bin_count")
PHYSICS_SOURCES = {
    "assets.py",
    "beam.py",
    "model.py",
    "streaming.py",
    "time_operator.py",
}


def context(config, record, runtime):
    """Compare historical physical identities, never current checkout bytes."""
    if record["input_manifest_sha256"] != config["_verified_manifest_sha256"]:
        raise ValueError("chunk does not bind its exact model manifest")
    if record["calibration_stage"] != config["native_stage"]["sha256"]:
        raise ValueError("chunk calibration binding differs")
    if record["pointing_stage"] != config["pointing_stage"]["sha256"]:
        raise ValueError("chunk pointing binding differs")
    if record["beam"] != {
        "role": config["beam"]["role"],
        "input_sha256": config["beam"]["sha256"],
    }:
        raise ValueError("chunk beam role/source binding differs")
    required = {"eigsim.simulate", "eigsim.rotations", "croissant.simulator"}
    sources = runtime["dependency_sources"]
    if (
        not required.issubset(config["runtime_modules"])
        or not required.issubset(sources)
        or not runtime["jax_binaries"]
        or any(sources[k]["sha256"] != config["runtime_modules"][k] for k in required)
    ):
        raise ValueError("actual runtime evidence differs from model inputs")
    if not PHYSICS_SOURCES.issubset(record["source_sha256"]):
        raise ValueError("chunk lacks actual physical producer snapshots")
    return {
        "configuration": {
            k: v
            for k, v in config.items()
            if k not in {"channels", "_verified_manifest_sha256", "runtime_evidence"}
        },
        "physical_source_sha256": {
            k: record["source_sha256"][k] for k in sorted(PHYSICS_SOURCES)
        },
        "actual_model_producer_commit": record["source_commit"],
        "actual_model_runtime_versions": record["runtime"],
        "runtime_dependency_sha256": {k: v["sha256"] for k, v in sources.items()},
        "runtime_binary_sha256": sorted(
            v["sha256"] for v in runtime["jax_binaries"].values()
        ),
        "iers": record["iers"],
        "horizon": record["horizon"],
    }


def join_arrays(pieces, expected_channels):
    """Assemble disjoint exact channels with identical row/bin/context IDs."""
    expected = np.asarray(expected_channels)
    if (
        expected.ndim != 1
        or expected.dtype.kind not in "iu"
        or len(expected) == 0
        or np.any(np.diff(expected) <= 0)
    ):
        raise ValueError("unique increasing exact expected channels required")
    if not pieces:
        raise ValueError("no verified model chunks")
    first, reference = pieces[0]
    n = len(first["time_best"])
    if n == 0 or any(np.shape(first[k]) != (n,) for k in IDENTITY):
        raise ValueError("native identity axes missing")
    for key in ("native_index", "row"):
        a = np.asarray(first[key])
        if a.dtype.kind not in "iu" or np.any(a < 0):
            raise ValueError("integer native identity required")
    if not np.isfinite(first["time_best"]).all():
        raise ValueError("finite native times required")
    keys = list(
        zip(
            first["source_file"].tolist(),
            first["row"].tolist(),
            first["input"].tolist(),
        )
    )
    if len(set(keys)) != n or len(np.unique(first["native_index"])) != n:
        raise ValueError("duplicate native identity")
    seen, arrays = [], []
    for data, identity in pieces:
        if identity != reference:
            raise ValueError("model chunk physical context/runtime/beam differs")
        for key in IDENTITY:
            np.testing.assert_array_equal(data[key], first[key], err_msg=key)
        channels = np.asarray(data["channels"])
        if (
            channels.ndim != 1
            or channels.dtype.kind not in "iu"
            or len(channels) == 0
            or np.any(np.diff(channels) <= 0)
            or not np.array_equal(data["frequencies_mhz"], channels * 250 / 1024)
        ):
            raise ValueError("native frequency/channel axes differ")
        seen.extend(channels.tolist())
        n, nf = len(data["time_best"]), len(channels)
        accepted = np.asarray(data["accepted"])
        count = np.asarray(data["bin_count"])
        if (
            accepted.dtype.kind != "b"
            or accepted.shape != (n, nf)
            or count.ndim != 2
            or count.shape[1] != nf
            or count.dtype.kind not in "iu"
            or np.any(count < 0)
        ):
            raise ValueError("chunk mask/count axes differ")
        for name in NATIVE:
            if data[name].shape != (n, nf) or not np.isfinite(data[name]).all():
                raise ValueError("chunk native model coverage missing")
        values = np.stack([data["native_S"], data["native_fg"], data["native_A"]], -1)
        means, actual = reduce_columns(
            values, accepted, data["fine_bin_id"], len(count)
        )
        np.testing.assert_array_equal(actual, count)
        for i, name in enumerate(("bin_S", "bin_fg", "bin_A")):
            np.testing.assert_allclose(
                data[name],
                means[..., i],
                rtol=1e-13,
                atol=1e-12,
                equal_nan=True,
                err_msg=name,
            )
        arrays.append(data)
    if len(set(seen)) != len(seen):
        raise ValueError("duplicate/overlapping frequency chunks")
    order = np.argsort(seen)
    if not np.array_equal(np.asarray(seen)[order], expected):
        raise ValueError("missing/unexpected native frequency coverage")
    output = {k: first[k] for k in IDENTITY}
    output["channels"], output["frequencies_mhz"] = expected, expected * 250 / 1024
    for key in (*NATIVE, *BINNED, "accepted"):
        output[key] = np.concatenate([a[key] for a in arrays], axis=1)[:, order]
    return output, reference


def read_piece(pin):
    done_path, input_path = checked_file(pin["done"]), checked_file(pin["inputs"])
    record, config = (
        json.loads(done_path.read_text()),
        json.loads(input_path.read_text()),
    )
    if record["status"] != "DONE":
        raise ValueError("incomplete model chunk")
    directory = done_path.parent
    if sha256(directory / "columns.npz") != record["columns_sha256"]:
        raise ValueError("chunk payload checksum differs")
    snapshots = {}
    for name, digest in record["source_sha256"].items():
        if Path(name).name != name or sha256(directory / "sources" / name) != digest:
            raise ValueError("chunk historical producer snapshot differs")
        snapshots[name] = digest
    runtime_pin = pin.get("runtime", config["runtime_evidence"])
    if runtime_pin["sha256"] != config["runtime_evidence"]["sha256"]:
        raise ValueError("runtime read view differs from historical evidence")
    runtime = json.loads(checked_file(runtime_pin).read_text())
    config["_verified_manifest_sha256"] = pin["inputs"]["sha256"]
    identity = context(config, record, runtime)
    with np.load(directory / "columns.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    if not np.array_equal(arrays["channels"], np.asarray(config["channels"])):
        raise ValueError("chunk payload channel IDs differ from its producer inputs")
    proof = {
        "done": pin["done"],
        "inputs": pin["inputs"],
        "actual_producer_commit": record["source_commit"],
        "actual_producer_source_sha256": snapshots,
        "columns_sha256": record["columns_sha256"],
        "runtime_evidence": config["runtime_evidence"],
        "runtime_read_view": runtime_pin,
    }
    return (arrays, identity), proof, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        config = json.loads(args.inputs.read_text())
        if config["schema"] != 1 or config["label"] != "diagnostic":
            raise ValueError("diagnostic chunk join schema required")
        loaded = [read_piece(pin) for pin in config["chunks"]]
        arrays, identity = join_arrays([p[0] for p in loaded], config["channels"])
        if loaded[0][2]["beam"]["role"] != config["beam_role"]:
            raise ValueError("joined model beam role differs")
        source_root = Path(__file__).resolve().parents[3]
        if subprocess.check_output(
            ["git", "-C", str(source_root), "status", "--porcelain"], text=True
        ).strip():
            raise ValueError("join requires a clean own source checkout")
        commit = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip()
        (args.out / "sources").mkdir()
        sources = {}
        for source in Path(__file__).parent.glob("*.py"):
            shutil.copyfile(source, args.out / "sources" / source.name)
            sources[source.name] = sha256(source)
        atomic_npz(args.out / "columns.npz", **arrays)
        first_record = loaded[0][2]
        write_json(
            args.out / "RUN_DONE.json",
            {
                "status": "DONE",
                "label": "diagnostic joined model columns; no new model generation",
                "source_commit": commit,
                "source_sha256": sources,
                "input_manifest_sha256": sha256(args.inputs),
                "columns_sha256": sha256(args.out / "columns.npz"),
                "calibration_stage": first_record["calibration_stage"],
                "pointing_stage": first_record["pointing_stage"],
                "beam": first_record["beam"],
                "python": sys.version,
                "runtime": {"numpy": metadata.version("numpy")},
                "actual_model_context": identity,
                "chunks": [p[1] for p in loaded],
                "sky_or_model_regenerated": False,
            },
        )
    except BaseException as exc:
        write_json(
            args.out / "RUN_FAILED.json",
            {
                "status": "FAILED",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
