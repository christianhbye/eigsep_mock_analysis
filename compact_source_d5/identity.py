"""Immutable input and producer identity for portable, historical products."""

import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
PACKAGES = ("numpy", "scipy", "healpy", "pygdsm", "astropy", "h5py", "matplotlib")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_inputs(root):
    """Verify only the pinned phase-one inputs; no map arrays are loaded."""
    root = Path(root).resolve()
    pins = json.loads((PACKAGE / "frozen_inputs.json").read_text())
    records = {}
    for role, entry in pins["inputs"].items():
        path = (root / entry["relative_path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"missing or escaped pinned input {role}: {path}")
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"SHA256 mismatch for {role}")
        records[role] = dict(entry, path=str(path))
    return records


def producer_identity():
    """Bind actual executable sources/runtime, without relabelling old products."""
    files = {
        p.name: sha256(p)
        for p in sorted(PACKAGE.iterdir())
        if p.suffix in (".py", ".json", ".md", ".txt")
    }
    repo = PACKAGE.parent
    state = {}
    for name, args in [
        ("commit", ["rev-parse", "HEAD"]),
        ("dirty", ["status", "--porcelain"]),
    ]:
        q = subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        )
        state[name] = q.stdout.strip()
    return dict(
        source_sha256=files,
        git=state,
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        host=platform.node(),
        packages_all={d.metadata["Name"]: d.version for d in metadata.distributions()},
        packages={p: metadata.version(p) for p in PACKAGES},
    )


def snapshot_producer(out, identity):
    destination = Path(out) / "producer"
    destination.mkdir()
    for name, digest in identity["source_sha256"].items():
        shutil.copyfile(PACKAGE / name, destination / name)
        if sha256(destination / name) != digest:
            raise RuntimeError("producer changed while taking snapshot")


def finalize(out, identity, inputs, results):
    """Hash every completed payload/snapshot and record its actual producer."""
    out = Path(out)
    write_json(out / "results.json", results)
    records = {
        str(p.relative_to(out)): sha256(p)
        for p in sorted(out.rglob("*"))
        if p.is_file()
        and p.name != "PRODUCT.json"
        and "astropy-cache" not in p.relative_to(out).parts
    }
    manifest = dict(
        schema=1,
        label="diagnostic; conditional map accounting only",
        producer=identity,
        inputs=inputs,
        files=records,
        phase2_source_replacement_allowed=False,
    )
    write_json(out / "PRODUCT.json", manifest)
    return manifest


def read_product(out):
    """Verify historical payload against its bound producer, not current code."""
    out = Path(out).resolve()
    manifest = json.loads((out / "PRODUCT.json").read_text())
    if manifest.get("schema") != 1:
        raise ValueError("unsupported product schema")
    for relative, digest in manifest["files"].items():
        path = (out / relative).resolve()
        if (
            not isinstance(relative, str)
            or not relative
            or not isinstance(digest, str)
            or re.fullmatch("[0-9a-f]{64}", digest) is None
            or not path.is_relative_to(out)
            or not path.is_file()
            or sha256(path) != digest
        ):
            raise ValueError(f"payload identity mismatch: {relative}")
    for name, digest in manifest["producer"]["source_sha256"].items():
        if (
            Path(name).name != name
            or manifest["files"].get("producer/" + name) != digest
        ):
            raise ValueError("declared producer is not bound to source snapshot")
    if (
        "RUN_STARTED.json" not in manifest["files"]
        or "results.json" not in manifest["files"]
    ):
        raise ValueError("missing bound start/results record")
    started = json.loads((out / "RUN_STARTED.json").read_text())
    if (
        started["producer"] != manifest["producer"]
        or started["inputs"] != manifest["inputs"]
    ):
        raise ValueError("producer/input identity differs from bound start record")
    return manifest, json.loads((out / "results.json").read_text())
