"""Atomic, fingerprinted study blocks; incomplete or stale blocks are misses."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


class BlockCache:
    """One writer per named block; metadata commits a complete array file."""

    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.identity_hash = fingerprint(identity)

    def _paths(self, name):
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("block name must be one filename component")
        return self.directory / f"{name}.npz", self.directory / f"{name}.json"

    def load(self, name, parameters):
        data_path, record_path = self._paths(name)
        if not data_path.exists() or not record_path.exists():
            return None
        try:
            record = json.loads(record_path.read_text())
            if (
                record.get("identity_hash") != self.identity_hash
                or record.get("parameters_hash") != fingerprint(parameters)
                or record.get("sha256") != file_sha256(data_path)
                or record.get("complete") is not True
            ):
                return None
            with np.load(data_path, allow_pickle=False) as data:
                return {key: data[key] for key in data.files}
        except (OSError, ValueError, EOFError, KeyError):
            return None

    def write(self, name, parameters, arrays):
        data_path, record_path = self._paths(name)
        temporary = data_path.with_name(data_path.name + f".{os.getpid()}.partial")
        for key, value in arrays.items():
            if np.asarray(value).dtype.hasobject:
                raise ValueError(f"object array is not a checkpoint: {key}")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(data_path)
        atomic_json(
            record_path,
            {
                "complete": True,
                "identity": self.identity,
                "identity_hash": self.identity_hash,
                "parameters": parameters,
                "parameters_hash": fingerprint(parameters),
                "sha256": file_sha256(data_path),
            },
        )


def load_results(directory, *, expected_phase=None):
    """Load a completed earlier phase, checking its identity and file inventory."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "RESULTS.json").read_text())
    record = json.loads((directory / "RUN_RECORD.json").read_text())
    identity = manifest["identity"]
    digest = fingerprint(identity)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("complete") is not True
        or manifest.get("label") != "diagnostic"
        or manifest["identity_hash"] != digest
        or record["identity_hash"] != digest
        or record["identity"] != identity
        or record["phase"] != manifest["phase"]
        or (expected_phase is not None and manifest["phase"] != expected_phase)
    ):
        raise ValueError("incomplete or inconsistent upstream result")
    blocks = {}
    for name, item in manifest["files"].items():
        paths = []
        for kind in ("data", "metadata"):
            relative = Path(item[kind])
            path = (directory / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(directory):
                raise ValueError("upstream result path leaves its directory")
            if file_sha256(path) != item[f"{kind}_sha256"]:
                raise ValueError(f"upstream {kind} hash mismatch: {name}")
            paths.append(path)
        data, metadata = paths
        meta = json.loads(metadata.read_text())
        if (
            meta.get("complete") is not True
            or meta["identity"] != identity
            or meta["identity_hash"] != digest
            or meta["parameters_hash"] != fingerprint(meta["parameters"])
            or meta["sha256"] != item["data_sha256"]
        ):
            raise ValueError(f"mixed upstream block: {name}")
        with np.load(data, allow_pickle=False) as block:
            blocks[name] = {key: block[key] for key in block.files}
    return manifest, blocks
