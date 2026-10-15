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
