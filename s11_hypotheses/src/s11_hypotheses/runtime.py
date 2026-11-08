"""Explicit separate private-analysis runtime; no version-only source claim."""

import hashlib
import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from eigsep_analysis import paths


def contract():
    """Read the installed immutable companion-source contract."""
    return json.loads(Path(__file__).with_name("runtime_contract.json").read_text())


def verify_checkout(actual, expected):
    """Require the pinned commit, clean tracked files and exact lock bytes."""
    actual = Path(actual).resolve()
    read_only_git = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    commit = subprocess.check_output(
        ["git", "-C", str(actual), "rev-parse", "HEAD"], text=True, env=read_only_git
    ).strip()
    if commit != expected["analysis_commit"]:
        raise ValueError("analysis checkout differs from the pinned source contract")
    dirty = subprocess.check_output(
        ["git", "-C", str(actual), "status", "--porcelain", "--untracked-files=no"],
        text=True,
        env=read_only_git,
    ).strip()
    if dirty:
        raise ValueError("analysis has dirty tracked source/configuration")
    lock = hashlib.sha256((actual / "uv.lock").read_bytes()).hexdigest()
    if lock != expected["analysis_lock_sha256"]:
        raise ValueError("analysis uv.lock differs from the pinned source contract")
    return commit, lock


def verify_environment(code_root=None):
    """Refuse wrong companion source/lock, import binding or JAX runtime.

    The producer's transitive source/payload fingerprint remains responsible
    for executable identities, including the supported noAVX conda case.
    This guard does not fabricate wheel records or accept version alone.
    """
    expected = contract()
    actual = Path(paths.CODE).resolve()
    if code_root is not None and Path(code_root).resolve() != actual:
        raise ValueError("analysis import differs from declared code_root")
    commit, lock = verify_checkout(actual, expected)
    if f"{sys.version_info.major}.{sys.version_info.minor}" != expected["python_minor"]:
        raise ValueError("Python minor differs from the declared runtime")
    versions = {
        name: metadata.version(name)
        for name in ["numpy", "scipy", "h5py", "pandas", "jax", "jaxlib"]
    }
    for name, version in expected["expected_versions"].items():
        if versions[name] != version:
            raise ValueError(f"{name}: separate C54 runtime requires {version}")
    if (
        metadata.version(expected["analysis_distribution"])
        != expected["analysis_version"]
    ):
        raise ValueError("analysis distribution version differs")
    return dict(
        analysis_root=str(actual),
        analysis_commit=commit,
        analysis_lock_sha256=lock,
        versions=versions,
        interpreter=sys.executable,
        python=sys.version,
    )
