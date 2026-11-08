"""Build only explicitly owned local sources; retain real snapshot distributions."""

import hashlib
import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path


def owned_sources(analysis, owner):
    """Validate every local locked build source and its writable Git metadata."""
    analysis, owner = Path(analysis).resolve(), Path(owner).resolve()
    lock = tomllib.loads((analysis / "uv.lock").read_text())
    rows = []
    for package in lock["package"]:
        source = package["source"]
        local = source.get("editable", source.get("directory"))
        if local is None:
            continue
        path = (analysis / local).resolve()
        if not path.is_relative_to(owner):
            raise ValueError(f"local build source escapes owned prefix: {path}")
        if not (path / "pyproject.toml").is_file():
            raise ValueError(f"missing local build project: {path}")
        env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
        git = ["git", "-C", str(path)]
        gitdir = Path(
            subprocess.check_output(
                git + ["rev-parse", "--absolute-git-dir"], text=True, env=env
            ).strip()
        ).resolve()
        if not gitdir.is_relative_to(owner):
            raise ValueError(f"local build Git metadata escapes owned prefix: {path}")
        commit = subprocess.check_output(
            git + ["rev-parse", "HEAD"], text=True, env=env
        ).strip()
        if subprocess.check_output(
            git + ["status", "--porcelain", "--untracked-files=no"],
            text=True,
            env=env,
        ).strip():
            raise ValueError(f"dirty tracked build source: {path}")
        rows.append(dict(name=package["name"], root=str(path), commit=commit))
    return rows


def retain_healjax(environment, destination):
    """Copy an actual installed distribution, preserving metadata and RECORD.

    The executable payload is fingerprinted by the producer. No Git revision
    is inferred for this non-editable distribution and no RECORD is invented.
    """
    source = Path(environment).resolve() / "lib/python3.12/site-packages"
    target = Path(destination).resolve() / "lib/python3.12/site-packages"
    candidates = list(source.glob("healjax-*.dist-info"))
    if len(candidates) != 1 or candidates[0].is_symlink():
        raise ValueError("one actual installed healjax distribution is required")
    metadata = candidates[0]
    if not (source / "healjax").is_dir() or not (metadata / "RECORD").is_file():
        raise ValueError("healjax payload/RECORD unavailable")
    files = {}
    for directory in [source / "healjax", metadata]:
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.is_symlink():
                raise ValueError("retained distribution must contain actual files")
            files[str(path.relative_to(source))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    for name in ["healjax", metadata.name]:
        if (target / name).exists():
            raise ValueError("refuse to overwrite an existing healjax distribution")
    for name in files:
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, dest)
        if hashlib.sha256(dest.read_bytes()).hexdigest() != files[name]:
            raise ValueError("retained distribution copy differs")
    return dict(
        source_environment=str(environment),
        files=files,
        payload_sha256=hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
        revision_claim=None,
    )
