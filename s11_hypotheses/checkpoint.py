"""Fail-closed C54 source/input identities and immutable checkpoint units."""

import ast
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path

import h5py
import numpy as np
from eigsep_analysis import run_contract as contract
from eigsep_analysis import s11_hypothesis_io as intake


def fingerprint(value):
    return contract.fingerprint(value)


def source_identity(mock_root, code_root):
    """Source-only transitive code/dependency closure; no notebook outputs.

    Existing041's strict AST/lock dependency implementation resolves code
    imports transitively. Mock package source is all scanned (including lazy
    paths); dynamic imports are rejected. Git commits are audit metadata,
    never the cache key: a report-only/output commit need not invalidate.
    """
    root = Path(mock_root)
    sources = {}
    for path in sorted((root / "s11_hypotheses").rglob("*.py")):
        if "/tests/" in str(path):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if (
                    name.endswith(
                        ("import_module", "exec_module", "spec_from_file_location")
                    )
                    or name == "__import__"
                ):
                    raise ValueError(f"unaudited dynamic import: {path}")
        sources[str(path.relative_to(root))] = intake.digest(path)
    for name in ["pyproject.toml", "s11_hypotheses/pyproject.toml"]:
        sources[name] = intake.digest(root / name)
    code = contract.code_manifest(
        code_root,
        [
            "src/eigsep_analysis/s11_hypotheses.py",
            "src/eigsep_analysis/s11_hypothesis_io.py",
            "src/eigsep_analysis/s11_hypothesis_pipeline.py",
            "src/eigsep_analysis/s11_hypothesis_report.py",
            "src/eigsep_analysis/beammap/beam.py",
            "src/eigsep_analysis/night_fit.py",
        ],
    )
    return dict(
        schema="c54-source-v1",
        mock_sources=sources,
        code=code,
        runtime={
            k: metadata.version(k)
            for k in ("numpy", "scipy", "h5py", "hera-filters", "matplotlib")
        },
    )


def require_committed(mock_root, code_root, identity):
    """Refuse modified/untracked consumed sources before any producer work."""
    for root, sources in [
        (mock_root, identity["mock_sources"]),
        (code_root, identity["code"]["sources"]),
    ]:
        for name, digest in sources.items():
            blob = subprocess.check_output(
                ["git", "-C", str(root), "show", f"HEAD:{name}"]
            )
            import hashlib

            if hashlib.sha256(blob).hexdigest() != digest:
                raise ValueError(f"uncommitted consumed source: {name}")


def write_json(path, obj):
    """Atomic JSON, reject overwriting a different immutable record."""
    path = Path(path)
    payload = (
        json.dumps(
            _clean(obj), indent=2, sort_keys=True, default=_plain, allow_nan=False
        )
        + "\n"
    )
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"immutable record differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload)
    os.replace(tmp, path)


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _clean(value.item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_h5(path, arrays):
    """Atomic array export preserving measured masks and coordinate arrays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"refusing to overwrite {path}")
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".h5.tmp")
    os.close(fd)
    try:
        with h5py.File(temp, "w") as out:
            for name, value in arrays.items():
                if value is None:
                    continue
                a = np.asarray(value)
                if a.dtype.kind in "OUS":
                    out.attrs[name] = json.dumps(value, default=_plain)
                else:
                    out.create_dataset(
                        name, data=a, compression="gzip" if a.ndim else None
                    )
        os.replace(temp, path)
    finally:
        if Path(temp).exists():
            Path(temp).unlink()


class Unit:
    """A resumable unit is reusable only with exact source/input/result bytes."""

    def __init__(self, root, name, key):
        self.path = Path(root) / "units" / name
        self.key = key
        self.start = time.monotonic()
        self.cpu = time.process_time()

    def completed(self):
        marker = self.path / "DONE.json"
        if not marker.exists():
            if self.path.exists() and any(self.path.iterdir()):
                raise ValueError(
                    f"incomplete unit retained; new run needed: {self.path}"
                )
            return False
        row = json.loads(marker.read_text())
        if row["key"] != self.key:
            raise ValueError(f"checkpoint identity mismatch: {self.path}")
        for name, digest in row["results"].items():
            if intake.digest(self.path / name) != digest:
                raise ValueError(f"checkpoint result changed: {name}")
        return True

    def finish(self, metadata_row):
        files = {
            str(p.relative_to(self.path)): intake.digest(p)
            for p in self.path.rglob("*")
            if p.is_file() and p.name != "DONE.json"
        }
        usage = resource.getrusage(resource.RUSAGE_SELF)
        write_json(
            self.path / "DONE.json",
            dict(
                key=self.key,
                results=files,
                metadata=metadata_row,
                wall_s=time.monotonic() - self.start,
                cpu_s=time.process_time() - self.cpu,
                max_rss_kib=usage.ru_maxrss,
                block_inputs=usage.ru_inblock,
                block_outputs=usage.ru_oublock,
                interpreter=sys.executable,
            ),
        )
