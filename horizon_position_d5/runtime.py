"""Run records for private, checkpointed Limbo computations."""

import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path

from .cache import file_sha256


def runtime_record():
    """Capture source, dependency lock, runtime versions and resource settings."""
    import jax
    import jaxlib

    import eigsim

    root = Path(__file__).resolve().parents[1]

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    dirty = git("status", "--porcelain")
    if dirty:
        raise RuntimeError("commit the source before running the science study")
    return {
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "lock_sha256": file_sha256(root / "uv.lock"),
        "host": platform.node(),
        "python": platform.python_version(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jaxlib_file": jaxlib.__file__,
        "eigsim_file": eigsim.__file__,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "numpy", "scipy", "astropy", "croissant-sim")
        },
    }
