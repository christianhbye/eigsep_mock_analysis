"""Self-describing header for the ensemble npz.

The failure this guards against is a data file outliving its generating
script -- exactly what happened to the ensemble this project replaces. So
the requirement is stronger than recording metadata: someone holding only
the npz, with no access to this repository, must be able to regenerate it.

``rebuild_params`` is the machine-check. It deliberately imports nothing
from ``priors``: reconstructing the draw from the header alone is what
proves the header is sufficient rather than merely populated.

Pure module; imports numpy/scipy/stdlib only.
"""

import datetime as dt
import json
import platform
import socket
import subprocess
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from scipy.stats import qmc

HEADER_VERSION = 1


def git_info(path):
    """Remote, commit and dirty flag for the repo containing ``path``.

    All three are None/False outside a repository rather than raising --
    provenance capture must never be the thing that fails a long run.
    """

    def run(*args):
        try:
            out = subprocess.run(
                ("git", "-C", str(path), *args),
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, OSError):
            return None
        return out.stdout.strip()

    return {
        "remote": run("config", "--get", "remote.origin.url"),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def package_versions(names):
    """Installed version per package name, None where not installed."""
    out = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def build_header(**fields):
    """Assemble the provenance header as an indented JSON string."""
    header = dict(fields)
    header["header_version"] = HEADER_VERSION
    header["created_utc"] = dt.datetime.now(dt.UTC).isoformat()
    header["hostname"] = socket.gethostname()
    header["platform"] = platform.platform()
    header["python_version"] = platform.python_version()
    header.setdefault(
        "citations",
        [
            "Munoz 2023a, arXiv:2302.08506 (Zeus21)",
            "Cruz et al. 2024, arXiv:2407.18294 (Pop III, LW, relative velocities)",
            "McGreer et al. 2015 (xHI dark-pixel limit behind the cut)",
            "Planck 2018 (Aghanim et al.), the fixed cosmology",
            "https://github.com/JulianBMunoz/Zeus21",
        ],
    )
    return json.dumps(header, indent=2, sort_keys=True)


def parse_header(text):
    """Parse a header written by :func:`build_header`."""
    return json.loads(str(text))


def rebuild_params(header):
    """Regenerate the parameter draw from ``header`` and nothing else.

    Imports nothing from ``priors`` on purpose -- see the module docstring.

    Note: scipy's Sobol scrambling is deterministic for a given seed but is
    not guaranteed stable across scipy releases, which is why the header
    records the scipy version and the npz stores ``params`` outright. The
    stored array is ground truth; this function is the release-time gate.
    """
    sampler = header["sampler"]
    if sampler["kind"] != "sobol":
        raise ValueError(f"unsupported sampler {sampler['kind']!r}")
    varied = header["varied"]
    lo = np.array([v["lo"] for v in varied])
    hi = np.array([v["hi"] for v in varied])
    unit = qmc.Sobol(
        d=len(varied), scramble=sampler["scramble"], seed=sampler["seed"]
    ).random_base2(sampler["m"])
    return lo + unit * (hi - lo)
