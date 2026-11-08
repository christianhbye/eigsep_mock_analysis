"""Install into a NEW own environment from the pinned companion lock.

The private analysis project has editable source dependencies; it is not a
self-contained PyPI dependency. Prepare its exact sibling checkouts first.
This script never resolves the mock workspace's JAX0.9.2 into C54's0.10.2.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from s11_hypotheses.install_contract import owned_sources, retain_healjax


def prepare(
    analysis, environment, owner, retained_healjax, python="3.12.12", offline=True
):
    project = Path(__file__).resolve().parents[1]
    mock = project.parent
    policy = json.loads(
        (project / "src/s11_hypotheses/runtime_contract.json").read_text()
    )
    analysis = Path(analysis).resolve()
    environment = Path(environment).resolve()
    owner = Path(owner).resolve()
    if not owner.is_relative_to(mock) or owner == mock:
        raise ValueError("owned build prefix must be inside this mock worktree")
    local_sources = owned_sources(analysis, owner)
    if not environment.is_relative_to(mock) or environment == mock:
        raise ValueError(
            "environment must be a new directory inside this mock worktree"
        )
    if environment.exists():
        raise ValueError(
            "environment already exists; preserve and inspect, never overwrite"
        )
    head = subprocess.check_output(
        ["git", "-C", str(analysis), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != policy["analysis_commit"]:
        raise ValueError("prepare the declared immutable analysis checkout first")
    if subprocess.check_output(
        ["git", "-C", str(analysis), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip():
        raise ValueError("analysis has dirty tracked source/configuration")
    lock = hashlib.sha256((analysis / "uv.lock").read_bytes()).hexdigest()
    if lock != policy["analysis_lock_sha256"]:
        raise ValueError("analysis lock differs")
    env = dict(
        os.environ,
        UV_PROJECT_ENVIRONMENT=str(environment),
        UV_CONCURRENT_BUILDS="1",
        UV_CONCURRENT_INSTALLS="1",
        UV_CONCURRENT_DOWNLOADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        GIT_OPTIONAL_LOCKS="0",
    )
    sync = [
        "uv",
        "sync",
        "--project",
        str(analysis),
        "--frozen",
        "--no-install-package",
        "healjax",
        "--extra",
        "dev",
        "--python",
        python,
    ]
    install = [
        "uv",
        "pip",
        "install",
        "--python",
        str(environment / "bin/python"),
        "--no-deps",
        "--editable",
        str(project),
    ]
    lint = [
        "uv",
        "pip",
        "install",
        "--python",
        str(environment / "bin/python"),
        "--no-deps",
        "ruff==0.16.10",
    ]
    if offline:
        sync.append("--offline")
        install.append("--offline")
        lint.append("--offline")
    record = dict(
        analysis_root=str(analysis),
        analysis_commit=head,
        analysis_lock_sha256=lock,
        environment=str(environment),
        commands=[sync, install, lint],
        offline=offline,
        owned_sources=local_sources,
        retained_healjax_from=str(Path(retained_healjax).resolve()),
        contract=(
            "companion locked runtime first, then installed src package; "
            "no workspace join"
        ),
    )
    path = mock / (environment.name + "-bootstrap.json")
    with path.open("x") as stream:
        json.dump(record, stream, indent=2)
    subprocess.run(sync, cwd=mock, env=env, check=True)
    retained = retain_healjax(retained_healjax, environment)
    with (mock / (environment.name + "-healjax.json")).open("x") as stream:
        json.dump(retained, stream, indent=2)
    for command in [install, lint]:
        subprocess.run(command, cwd=mock, env=env, check=True)
    verify = [
        str(environment / "bin/python"),
        "-B",
        str(project / "scripts/verify_runtime.py"),
        "--analysis-root",
        str(analysis),
        "--output",
        str(mock / (environment.name + "-runtime.json")),
    ]
    subprocess.run(verify, cwd=mock, env=env, check=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--environment", default=".venv-s11")
    parser.add_argument("--owned-source-root", required=True)
    parser.add_argument("--retained-healjax-from", required=True)
    parser.add_argument("--python", default="3.12.12")
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.analysis_root,
                args.environment,
                args.owned_source_root,
                args.retained_healjax_from,
                args.python,
                not args.online,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
