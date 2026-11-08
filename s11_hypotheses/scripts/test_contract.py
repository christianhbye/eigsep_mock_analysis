"""Run the whole configured standalone suite with exact source/runtime records."""

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from s11_hypotheses import checkpoint, runtime


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    os.environ["GIT_OPTIONAL_LOCKS"] = "0"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    mock = project.parent
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    error, result, counts = None, None, None
    (output / "PREPARE_RECORD.json").write_text(
        json.dumps(
            dict(
                started_utc=datetime.now(timezone.utc).isoformat(),
                argv=sys.argv,
                interpreter=sys.executable,
                uid=os.getuid(),
                gid=os.getgid(),
                groups=os.getgroups(),
            ),
            indent=2,
        )
        + "\n"
    )
    try:
        actual = runtime.verify_environment(args.analysis_root)
        identity = checkpoint.source_identity(mock, args.analysis_root)
        head = subprocess.check_output(
            ["git", "-C", str(mock), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if subprocess.check_output(
            ["git", "-C", str(mock), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        ).strip():
            raise ValueError("mock delivery has dirty tracked source/configuration")
        checkpoint.require_committed(mock, args.analysis_root, identity)
        argv = [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            str(project / "tests"),
            "--basetemp",
            str(output / "tmp"),
            "--junitxml",
            str(output / "pytest.xml"),
        ]
        record = dict(
            started_utc=datetime.now(timezone.utc).isoformat(),
            mock_commit=head,
            runtime=actual,
            source_identity=identity,
            source_fingerprint=checkpoint.fingerprint(identity),
            argv=argv,
            pyproject_sha256=sha(project / "pyproject.toml"),
            test_sources={
                str(p.relative_to(mock)): sha(p)
                for p in (project / "tests").glob("*.py")
            },
            test_driver_sha256=sha(__file__),
        )
        (output / "RUN_RECORD.json").write_text(json.dumps(record, indent=2) + "\n")
        with (output / "pytest.log").open("x") as log:
            result = subprocess.run(
                argv, cwd=project, stdout=log, stderr=subprocess.STDOUT
            )
        tree = ET.parse(output / "pytest.xml").getroot()
        suites = list(tree.iter("testsuite"))
        counts = {
            key: sum(int(s.attrib.get(key, 0)) for s in suites)
            for key in ["tests", "errors", "failures", "skipped"]
        }
    except Exception as exc:
        error = repr(exc)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    passed = bool(
        error is None
        and result is not None
        and result.returncode == 0
        and counts is not None
        and not counts["errors"]
        and not counts["failures"]
    )
    done = dict(
        finished_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_s=time.monotonic() - start,
        cpu_s=usage.ru_utime + usage.ru_stime,
        maxrss_kib=usage.ru_maxrss,
        returncode=None if result is None else result.returncode,
        error=error,
        counts=counts,
        xml_sha256=sha(output / "pytest.xml")
        if (output / "pytest.xml").exists()
        else None,
        passed=passed,
    )
    (output / ("RUN_DONE" if done["passed"] else "RUN_FAILED")).write_text(
        json.dumps(done, indent=2) + "\n"
    )
    print(json.dumps(done, indent=2))
    return 0 if done["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
