"""Verify the actual installed src package and companion source/runtime."""

import argparse
import json
import os
from pathlib import Path

import s11_hypotheses
from s11_hypotheses import checkpoint, runtime


def main():
    os.environ["GIT_OPTIONAL_LOCKS"] = "0"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if not Path(s11_hypotheses.__file__).resolve().is_relative_to(project / "src"):
        raise ValueError("tests imported another flat/installed package")
    result = runtime.verify_environment(args.analysis_root)
    result["installed_package"] = str(Path(s11_hypotheses.__file__).resolve())
    identity = checkpoint.source_identity(project.parent, args.analysis_root)
    result["source_fingerprint"] = checkpoint.fingerprint(identity)
    result["source_identity"] = identity
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, indent=2)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "analysis_root",
                    "analysis_commit",
                    "source_fingerprint",
                    "installed_package",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
