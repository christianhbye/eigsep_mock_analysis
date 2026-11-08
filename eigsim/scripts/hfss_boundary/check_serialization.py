"""Pure-stdlib source-signature parity check; no Desktop, model write or license API.

Run with CPython3 and standalone vendor IronPython2 against the same exact pins.
"""

from __future__ import print_function

import argparse
import json
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard", required=True)
    parser.add_argument("--pin", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--old-guard")
    args = parser.parse_args()
    guard = runpy.run_path(args.guard)
    with open(args.pin) as stream:
        pin = json.load(stream)
    protected = guard["protected_hashes"](pin)
    source = pin["source_path"]
    result = {
        "runtime": sys.version,
        "version_info": list(sys.version_info),
        "executable": sys.executable,
        "guard_sha256": guard["sha256"](args.guard),
        "source_sha256": guard["sha256"](source),
        "expected_signature": pin["physical_signature_sha256"],
        "scope": "Pure stdlib: no Desktop/API/project save/solve/license call",
    }
    result["actual_signature"] = guard["digest"](
        guard["clone_signature"](source, pin["design"])
    )
    assert result["actual_signature"] == result["expected_signature"]
    if args.old_guard:
        old = runpy.run_path(args.old_guard)
        try:
            result["old_signature"] = old["digest"](
                old["clone_signature"](source, pin["design"])
            )
        except UnicodeDecodeError as error:
            result["old_error"] = repr(error)
        if sys.version_info[0] == 2:
            assert "old_error" in result, "Native failure was not reproduced"
        else:
            assert result["old_signature"] == result["expected_signature"]
    assert guard["protected_hashes"](pin) == protected
    result["all5_protected_unchanged"] = True
    result["passed"] = True
    guard["write_result"](args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
