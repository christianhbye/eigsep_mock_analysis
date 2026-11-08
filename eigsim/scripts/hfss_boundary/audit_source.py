"""Read-only source-byte and saved-signature check; does not import AEDT."""

import argparse
import json
from pathlib import Path

from eigsim import hfss_boundary_pilot as guard


def audit(source, pin_path):
    pin = json.loads(Path(pin_path).read_text())
    guard.require(
        guard.sha256(source) == pin["source_sha256"], "Source byte identity failed"
    )
    signature = guard.clone_signature(source, pin["design"])
    guard.compare(pin["signature"], signature)
    return dict(
        phase="SOURCE_SIGNATURE_VERIFIED",
        source=str(source),
        source_sha256=guard.sha256(source),
        source_pin_sha256=guard.digest(pin),
        signature_sha256=guard.digest(signature),
        limits="No AEDT operation; remote stored-field identities not checked here",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--pin", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.pin), indent=2, sort_keys=True))
