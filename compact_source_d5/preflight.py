"""Verify staged inputs/imports without loading a map or constructing GSM."""

import argparse
import importlib
from pathlib import Path

from .identity import PACKAGE, producer_identity, verify_inputs, write_json
from .run import verify_runtime


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    identity = producer_identity()
    if identity["git"]["dirty"]:
        raise ValueError("preflight requires committed producer")
    pins = json.loads((PACKAGE / "frozen_inputs.json").read_text())
    loader = verify_runtime(identity, pins)
    imports = {
        name: str(importlib.import_module(name).__file__)
        for name in ["numpy", "scipy", "healpy", "h5py", "astropy", "pygdsm"]
    }
    inputs = verify_inputs(args.inputs)
    if args.out.exists():
        raise ValueError("do not overwrite historical preflight")
    write_json(
        args.out,
        dict(
            label="input/import readiness, no map generation",
            producer=identity,
            imports=imports,
            pygdsm_loader=loader,
            inputs=inputs,
        ),
    )


if __name__ == "__main__":
    main()
