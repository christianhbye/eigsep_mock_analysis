"""Run the full private mock suite with the study's hashed IERS tables.

Invoke as a module or as a copied, committed script from a detached job directory.
This avoids changing global Astropy download configuration at test collection.
"""

import argparse
from pathlib import Path

import pytest

from horizon_position_d5.inputs import setup_iers, verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    args, pytest_args = parser.parse_known_args()
    verify_inputs(args.inputs)
    setup_iers(args.inputs)
    raise SystemExit(pytest.main(pytest_args))


if __name__ == "__main__":
    main()
