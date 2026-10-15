"""Read-only inventory for the D5 reference-convergence study.

This first implementation prepares and validates work; it does not execute it.
Run with --plan FILE --pilot DIR --ensemble DIR --dry-run. JSON goes to stdout.
"""

import argparse
import json
import subprocess
from pathlib import Path

from .cache import file_sha256
from .convergence_plan import ConvergencePlan, build_inventory, load_upstreams


def source_identity():
    root = Path(__file__).resolve().parents[1]
    paths = {
        "reference_sha256": "horizon_position_d5/reference.py",
        "temperature_sha256": "horizon_position_d5/temperature.py",
        "mask_sha256": "eigsim/src/eigsim/horizon.py",
        "criteria_sha256": "horizon_position_d5/convergence.py",
        "cache_sha256": "horizon_position_d5/cache.py",
        "pilot_sha256": "horizon_position_d5/pilot.py",
        "planner_sha256": "horizon_position_d5/convergence_plan.py",
        "cli_sha256": "horizon_position_d5/reference_convergence.py",
        "lock_sha256": "uv.lock",
    }

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    return {
        **{name: file_sha256(root / path) for name, path in paths.items()},
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "source_dirty": bool(git("status", "--porcelain")),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        "--inventory-only",
        action="store_true",
        required=True,
        help="validate pinned upstream exports and print work inventory",
    )
    args = parser.parse_args(argv)
    plan = ConvergencePlan.from_dict(json.loads(args.plan.read_text()))
    pilot, ensemble, pilot_arrays, ensemble_arrays = load_upstreams(
        plan, args.pilot, args.ensemble
    )
    inventory = build_inventory(
        plan, pilot, ensemble, pilot_arrays, ensemble_arrays, source=source_identity()
    )
    print(json.dumps(inventory, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
