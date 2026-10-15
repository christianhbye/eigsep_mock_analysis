"""Plan or execute the checkpointed D5 reference-convergence study.

--dry-run is read-only. --execute requires private Limbo paths and a frozen gate.
--probe executes only nominal-position tasks and never writes complete RESULTS.
"""

import argparse
import json
import os
import platform
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
        "runner_sha256": "horizon_position_d5/convergence_runner.py",
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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        "--inventory-only",
        action="store_true",
        help="validate pinned upstream exports and print work inventory",
    )
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args(argv)
    if args.probe and not args.execute:
        parser.error("--probe requires --execute")
    plan = ConvergencePlan.from_dict(json.loads(args.plan.read_text()))
    pilot, ensemble, pilot_arrays, ensemble_arrays = load_upstreams(
        plan, args.pilot, args.ensemble
    )
    inventory = build_inventory(
        plan, pilot, ensemble, pilot_arrays, ensemble_arrays, source=source_identity()
    )
    if args.dry_run:
        print(json.dumps(inventory, indent=2, sort_keys=True, allow_nan=False))
        return
    if platform.node().split(".")[0] != "limbo":
        parser.error("science execution belongs on Limbo")
    if any(v is None for v in (args.inputs, args.output, args.gate)):
        parser.error("--execute requires --inputs, --output and --gate")
    from .inputs import verify_inputs
    from .launch import private_path

    inputs, output = private_path(args.inputs), private_path(args.output)
    protected = [
        inputs,
        Path(__file__).resolve().parents[1],
        private_path(args.pilot),
        private_path(args.ensemble),
    ]
    if any(
        output == p or output.is_relative_to(p) or p.is_relative_to(output)
        for p in protected
    ):
        parser.error("output must be separate from source, inputs and upstream results")
    gate = json.loads(args.gate.read_text())
    phase = "reference-probe" if args.probe else "reference-convergence"
    if (
        gate.get("producer_commit") != inventory["source"]["source_commit"]
        or gate.get("plan_sha256") != file_sha256(args.plan)
        or gate.get("inventory_hash") != inventory["inventory_hash"]
        or gate.get("workers") != args.workers
        or phase not in gate.get("allowed_phases", [])
    ):
        parser.error(
            "gate does not pin this source, plan, inventory, workers and phase"
        )
    if (
        type(args.workers) is not int
        or args.workers < 1
        or not args.workers
        <= len(os.sched_getaffinity(0))
        <= gate.get("allocated_cpus", 0)
        or any(
            os.environ.get(k) != "1"
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        )
    ):
        parser.error(
            "execution requires the gate's CPU affinity and single-thread libraries"
        )
    hashes = {k: v["sha256"] for k, v in verify_inputs(inputs).items()}
    if hashes != pilot["identity"]["input_hashes"]:
        parser.error("inputs differ from the pinned upstream model")
    import jax

    jax.config.update("jax_enable_x64", True)
    from .convergence_runner import ConvergenceRunner, TemperatureEvaluator
    from .runtime import runtime_record

    runtime = runtime_record()
    runtime["input_hashes"] = hashes
    evaluator = TemperatureEvaluator(
        pilot_arrays["response"], pilot["identity"]["config"]["lmax"]
    )
    runner = ConvergenceRunner(
        output,
        inventory,
        runtime,
        workers=args.workers,
        dem_path=inputs / "dem.npz",
        evaluator=evaluator,
    )
    report = runner.run(probe=args.probe)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
