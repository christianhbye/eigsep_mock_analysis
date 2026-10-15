"""Launch one checkpointed private Limbo stage with a frozen resource gate.

Example: python -m horizon_position_d5.launch --phase pilot --inputs DIR
--output RUN --gate gate.json --commit FULL_SHA --cpus 0,1
--reserved-other-cpus 3 --dry-run

Dry runs inspect the clean source, gate, existing unit and checkpoint inventory;
they create no files. The operator reserves the peak CPU budget of other cbye
jobs, not their momentarily idle usage. The quota here limits this unit; it does
not change or control teammates' jobs. Use the same command without --dry-run
to launch. Finished checkpoints are validated by the selected stage on resume.
"""

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import time
from pathlib import Path

PRIVATE_ROOT = Path("/mnt/data02/cbye/research")


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def private_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(PRIVATE_ROOT):
        raise ValueError(f"path must be under private scratch {PRIVATE_ROOT}: {path}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("response", "pilot", "ensemble"), required=True
    )
    for name in ("inputs", "output", "gate"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--cpus", required=True, help="logical CPU IDs, e.g. 0,1")
    parser.add_argument("--reserved-other-cpus", required=True, type=float)
    parser.add_argument("--max-seconds", type=int, default=21600)
    parser.add_argument("--stack-mib", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if platform.node().split(".")[0] != "limbo":
        parser.error("this launcher is for the private Limbo scratch only")
    root = private_path(command("git", "rev-parse", "--show-toplevel"))
    inputs = private_path(args.inputs)
    output = private_path(args.output)
    if not inputs.is_dir() or output == inputs or root.is_relative_to(output):
        parser.error("output must be separate from the source and input directories")
    if output.is_relative_to(root) or output.is_relative_to(inputs):
        parser.error("logs/products must stay outside source and input directories")
    head = command("git", "rev-parse", "HEAD")
    if len(args.commit) != 40 or head != args.commit:
        parser.error("checkout differs from the full producer commit pin")
    if command("git", "status", "--porcelain", "--untracked-files=all"):
        parser.error("producer checkout must be clean")
    cpus = [int(value) for value in args.cpus.split(",")]
    available = os.sched_getaffinity(0)
    budget = os.cpu_count() / 3
    if (
        len(cpus) != len(set(cpus))
        or not 1 <= len(cpus) <= 4
        or not set(cpus) <= available
        or args.reserved_other_cpus < 0
        or len(cpus) + args.reserved_other_cpus > budget
        or args.max_seconds <= 0
        or not 8 <= args.stack_mib <= 256
    ):
        parser.error("CPU reservations, runtime or stack exceed the declared limits")
    gate_bytes = args.gate.read_bytes()
    gate = json.loads(gate_bytes)
    if (
        gate.get("producer_commit") != head
        or args.phase not in gate.get("allowed_phases", [])
        or not gate.get("scope")
        or not gate.get("timing_basis")
    ):
        parser.error("gate must pin this commit, allowed phase, scope and timing basis")
    gate_hash = hashlib.sha256(gate_bytes).hexdigest()
    frozen_gate = output / f"gate-{args.phase}.json"
    if frozen_gate.exists() and frozen_gate.read_bytes() != gate_bytes:
        parser.error("existing gate differs; use a new output directory")
    module = "ensemble" if args.phase == "ensemble" else "pilot"
    python = root / ".venv/bin/python"
    invocation = [
        str(python),
        "-u",
        "-m",
        "horizon_position_d5." + module,
        "--inputs",
        str(inputs),
        "--output",
        str(output),
    ]
    if args.phase == "response":
        invocation.append("--response-only")
    if args.phase == "ensemble":
        if args.pilot is None or not (args.pilot / "RESULTS.json").is_file():
            parser.error("ensemble requires a completed --pilot directory")
        invocation.extend(
            [
                "--pilot",
                str(private_path(args.pilot)),
                "--workers",
                str(min(len(cpus), 2)),
            ]
        )
    run_hash = hashlib.sha256(str(output).encode()).hexdigest()[:10]
    unit = f"d5-derivative-run-{run_hash}.service"
    state = command("systemctl", "--user", "show", unit, "-p", "ActiveState", "--value")
    if state in ("active", "activating", "deactivating"):
        parser.error(f"unit already live: {unit} ({state})")
    # Also catch earlier/manual runners that did not use this unit name.
    # Match an entire argv element, not a substring of an SSH shell command.
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            argv = (proc / "cmdline").read_bytes().split(b"\0")
            if os.fsencode(output) in argv:
                parser.error(f"output already owned by live PID {proc.name}")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    done = output / (args.phase.upper() + "_JOB_DONE")
    if done.exists():
        parser.error("stage already completed; inspect its products before rerunning")
    attempt = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    record = {
        "phase": args.phase,
        "producer_commit": head,
        "host": platform.node(),
        "gate_sha256": gate_hash,
        "unit": unit,
        "attempt": attempt,
        "nice": 2,
        "cpus": cpus,
        "cpu_quota_percent": 100 * len(cpus),
        "reserved_other_cbye_cpus": args.reserved_other_cpus,
        "aggregate_cbye_cap_cpus": budget,
        "stack_mib": args.stack_mib,
        "max_seconds": args.max_seconds,
        "command": invocation,
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint_json_files": sorted(
            str(p.relative_to(output)) for p in output.glob("blocks/*.json")
        ),
    }
    print(json.dumps(record, indent=2), flush=True)
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=True)
    frozen_gate.write_bytes(gate_bytes)
    prefix = output / f"{args.phase}-{attempt}"
    job = Path(str(prefix) + ".sh")
    failed = output / (args.phase.upper() + "_JOB_FAILED")
    quote = shlex.quote
    # Retain each attempt's log, time report and exit sentinel. Do not erase
    # failures when resuming: the attempt record explains subsequent success.
    job.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                "exec >> " + quote(str(prefix) + ".log") + " 2>&1",
                "trap 'rc=$?; if [ $rc -eq 0 ]; then touch "
                + quote(str(done))
                + "; else echo $rc > "
                + quote(str(failed))
                + "; fi' EXIT",
                f"ulimit -s {args.stack_mib * 1024}",
                "ulimit -c 0",
                "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
                "export JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1",
                "cd " + quote(str(root)),
                'test "$(git rev-parse HEAD)" = ' + quote(head),
                'test -z "$(git status --porcelain --untracked-files=all)"',
                "/usr/bin/time -v -o "
                + quote(str(prefix) + ".time.txt")
                + " taskset -c "
                + quote(args.cpus)
                + " "
                + shlex.join(invocation),
                "",
            ]
        )
    )
    # A failed transient unit otherwise remains registered and prevents a
    # checkpoint resume under its output-scoped name.
    if state == "failed":
        subprocess.run(["systemctl", "--user", "reset-failed", unit], check=True)
    subprocess.run(
        [
            "systemd-run",
            "--user",
            "--unit=" + unit,
            "--property=Nice=2",
            f"--property=CPUQuota={100 * len(cpus)}%",
            "--property=KillMode=control-group",
            f"--property=RuntimeMaxSec={args.max_seconds}",
            "/bin/bash",
            str(job),
        ],
        check=True,
    )
    record["invocation_id"] = command(
        "systemctl", "--user", "show", unit, "-p", "InvocationID", "--value"
    )
    Path(str(prefix) + ".execution.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print("Launched " + unit + "; InvocationID=" + record["invocation_id"])


if __name__ == "__main__":
    main()
