"""Small operational controls: no SSH, systemd jobs or science computation."""

import fcntl
import json
import subprocess

import pytest

from horizon_position_d5 import launch


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "INPUT_MANIFEST.json").write_text("{}")
    (root / "uv.lock").write_text("frozen lock")
    output = tmp_path / "output with spaces"
    commit = "a" * 40
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "producer_commit": commit,
                "allowed_phases": ["response", "pilot"],
                "scope": "synthetic",
                "timing_basis": "test",
            }
        )
    )
    calls = []
    state = {"value": "inactive"}

    def command(*args):
        calls.append(args)
        if args == ("git", "rev-parse", "--show-toplevel"):
            return str(root)
        if args == ("git", "rev-parse", "HEAD"):
            return commit
        if args == ("git", "rev-parse", "HEAD^{tree}"):
            return "b" * 40
        if args[0] == "git":
            return ""
        if "InvocationID" in args:
            return "synthetic-invocation"
        return state["value"]

    monkeypatch.setattr(launch, "PRIVATE_ROOT", tmp_path)
    monkeypatch.setattr(launch, "command", command)
    monkeypatch.setattr(launch.platform, "node", lambda: "limbo")
    monkeypatch.setattr(launch.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(launch.os, "sched_getaffinity", lambda _: {0, 1, 2, 3})
    monkeypatch.setattr(launch.os, "getuid", lambda: -1)
    args = [
        "--phase",
        "response",
        "--inputs",
        str(inputs),
        "--output",
        str(output),
        "--gate",
        str(gate),
        "--commit",
        commit,
        "--cpus",
        "0,1",
        "--reserved-other-cpus",
        "3",
    ]
    return args, output, root, inputs, commit, calls, state


@pytest.mark.parametrize("budget", ["nan", "inf", "-inf", "-1", "3.34"])
def test_reject_nonfinite_or_overallocated_cpu_reservations(launcher, budget):
    args, output, *_ = launcher
    args[-1] = budget
    with pytest.raises(SystemExit):
        launch.main(args + ["--dry-run"])
    assert not output.exists()


def test_dry_run_writes_nothing_and_unit_owns_all_phases(launcher):
    args, output, _, _, _, calls, state = launcher
    launch.main(args + ["--dry-run"])
    first = [c for c in calls if c[0] == "systemctl"][-1]
    args[1] = "pilot"
    state["value"] = "active"
    with pytest.raises(SystemExit):
        launch.main(args + ["--dry-run"])
    second = [c for c in calls if c[0] == "systemctl"][-1]
    assert first[3] == second[3]
    assert not output.exists()


def test_concurrent_launch_lock_refuses_second_writer(launcher):
    args, output, *_ = launcher
    output.mkdir()
    with (output / "LAUNCH.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit):
            launch.main(args)
    assert not list(output.glob("*.sh"))


def test_failed_resume_resets_only_own_unit_and_quotes_sentinel(launcher, monkeypatch):
    args, output, _, _, _, _, state = launcher
    state["value"] = "failed"
    run_calls = []
    original_run = subprocess.run

    def run(command, **kwargs):
        run_calls.append(command)
        if command[0] == "systemd-run":
            job = command[-1]
            # Execute only the exact generated redirection/trap preamble,
            # followed by a controlled failure; never run the science command.
            with open(job) as stream:
                preamble = stream.read().split("ulimit -s", 1)[0]
            result = original_run(["bash"], input=preamble + "exit 7\n", text=True)
            assert result.returncode == 7
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launch.subprocess, "run", run)
    launch.main(args)
    assert run_calls[0][:3] == ["systemctl", "--user", "reset-failed"]
    assert run_calls[0][3] == run_calls[1][2].split("=", 1)[1]
    assert (output / "RESPONSE_JOB_FAILED").read_text().strip() == "7"
    assert not (output / "RESPONSE_JOB_DONE").exists()
    assert list(output.glob("*.exit-code"))[0].read_text().strip() == "7"


def test_checkpoint_audit_checks_file_and_producer_identity(launcher):
    _, output, root, inputs, head, _, _ = launcher
    blocks = output / "blocks"
    blocks.mkdir(parents=True)
    identity = {
        "source_commit": head,
        "source_tree": "b" * 40,
        "lock_sha256": launch.sha256(root / "uv.lock"),
        "input_hashes": {},
        "versions": {},
    }
    record = {"identity": identity, "identity_hash": launch.fingerprint(identity)}
    (output / "RUN_RECORD.json").write_text(json.dumps(record))
    data = blocks / "tiny.npz"
    data.write_bytes(b"metadata-only audit does not load arrays")
    meta = {
        **record,
        "complete": True,
        "parameters": {"value": 1},
        "parameters_hash": launch.fingerprint({"value": 1}),
        "sha256": launch.sha256(data),
    }
    (blocks / "tiny.json").write_text(json.dumps(meta))
    audit = launch.audit_checkpoints(output, root, inputs, head)
    assert audit["run_identity_compatible"]
    assert audit["blocks"][0]["status"].startswith("integrity-valid")
    data.write_bytes(b"corrupt")
    assert (
        launch.audit_checkpoints(output, root, inputs, head)["blocks"][0]["status"]
        == "data hash mismatch"
    )
    assert not launch.audit_checkpoints(output, root, inputs, "c" * 40)[
        "run_identity_compatible"
    ]
