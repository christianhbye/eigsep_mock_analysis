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


@pytest.mark.parametrize("budget", ["nan", "inf", "-inf", "-1", "6.01"])
def test_reject_nonfinite_or_overallocated_cpu_reservations(launcher, budget):
    args, output, *_ = launcher
    args[-1] = budget
    with pytest.raises(SystemExit):
        launch.main(args + ["--dry-run"])
    assert not output.exists()


def test_allow_exact_half_host_aggregate_reservation(launcher, capsys):
    args, output, *_ = launcher
    args[-1] = "6"
    launch.main(args + ["--dry-run"])
    record = json.loads(capsys.readouterr().out)
    assert record["aggregate_cbye_cap_cpus"] == 8
    assert len(record["cpus"]) + record["reserved_other_cbye_cpus"] == 8
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
    assert "--property=Nice=0" in run_calls[1]
    assert (output / "RESPONSE_JOB_FAILED").read_text().strip() == "7"
    assert not (output / "RESPONSE_JOB_DONE").exists()
    assert list(output.glob("*.exit-code"))[0].read_text().strip() == "7"
    job = next(output.glob("*.sh")).read_text()
    assert "export JAX_ENABLE_COMPILATION_CACHE=false" in job
    assert "unset JAX_COMPILATION_CACHE_DIR" in job
    record = json.loads(next(output.glob("*.execution.json")).read_text())
    assert record["nice"] == 0
    assert record["persistent_jax_cache_enabled"] is False
    assert record["persistent_jax_cache_dir"] is None


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
        "config": {"fuse_orientation": True, "jit_response_adjoint": True},
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


def test_dry_run_rejects_changed_compilation_mode(launcher):
    args, output, root, _, head, _, _ = launcher
    output.mkdir()
    identity = {
        "source_commit": head,
        "source_tree": "b" * 40,
        "lock_sha256": launch.sha256(root / "uv.lock"),
        "input_hashes": {},
        "versions": {},
        "config": {"fuse_orientation": True, "jit_response_adjoint": True},
    }
    (output / "RUN_RECORD.json").write_text(
        json.dumps(
            {"identity": identity, "identity_hash": launch.fingerprint(identity)}
        )
    )
    launch.main(args + ["--dry-run"])
    with pytest.raises(SystemExit):
        launch.main(args + ["--dry-run", "--split-orientation"])
    assert not list(output.glob("*.sh"))


@pytest.mark.parametrize("standalone", [False, True])
@pytest.mark.parametrize(
    "wrapper", [[], [b"uv", b"run"], [b"uv", b"run", b"--no-sync"]]
)
def test_output_ownership_allows_own_uv_parent_but_rejects_competitors(
    tmp_path, monkeypatch, standalone, wrapper
):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    output = tmp_path / "run"
    launcher_args = (
        [launch.os.fsencode(launch.Path(launch.__file__).resolve())]
        if standalone
        else [b"-m", b"horizon_position_d5.launch"]
    )

    def process(pid, parent, entry):
        proc = proc_root / str(pid)
        proc.mkdir()
        (proc / "status").write_text(f"PPid:\t{parent}\n")
        (proc / "cmdline").write_bytes(
            b"\0".join([*wrapper, b"python", *entry, b"--output", str(output).encode()])
        )

    monkeypatch.setattr(launch.os, "getpid", lambda: 30)
    monkeypatch.setattr(launch.os, "getppid", lambda: 20)
    process(30, 20, launcher_args)
    process(20, 1, launcher_args)
    assert list(launch.output_owners(output, proc_root)) == []
    process(40, 1, launcher_args)
    process(50, 1, [b"-m", b"horizon_position_d5.pilot"])
    assert set(launch.output_owners(output, proc_root)) == {40, 50}
    # An ancestor with an actual science command is still an owner.
    (proc_root / "20" / "cmdline").write_bytes(
        b"\0".join(
            [b"python", b"-m", b"horizon_position_d5.pilot", str(output).encode()]
        )
    )
    assert set(launch.output_owners(output, proc_root)) == {20, 40, 50}
    # Matching the basename is not enough to exempt an unrelated script.
    (proc_root / "20" / "cmdline").write_bytes(
        b"\0".join(
            [b"python", str(tmp_path / "launch.py").encode(), str(output).encode()]
        )
    )
    assert set(launch.output_owners(output, proc_root)) == {20, 40, 50}
    # Receiving a launcher path/module as data does not make a science
    # ancestor a launcher, even with a later Python-looking argument.
    for data_args in (
        [b"--launcher", launch.os.fsencode(launch.Path(launch.__file__).resolve())],
        [b"--module", b"horizon_position_d5.launch"],
        [b"--args", b"python", *launcher_args],
    ):
        (proc_root / "20" / "cmdline").write_bytes(
            b"\0".join(
                [*wrapper, b"python", b"driver.py", *data_args, str(output).encode()]
            )
        )
        assert set(launch.output_owners(output, proc_root)) == {20, 40, 50}


def test_persistent_cache_is_private_recorded_and_dry_run_creates_nothing(
    launcher, monkeypatch
):
    args, output, *_ = launcher
    launch.main(args + ["--persistent-jax-cache", "--dry-run"])
    assert not output.exists()
    monkeypatch.setattr(
        launch.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
    )
    launch.main(args + ["--persistent-jax-cache"])
    cache = output / "jax-cache"
    assert cache.is_dir() and cache.stat().st_mode & 0o777 == 0o700
    record = json.loads(next(output.glob("*.execution.json")).read_text())
    assert record["persistent_jax_cache_enabled"] is True
    assert record["persistent_jax_cache_dir"] == str(cache)
    job = next(output.glob("*.sh")).read_text()
    assert "export JAX_ENABLE_COMPILATION_CACHE=true" in job
    assert "export JAX_COMPILATION_CACHE_DIR=" + launch.shlex.quote(str(cache)) in job


@pytest.mark.parametrize("kind", ["link", "broken-link", "group-writable"])
def test_reject_untrusted_persistent_cache(launcher, monkeypatch, kind):
    args, output, root, *_ = launcher
    output.mkdir()
    cache = output / "jax-cache"
    if kind in ("link", "broken-link"):
        cache.symlink_to(
            root if kind == "link" else root / "absent", target_is_directory=True
        )
    else:
        cache.mkdir(mode=0o777)
        cache.chmod(0o777)
    monkeypatch.setattr(launch.os, "getuid", lambda: cache.stat().st_uid)
    with pytest.raises(SystemExit):
        launch.main(args + ["--persistent-jax-cache", "--dry-run"])
    assert not list(output.glob("*.sh"))


def ensemble_args(launcher):
    args, output, root, *_ = launcher
    args[1] = "ensemble"
    args[args.index("--cpus") + 1] = "0,1,2,3"
    args[args.index("--reserved-other-cpus") + 1] = "1"
    gate_path = launch.Path(args[args.index("--gate") + 1])
    gate = json.loads(gate_path.read_text())
    gate["allowed_phases"].append("ensemble")
    gate_path.write_text(json.dumps(gate))
    pilot = root.parent / "pilot"
    pilot.mkdir()
    (pilot / "RESULTS.json").write_text("{}")
    return args + ["--pilot", str(pilot)], output


@pytest.mark.parametrize("workers", [None, 1, 4])
def test_ensemble_worker_allocation_preserves_affinity_budget(
    launcher, capsys, workers
):
    args, output = ensemble_args(launcher)
    if workers is not None:
        args += ["--workers", str(workers)]
    launch.main(args + ["--dry-run"])
    record = json.loads(capsys.readouterr().out)
    count = 2 if workers is None else workers
    assert record["workers"] == count
    assert record["command"][-2:] == ["--workers", str(count)]
    assert record["cpus"] == [0, 1, 2, 3]
    assert record["aggregate_cbye_cap_cpus"] == 8
    assert len(record["cpus"]) + record["reserved_other_cbye_cpus"] <= 8
    assert not output.exists()


@pytest.mark.parametrize("workers", [0, -1, 3, 5])
def test_reject_workers_exceeding_cpu_allocation_before_writes(launcher, workers):
    args, output = ensemble_args(launcher)
    args[args.index("--cpus") + 1] = "0,1"
    with pytest.raises(SystemExit):
        launch.main(args + ["--workers", str(workers)])
    assert not output.exists()


def test_reject_worker_option_for_nonensemble_stage(launcher):
    args, output, *_ = launcher
    with pytest.raises(SystemExit):
        launch.main(args + ["--workers", "1"])
    assert not output.exists()


def test_reject_ensemble_resume_with_changed_worker_identity(launcher):
    args, output = ensemble_args(launcher)
    _, _, root, _, head, *_ = launcher
    output.mkdir()
    identity = {
        "source_commit": head,
        "source_tree": "b" * 40,
        "lock_sha256": launch.sha256(root / "uv.lock"),
        "input_hashes": {},
        "versions": {},
        "config": {"workers": 2},
    }
    (output / "RUN_RECORD.json").write_text(
        json.dumps(
            {"identity": identity, "identity_hash": launch.fingerprint(identity)}
        )
    )
    launch.main(args + ["--dry-run"])
    with pytest.raises(SystemExit):
        launch.main(args + ["--workers", "4", "--dry-run"])
    assert not list(output.glob("*.sh"))


def reference_args(launcher, monkeypatch, phase="reference-probe"):
    args, output, root, inputs, head, calls, state = launcher
    args = list(args)
    args[1] = phase
    plan = inputs.parent / "plan.json"
    plan.write_text('{"schema_version": 1}')
    pilot, ensemble = inputs.parent / "pilot", inputs.parent / "ensemble"
    pilot.mkdir()
    ensemble.mkdir()
    original = launch.command

    def command(*values):
        if "horizon_position_d5.reference_convergence" in values:
            return json.dumps(
                {
                    "inventory_hash": "i" * 64,
                    "plan_hash": "p" * 64,
                    "source": {"source_dirty": False},
                }
            )
        return original(*values)

    monkeypatch.setattr(launch, "command", command)
    gate_path = inputs.parent / "gate.json"
    gate = {
        "producer_commit": head,
        "allowed_phases": [phase],
        "scope": "synthetic reference",
        "timing_basis": "measured tiny control",
        "plan_sha256": launch.sha256(plan),
        "inventory_hash": "i" * 64,
        "workers": 2,
        "allocated_cpus": 2,
        "max_seconds": 21600,
        "memory_high_bytes": 100000000,
        "memory_max_bytes": 200000000,
    }
    gate_path.write_text(json.dumps(gate))
    args += ["--plan", str(plan), "--pilot", str(pilot), "--ensemble", str(ensemble)]
    return args, output, gate_path, plan


def test_reference_dry_run_and_detached_memory_plan(launcher, monkeypatch, capsys):
    args, output, gate_path, plan = reference_args(launcher, monkeypatch)
    launch.main(args + ["--dry-run"])
    record = json.loads(capsys.readouterr().out)
    assert "--execute" in record["command"] and "--probe" in record["command"]
    assert not output.exists()
    calls = []
    monkeypatch.setattr(
        launch.subprocess, "run", lambda args, **kwargs: calls.append(args)
    )
    launch.main(args)
    assert (output / "convergence-plan.json").read_bytes() == plan.read_bytes()
    assert (output / "gate-reference-probe.json").read_bytes() == gate_path.read_bytes()
    systemd = next(c for c in calls if c[0] == "systemd-run")
    assert "--property=MemoryHigh=100000000" in systemd
    assert "--property=MemoryMax=200000000" in systemd
    assert "--property=Nice=0" in systemd


@pytest.mark.parametrize(
    "change",
    [
        {"plan_sha256": "0" * 64},
        {"inventory_hash": "0" * 64},
        {"workers": 3},
        {"allocated_cpus": 3},
        {"max_seconds": 100},
        {"memory_high_bytes": -1},
        {"memory_max_bytes": 1},
    ],
)
def test_reference_bad_gate_rejected_before_writes(launcher, monkeypatch, change):
    args, output, gate_path, _ = reference_args(launcher, monkeypatch)
    gate = json.loads(gate_path.read_text())
    gate.update(change)
    gate_path.write_text(json.dumps(gate))
    with pytest.raises(SystemExit):
        launch.main(args)
    assert not output.exists()


def test_reference_full_run_has_distinct_completion_marker(
    launcher, monkeypatch, capsys
):
    args, output, _, _ = reference_args(launcher, monkeypatch, "reference-convergence")
    output.mkdir()
    (output / "REFERENCE-PROBE_JOB_DONE").touch()
    launch.main(args + ["--dry-run"])
    record = json.loads(capsys.readouterr().out)
    assert "--probe" not in record["command"]
    assert not (output / "RUN_RECORD.json").exists()


def test_reference_resume_rejects_changed_plan_identity(launcher, monkeypatch):
    args, output, _, _ = reference_args(launcher, monkeypatch)
    _, _, root, _, head, *_ = launcher
    output.mkdir()
    identity = {
        "source_commit": head,
        "source_tree": "b" * 40,
        "lock_sha256": launch.sha256(root / "uv.lock"),
        "input_hashes": {},
        "versions": {},
        "config": {"workers": 2, "plan_hash": "old", "inventory_hash": "i" * 64},
    }
    (output / "RUN_RECORD.json").write_text(
        json.dumps(
            {"identity": identity, "identity_hash": launch.fingerprint(identity)}
        )
    )
    with pytest.raises(SystemExit):
        launch.main(args + ["--dry-run"])
