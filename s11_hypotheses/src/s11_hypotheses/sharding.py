"""Deterministic execution-only partitions and byte-verified collection."""

import json
import shutil
from pathlib import Path

import h5py
from eigsep_analysis import s11_hypothesis_io as intake

from . import checkpoint as ck


def task_name(task):
    return (
        f"facet_{task['configuration']}_{task['epoch']}_"
        f"m{task['member']:03d}_{task['grid']}"
    )


def plan(draws, shards=4, grids=("native", "fine")):
    """Declare all native/fine tasks; primary includes fine member0 per epoch.

    Partition labels never enter physical model parameters or their keys.
    Remaining fine draws are still declared, not silently dropped.
    """
    if not 1 <= shards <= 8:
        raise ValueError("shards must be between1 and8")
    if not grids or set(grids) - {"native", "fine"} or len(grids) != len(set(grids)):
        raise ValueError("invalid explicit grid selection")
    tasks = [dict(p, grid=g) for p in draws for g in grids]
    if len({task_name(p) for p in tasks}) != len(tasks):
        raise ValueError("duplicate physical draw/configuration")
    stages = {"primary": [], "fine": []}
    for t in tasks:
        stage = "primary" if t["grid"] == "native" or t["member"] == 0 else "fine"
        stages[stage].append(t)
    partitions = {
        stage: [[task_name(t) for t in group[i::shards]] for i in range(shards)]
        for stage, group in stages.items()
    }
    record = dict(
        schema="c54-execution-plan-v1",
        tasks=tasks,
        partitions=partitions,
        shards=shards,
    )
    record["plan_sha256"] = ck.fingerprint(record)
    return record


def selected(record, stage, index):
    if stage not in record["partitions"] or not 0 <= index < record["shards"]:
        raise ValueError("unknown execution partition")
    names = set(record["partitions"][stage][index])
    return [t for t in record["tasks"] if task_name(t) in names]


def verify_task(root, task, identity, manifest_sha, choices, geometry_root):
    """Reconstruct identity from frozen scientific parameters and payload."""
    name = task_name(task)
    marker = Path(root) / "units" / name / "DONE.json"
    record = json.loads(marker.read_text())
    params = record["metadata"]["parameters"]
    allowed = set(task) | {"frequency_sha256", "cell_result_sha256"}
    if set(params) != allowed or any(params[k] != v for k, v in task.items()):
        raise ValueError(f"unit parameters differ from plan: {name}")
    key = ck.fingerprint(
        dict(
            source=identity, input_manifest=manifest_sha, choices=choices, params=params
        )
    )
    if not ck.Unit(root, name, key).completed():
        raise ValueError(f"incomplete unit: {name}")
    with h5py.File(marker.parent / "spectra.h5", "r") as h:
        if ck.fingerprint(h["frequency_hz"][:].tolist()) != params["frequency_sha256"]:
            raise ValueError(f"unit coordinates differ: {name}")
    cell = (
        Path(geometry_root)
        / "units"
        / f"cells_{task['configuration']}_{task['epoch']}"
        / "cells.h5"
    )
    draw = {k: v for k, v in task.items() if k != "grid"}
    cell_params = dict(draw, member="geometry")
    cell_key = ck.fingerprint(
        dict(
            source=identity,
            input_manifest=manifest_sha,
            choices=choices,
            params=cell_params,
        )
    )
    cell_name = cell.parent.name
    if not ck.Unit(geometry_root, cell_name, cell_key).completed():
        raise ValueError("incomplete shared geometry")
    if intake.digest(cell) != params["cell_result_sha256"]:
        raise ValueError(f"unit geometry upstream differs: {name}")
    key = ck.fingerprint(
        dict(
            source=identity, input_manifest=manifest_sha, choices=choices, params=params
        )
    )
    if not ck.Unit(root, name, key).completed():
        raise ValueError(f"incomplete unit: {name}")
    return record


def gather(
    record,
    shard_roots,
    output_root,
    identity,
    manifest_sha,
    choices,
    geometry_root,
    stage="primary",
):
    """Copy verified unchanged unit bytes; refuse overlaps or missing tasks."""
    if stage not in ["primary", "all"]:
        raise ValueError("gather stage must be primary or all")
    expected = {
        task_name(t)
        for t in record["tasks"]
        if stage == "all" or t["grid"] == "native" or t["member"] == 0
    }
    task_map = {task_name(t): t for t in record["tasks"]}
    seen, imports = set(), []
    for root in map(Path, shard_roots):
        done = json.loads((root / "SHARD_DONE.json").read_text())
        run_path = root / "RUN_RECORD_physical-shard.json"
        run = json.loads(run_path.read_text())
        saved_plan = json.loads((root / "EXECUTION_PLAN.json").read_text())
        if saved_plan != record or run["source_identity"] != identity:
            raise ValueError("gather run-record source/plan mismatch")
        if (
            run["input_manifest_sha256"] != manifest_sha
            or run["configuration"]["choices"] != choices
        ):
            raise ValueError("gather run-record input/choices mismatch")
        if run["execution_selection"] != dict(
            stage=done["stage"], index=done["index"], plan_sha256=record["plan_sha256"]
        ):
            raise ValueError("gather run-record partition mismatch")
        if done["plan_sha256"] != record["plan_sha256"]:
            raise ValueError("gather execution-plan mismatch")
        if (
            done["source_fingerprint"] != ck.fingerprint(identity)
            or done["manifest_sha256"] != manifest_sha
        ):
            raise ValueError("gather source/input mismatch")
        ids = {task_name(t) for t in selected(record, done["stage"], done["index"])}
        if set(done["units"]) != ids:
            raise ValueError("shard completion does not cover its partition")
        if seen & ids or not ids <= expected:
            raise ValueError("overlapping/out-of-stage gather units")
        present = {p.name for p in (root / "units").iterdir() if p.is_dir()}
        if present != ids:
            raise ValueError("unlisted or missing physical unit directories")
        for name in sorted(ids):
            row = verify_task(
                root, task_map[name], identity, manifest_sha, choices, geometry_root
            )
            src, dst = root / "units" / name, Path(output_root) / "units" / name
            if dst.exists():
                if intake.digest(dst / "DONE.json") != intake.digest(src / "DONE.json"):
                    raise ValueError("gather destination already differs")
            else:
                shutil.copytree(src, dst)
            verify_task(
                output_root,
                task_map[name],
                identity,
                manifest_sha,
                choices,
                geometry_root,
            )
            imports.append(
                dict(
                    unit=name,
                    original_root=str(root),
                    original_done_sha256=intake.digest(src / "DONE.json"),
                    original_run_record_sha256=intake.digest(run_path),
                    key=row["key"],
                    unchanged_results=row["results"],
                )
            )
        seen |= ids
    if seen != expected:
        raise ValueError(f"missing planned units: {sorted(expected - seen)}")
    result = dict(
        schema="c54-gather-v1",
        stage=stage,
        plan_sha256=record["plan_sha256"],
        imports=imports,
        pending_units=sorted(set(task_map) - seen),
        interpretation="verified original unit bytes; no new model evaluation",
    )
    ck.write_json(Path(output_root) / f"GATHER_{stage}.json", result)
    return result
