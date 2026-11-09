"""Checkpointed C57 native/centre columns using one qualified fixed frame.

Physical frequencies remain explicit. Native measured masks are owned by
one immutable calibration stage, never replaced by evaluated model points.
"""

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import assets
from .field_operands import FieldOperands
from .model_context import load
from .shared_frame import SharedFrame
from .streaming import simulate_setup


def tasks(native, centres, roles, phases):
    """Freeze the exact three-frequency schedule before any model call."""
    specs = {
        "primary_native": (
            "native",
            native,
            ["raster_hybrid", "full_native_hfss_control"],
        ),
        "centre_control": (
            "four-channel-centres",
            centres,
            ["raster_hybrid", "full_native_hfss_control"],
        ),
        "secondary_native": (
            "native",
            native,
            [r for r in roles if r != "raster_hybrid"],
        ),
    }
    if (
        not phases
        or len(set(phases)) != len(phases)
        or any(p not in specs for p in phases)
    ):
        raise ValueError("explicit unique model phases required")
    result = []
    for phase in phases:
        mode, frequencies, selected = specs[phase]
        if len(frequencies) % 3:
            raise ValueError("fixed exact three-frequency chunks required")
        for first in range(0, len(frequencies), 3):
            for role in selected:
                result.append(
                    dict(
                        phase=phase,
                        frequency_operator=mode,
                        role=role,
                        evaluation_index=list(range(first, first + 3)),
                        frequencies_mhz=np.asarray(
                            frequencies[first : first + 3]
                        ).tolist(),
                    )
                )
    return result


def checkpoint(target, expected, source_manifest):
    """Reuse only a complete exact-source/physical-context checkpoint."""
    path = target / "RUN_DONE.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    for key, value in {**expected, "source_snapshot": source_manifest}.items():
        if record[key] != value:
            raise ValueError("completed model checkpoint differs: " + key)
    payload = target / "columns.npz"
    if assets.sha256(payload) != record["columns_sha256"]:
        raise ValueError("completed model payload changed")
    with np.load(payload, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["frequencies_mhz"], expected["frequencies_mhz"])
        for key in ["native_S", "native_fg", "native_A"]:
            if (
                z[key].shape != (expected["native_rows"], 3)
                or not np.isfinite(z[key]).all()
            ):
                raise ValueError("completed model coverage changed")
    return record


def preserve_incomplete(target):
    """Move an incomplete target aside; never overwrite partial evidence."""
    if target.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        shutil.move(
            str(target),
            str(target.with_name(target.name + ".incomplete-" + stamp)),
        )


def run(config, out):
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    conformance = json.loads(assets.checked_file(config["conformance"]).read_text())
    if not conformance["passed"] or len(conformance["cases"]) != 5:
        raise ValueError("full fixed-frame numerical conformance required")
    workflow = json.loads(
        assets.checked_file(config["conformance_workflow"]).read_text()
    )
    qualification = json.loads(
        assets.checked_file(config["conformance_inputs"]).read_text()
    )
    if (
        not workflow["source_runtime_equal_before_after"]
        or workflow["conformance_results_sha256"] != config["conformance"]["sha256"]
        or workflow["final_inputs_sha256"] != config["conformance_inputs"]["sha256"]
        or qualification["field_operands"] != config["field_operands"]
        or qualification["raw_beam"] != config["raw_beam"]
    ):
        raise ValueError("qualified workflow/field/beam binding differs")
    for key in [
        "native_stage",
        "pointing_stage",
        "iers",
        "horizon",
        "gsm",
        "nominal034",
        "provenance034",
        "runtime_versions",
        "runtime_modules",
    ]:
        if qualification["model"][key] != config["model"][key]:
            raise ValueError("qualified physical context differs: " + key)
    qualified_source = json.loads(
        assets.checked_file(qualification["model"]["source_snapshot"]).read_text()
    )
    actual_source = json.loads(
        assets.checked_file(config["model"]["source_snapshot"]).read_text()
    )
    for name, digest in qualified_source["files_sha256"].items():
        if actual_source["files_sha256"].get(name) != digest:
            raise ValueError("qualified implementation changed: " + name)
    ctx = load(config["model"])
    fields = FieldOperands(config["field_operands"])
    cells, point = ctx["cells"], ctx["pointing"]
    np.testing.assert_array_equal(cells["chans"], np.arange(264, 984))
    times = cells["time_best"]
    context = dict(
        physical=ctx["identity"],
        field_operands=config["field_operands"],
        raw_beam=config["raw_beam"],
        source_commit=ctx["source_commit"],
        source_snapshot=config["model"]["source_snapshot"],
        conformance=config["conformance"],
        conformance_workflow=config["conformance_workflow"],
        conformance_inputs=config["conformance_inputs"],
        roles=fields.roles,
        phases=config["phases"],
    )
    context_bytes = json.dumps(context, sort_keys=True, allow_nan=False).encode()
    context_sha = hashlib.sha256(context_bytes).hexdigest()
    context_path = out / "BATCH_CONTEXT.json"
    if context_path.exists() and json.loads(context_path.read_text()) != context:
        raise ValueError("resumed batch physical/source context changed")
    assets.write_json(context_path, context)
    native = cells["chans"] * 250 / 1024
    centres = cells["chans"].reshape(-1, 4).mean(1) * 250 / 1024
    schedule = tasks(native, centres, fields.roles, config["phases"])
    limit = config["max_new_evaluations"]
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("positive bounded evaluation count required")
    assets.write_json(
        out / "SCHEDULE.json",
        dict(tasks=schedule, source_snapshot=config["model"]["source_snapshot"]),
    )
    records = []
    source_manifest = config["model"]["source_snapshot"]
    pending = []

    def target_for(task):
        return (
            out
            / task["phase"]
            / f"chunk_{task['evaluation_index'][0] // 3:04}"
            / task["role"]
        )

    def receipt(task):
        path = target_for(task) / "RUN_DONE.json"
        return dict(path=str(path), sha256=assets.sha256(path), **task)

    for task in schedule:
        expected = dict(**task, context_sha256=context_sha, native_rows=len(times))
        if checkpoint(target_for(task), expected, source_manifest) is None:
            pending.append(task)
        else:
            records.append(receipt(task))
    selected = pending[:limit]
    attempt = out / "attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    attempt.mkdir(parents=True)
    if selected:
        bootstrap = np.array([268, 604, 980]) * 250 / 1024
        needed = np.unique(
            np.concatenate([bootstrap, *[t["frequencies_mhz"] for t in selected]])
        )
        maps, gsm_checks = assets.gsm_maps(
            config["model"]["gsm"], needed, attempt / "gsm"
        )
        raw_bm, _, raw_qc = assets.scalar_beam(config["raw_beam"], needed)
        assets.atomic_npz(attempt / "raw_interpolation_qc.npz", **raw_qc)
        assets.write_json(attempt / "GSM_CHECKS.json", gsm_checks)
        import croissant as cro

        def sky_at(f):
            i = np.searchsorted(needed, f)
            np.testing.assert_array_equal(needed[i], f)
            return cro.Sky(maps[i], f, sampling="healpix", coord="galactic", niter=3)

        sky0 = sky_at(bootstrap)
        initial, _ = fields.beam("raster_hybrid", bootstrap)
        frame = SharedFrame(
            initial,
            bootstrap,
            sky0,
            times,
            ctx["horizon"],
            ctx["site"],
            ctx["identity"],
        )
        if frame.signature != conformance["frame_signature"]:
            raise ValueError("actual simulation frame differs from qualified frame")
        common = dict(
            times=times,
            horizon=ctx["horizon"],
            site=ctx["site"],
            context=ctx["identity"],
        )
        for task in selected:
            f = np.array(task["frequencies_mhz"])
            sky = sky_at(f)
            t = time.monotonic()
            frame.sky_alm(sky, f, **common)
            sky_s = time.monotonic() - t
            target = target_for(task)
            preserve_incomplete(target)
            t = time.monotonic()
            norm = {}
            if task["role"] == "full_native_hfss_control":
                bm = raw_bm[np.searchsorted(needed, f)]
            else:
                bm, norm = fields.beam(task["role"], f)
            setup, rescale = frame.replace(bm, f, sky, **common)
            prepare_s = time.monotonic() - t
            values, qc = simulate_setup(
                setup,
                f,
                times,
                point["el_drive_deg"],
                point["az_drive_deg"],
                point["rotations"],
                point["misalignment"],
                target,
                save_chunks=False,
            )
            assets.atomic_npz(
                target / "columns.npz",
                frequencies_mhz=f,
                native_S=values[..., 0],
                native_fg=values[..., 1],
                native_A=values[..., 0] + ctx["cmb"] * (1 - values[..., 1]),
                **{"field_" + key: value for key, value in norm.items()},
                **{"frame_" + key: value for key, value in rescale.items()},
            )
            part = dict(
                label="diagnostic model columns; no new observations",
                **task,
                native_rows=len(times),
                context_sha256=context_sha,
                source_snapshot=source_manifest,
                columns_sha256=assets.sha256(target / "columns.npz"),
                sky_alm_s=sky_s,
                prepare_s=prepare_s,
                qc=qc,
            )
            assets.write_json(target / "RUN_DONE.json", part)
            records.append(receipt(task))
            assets.write_json(
                out / "BATCH_PROGRESS.json",
                dict(
                    completed=records,
                    scheduled=len(schedule),
                    elapsed_s=time.monotonic() - start,
                ),
            )
    complete = len(records) == len(schedule)
    for phase in config["phases"]:
        phase_records = [r for r in records if r["phase"] == phase]
        if len(phase_records) == sum(t["phase"] == phase for t in schedule):
            assets.write_json(
                out / (phase.upper() + "_DONE.json"),
                dict(
                    status="COMPLETE model phase",
                    label="diagnostic",
                    context_sha256=context_sha,
                    records=phase_records,
                    native_stage=config["model"]["native_stage"],
                    pointing_stage=config["model"]["pointing_stage"],
                    source_snapshot=source_manifest,
                ),
            )
    result = dict(
        label="diagnostic model batch",
        status="COMPLETE" if complete else "BOUNDED_PARTIAL",
        context=context,
        context_sha256=context_sha,
        records=records,
        newly_completed=len(selected),
        remaining=len(schedule) - len(records),
        elapsed_s=time.monotonic() - start,
        no_sky_fit=True,
    )
    assets.write_json(attempt / "RUN_DONE.json", result)
    assets.write_json(
        out / ("RUN_DONE.json" if complete else "BATCH_PARTIAL.json"), result
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    run(json.loads(a.inputs.read_text()), a.out)
