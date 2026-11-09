"""Actual-frequency numerical gate for fixed-frame reuse, never a sky fit."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import assets
from .field_operands import FieldOperands
from .model_context import load
from .shared_frame import SharedFrame
from .streaming import reduce_columns, simulate_setup

CHUNKS = [[268, 604, 980], [269, 605, 981]]


def prepare(config, out):
    """Freeze ordinary-reference beam operands/configs before running them."""
    out.mkdir(parents=True, exist_ok=False)
    fields = FieldOperands(config["field_operands"])
    records = []
    for role in ["raster_hybrid", "represented_hfss_control"]:
        f = np.array(CHUNKS[1]) * 250 / 1024
        bm, norm = fields.beam(role, f)
        path = out / (role + "_beam.npz")
        assets.atomic_npz(
            path,
            freqs=f * 1e6,
            bm=bm,
            lmax=128,
            role=role,
            parent_field_operands=config["field_operands"]["sha256"],
        )
        assets.atomic_npz(out / (role + "_norm.npz"), **norm)
        cfg = {
            **config["model"],
            "channels": CHUNKS[1],
            "beam": dict(path=str(path), sha256=assets.sha256(path), role=role),
            "frequency_operator": "native",
        }
        target = out / ("ordinary_" + role + ".json")
        assets.write_json(target, cfg)
        records.append(
            dict(
                chunk=1,
                role=role,
                inputs=str(target),
                inputs_sha256=assets.sha256(target),
            )
        )
    cfg = {
        **config["model"],
        "channels": CHUNKS[0],
        "beam": config["raw_beam"],
        "frequency_operator": "native",
    }
    target = out / "ordinary_full_native_hfss_control.json"
    assets.write_json(target, cfg)
    records.append(
        dict(
            chunk=0,
            role="full_native_hfss_control",
            inputs=str(target),
            inputs_sha256=assets.sha256(target),
        )
    )
    assets.write_json(
        out / "RUN_DONE.json",
        dict(records=records, field_operands_sha256=config["field_operands"]["sha256"]),
    )


def compare(reference, values, means, counts, ctx, channels, role, beam, raw_pin):
    path = assets.checked_file(reference)
    done = json.loads(path.read_text())
    source = path.parent / "columns.npz"
    if assets.sha256(source) != done["columns_sha256"]:
        raise ValueError("ordinary reference payload changed")
    model = done["input_config"]
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
        if model[key] != ctx["configuration"][key]:
            raise ValueError("ordinary reference physical identity differs: " + key)
    if done["model_qc"]["first_unchanged_operator_max_abs_K"] > 1e-9:
        raise ValueError("ordinary reference failed its existing operator closure")
    if model["channels"] != channels or model["frequency_operator"] != "native":
        raise ValueError("ordinary reference frequency semantics differ")
    if role == "full_native_hfss_control":
        if model["beam"] != raw_pin:
            raise ValueError("ordinary full-HFSS beam identity changed")
    else:
        if model["beam"]["role"] != role:
            raise ValueError("ordinary field beam role changed")
        beam_path = assets.checked_file(model["beam"])
        with np.load(beam_path, allow_pickle=False) as z:
            frequencies = z["freqs"] / 1e6
            wanted = np.asarray(channels) * 250 / 1024
            idx = np.searchsorted(frequencies, wanted)
            if np.any(idx >= len(frequencies)) or not np.array_equal(
                frequencies[idx], wanted
            ):
                raise ValueError("ordinary field beam frequencies changed")
            if str(z["role"]) != role:
                raise ValueError("ordinary exported field role changed")
            np.testing.assert_allclose(z["bm"][idx], beam, rtol=1e-12, atol=1e-12)
    with np.load(source, allow_pickle=False) as z:
        for key in ["source_file", "row", "input", "time_best"]:
            np.testing.assert_array_equal(z[key], ctx["cells"][key])
        np.testing.assert_array_equal(z["fine_bin_id"], ctx["mapping"])
        np.testing.assert_array_equal(z["channels"], channels)
        np.testing.assert_array_equal(z["bin_count"], counts)
        a = values[..., 0] + ctx["cmb"] * (1 - values[..., 1])
        ba = means[..., 0] + ctx["cmb"] * (1 - means[..., 1])
        ref_values = np.stack([z["native_S"], z["native_fg"]], axis=-1)
        ref_means, ref_counts = reduce_columns(
            ref_values, z["accepted"], ctx["mapping"], ctx["n_bins"]
        )
        np.testing.assert_array_equal(ref_counts, counts)
        maxima = {}
        for name, new, old, tol in [
            ("native_S", values[..., 0], z["native_S"], 1e-9),
            ("native_A", a, z["native_A"], 1e-9),
            ("native_fg", values[..., 1], z["native_fg"], 1e-12),
            ("bin_A", ba, z["bin_A"], 1e-9),
            ("bin_S", means[..., 0], ref_means[..., 0], 1e-9),
            ("bin_fg", means[..., 1], ref_means[..., 1], 1e-12),
        ]:
            np.testing.assert_allclose(new, old, rtol=0, atol=tol, equal_nan=True)
            maxima[name] = float(np.nanmax(abs(new - old)))
        index = np.searchsorted(ctx["cells"]["chans"], channels)
        np.testing.assert_array_equal(z["accepted"], ctx["cells"]["accepted"][:, index])
    return maxima


def run(config, out):
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    ctx = load(config["model"])
    fields = FieldOperands(config["field_operands"])
    frequencies = np.unique(np.array(CHUNKS).ravel()) * 250 / 1024
    maps, _ = assets.gsm_maps(config["model"]["gsm"], frequencies, out / "gsm")
    import croissant as cro

    def sky_at(f):
        return cro.Sky(
            maps[np.searchsorted(frequencies, f)],
            f,
            sampling="healpix",
            coord="galactic",
            niter=3,
        )

    f0 = np.array(CHUNKS[0]) * 250 / 1024
    sky0 = sky_at(f0)
    initial, _ = fields.beam("raster_hybrid", f0)
    times = ctx["cells"]["time_best"]
    point = ctx["pointing"]
    frame = SharedFrame(
        initial, f0, sky0, times, ctx["horizon"], ctx["site"], ctx["identity"]
    )
    common = dict(
        times=times, horizon=ctx["horizon"], site=ctx["site"], context=ctx["identity"]
    )
    records = []
    for chunk, channels in enumerate(CHUNKS):
        f = np.array(channels) * 250 / 1024
        sky = sky_at(f)
        if chunk:
            t = time.monotonic()
            frame.sky_alm(sky, f, **common)
            sky_seconds = time.monotonic() - t
        else:
            sky_seconds = 0.0
        roles = ["raster_hybrid", "represented_hfss_control"]
        if chunk == 0:
            roles += ["full_native_hfss_control"]
        for role in roles:
            t = time.monotonic()
            if role == "full_native_hfss_control":
                bm, _, _ = assets.scalar_beam(config["raw_beam"], f)
            else:
                bm, _ = fields.beam(role, f)
            setup, norm = frame.replace(bm, f, sky, **common)
            prepared_seconds = time.monotonic() - t
            target = out / (str(chunk) + "_" + role)
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
            indices = np.searchsorted(ctx["cells"]["chans"], channels)
            means, counts = reduce_columns(
                values,
                ctx["cells"]["accepted"][:, indices],
                ctx["mapping"],
                ctx["n_bins"],
            )
            key = str(chunk) + "_" + role
            maxima = compare(
                config["references"][key],
                values,
                means,
                counts,
                ctx,
                channels,
                role,
                bm,
                config["raw_beam"],
            )
            assets.atomic_npz(
                target / "columns.npz",
                frequencies_mhz=f,
                columns=values,
                bin_columns=means,
                bin_count=counts,
                **norm,
            )
            records.append(
                dict(
                    chunk=chunk,
                    role=role,
                    maxima=maxima,
                    qc=qc,
                    prepare_seconds=prepared_seconds,
                    sky_alm_seconds=sky_seconds,
                    columns_sha256=assets.sha256(target / "columns.npz"),
                )
            )
            assets.write_json(out / "CONFORMANCE_PROGRESS.json", dict(cases=records))
    assets.write_json(
        out / "RUN_DONE.json",
        dict(
            label="diagnostic numerical conformance",
            passed=True,
            cases=records,
            frame_signature=frame.signature,
            source_commit=ctx["source_commit"],
            elapsed_s=time.monotonic() - start,
            absolute_temperature_gate_K=1e-9,
            absolute_ground_fraction_gate=1e-12,
            no_sky_fit=True,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "compare"])
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    (prepare if a.stage == "prepare" else run)(json.loads(a.inputs.read_text()), a.out)
