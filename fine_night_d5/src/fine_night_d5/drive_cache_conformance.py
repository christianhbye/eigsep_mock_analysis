"""Prospective exact full-D reuse qualification against saved native oracles."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import assets
from .drive_operands import DriveOperands, build, drive_eulers
from .field_operands import FieldOperands
from .model_context import load
from .shared_conformance import CHUNKS, compare
from .shared_frame import SharedFrame
from .streaming import reduce_columns, simulate_setup


def compare_secondary(
    reference, values, means, counts, ctx, channels, role, beam, raw_pin
):
    """Compare all measured/native axes and count-reduced physical columns."""
    path = assets.checked_file(reference)
    record = json.loads(path.read_text())
    context_path = path.parents[3] / "BATCH_CONTEXT.json"
    context = json.loads(context_path.read_text())
    digest = hashlib.sha256(
        json.dumps(context, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if digest != record["context_sha256"]:
        raise ValueError("secondary reference context changed")
    if (
        context["physical"] != ctx["identity"]
        or context["field_operands"] != ctx["field_operands"]
        or context["raw_beam"] != raw_pin
        or record["source_snapshot"] != context["source_snapshot"]
        or record["role"] != role
        or record["frequency_operator"] != "native"
    ):
        raise ValueError("secondary oracle physical/role context changed")
    np.testing.assert_array_equal(
        record["frequencies_mhz"], np.array(channels) * 250 / 1024
    )
    if record["qc"]["first_unchanged_operator_max_abs_K"] > 1e-9:
        raise ValueError("secondary oracle original closure failed")
    payload = path.parent / "columns.npz"
    if assets.sha256(payload) != record["columns_sha256"]:
        raise ValueError("secondary reference payload changed")
    with np.load(payload, allow_pickle=False) as z:
        np.testing.assert_array_equal(
            z["frequencies_mhz"], np.array(channels) * 250 / 1024
        )
        if z["native_S"].shape != (len(ctx["cells"]["time_best"]), 3):
            raise ValueError("secondary reference native row coverage changed")
        original = np.stack([z["native_S"], z["native_fg"]], axis=-1)
        indices = np.searchsorted(ctx["cells"]["chans"], channels)
        ref_means, ref_counts = reduce_columns(
            original,
            ctx["cells"]["accepted"][:, indices],
            ctx["mapping"],
            ctx["n_bins"],
        )
        np.testing.assert_array_equal(ref_counts, counts)
        maxima = {}
        for name, new, old, tol in [
            ("native_S", values[..., 0], z["native_S"], 1e-9),
            (
                "native_A",
                values[..., 0] + ctx["cmb"] * (1 - values[..., 1]),
                z["native_A"],
                1e-9,
            ),
            ("native_fg", values[..., 1], z["native_fg"], 1e-12),
            ("bin_S", means[..., 0], ref_means[..., 0], 1e-9),
            ("bin_fg", means[..., 1], ref_means[..., 1], 1e-12),
            (
                "bin_A",
                means[..., 0] + ctx["cmb"] * (1 - means[..., 1]),
                ref_means[..., 0] + ctx["cmb"] * (1 - ref_means[..., 1]),
                1e-9,
            ),
        ]:
            np.testing.assert_allclose(new, old, rtol=0, atol=tol, equal_nan=True)
            maxima[name] = float(np.nanmax(abs(new - old)))
    return maxima


def run(config, out):
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    ctx = load(config["model"])
    ctx["field_operands"] = config["field_operands"]
    fields = FieldOperands(config["field_operands"])
    chunks = CHUNKS + [[264, 265, 266]]
    frequencies = np.unique(np.array(chunks).ravel()) * 250 / 1024
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
    eulers = drive_eulers(point)
    table_start = time.monotonic()
    pin = build(eulers, ctx["identity"], out / "drive_tables")
    operands = DriveOperands(pin, eulers, ctx["identity"])
    table_seconds = time.monotonic() - table_start
    orient = operands.orientation(frame.template.beam_norm, frame.template.sim.lmax + 1)
    records = []
    for chunk, channels in enumerate(chunks):
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
        if chunk == 2:
            roles = list(config["secondary_references"])
        for role in roles:
            t = time.monotonic()
            if role == "full_native_hfss_control":
                bm, _, _ = assets.scalar_beam(config["raw_beam"], f)
            else:
                bm, _ = fields.beam(role, f)
            setup, norm = frame.replace(bm, f, sky, **common)
            setup = setup._replace(orient=orient)
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
            comparator = compare_secondary if chunk == 2 else compare
            reference = (
                config["secondary_references"][role]
                if chunk == 2
                else config["references"][key]
            )
            maxima = comparator(
                reference,
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
            drive_operands=pin,
            drive_table_seconds=table_seconds,
            orientation_count=len(eulers),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    run(json.loads(a.inputs.read_text()), a.out)
