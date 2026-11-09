"""Explicit completed-qualification gate for optional full-resolution operands."""

import json
import time

import numpy as np

from . import assets
from .drive_operands import DriveOperands, drive_eulers
from .sht_operands import SHTOperands
from .streaming import orientation_groups

PHYSICS = [
    "native_stage",
    "pointing_stage",
    "iers",
    "horizon",
    "gsm",
    "nominal034",
    "provenance034",
    "runtime_versions",
    "runtime_modules",
]


def checked(pin):
    path = assets.checked_file(pin)
    return path, json.loads(path.read_text())


def ten_cases(cases):
    roles = [
        "represented_hfss_control",
        "penalty_0.01",
        "delete_stripe_9",
        "delete_stripe_10",
        "delete_stripe_12",
    ]
    expected = {
        (0, r)
        for r in [
            "raster_hybrid",
            "represented_hfss_control",
            "full_native_hfss_control",
        ]
    }
    expected |= {(1, r) for r in ["raster_hybrid", "represented_hfss_control"]}
    expected |= {(2, r) for r in roles}
    if len(cases) != 10 or {(c["chunk"], c["role"]) for c in cases} != expected:
        raise ValueError("complete ten independent full-resolution cases required")
    for case in cases:
        if (
            case["qc"]["rows"] != 37697
            or case["qc"]["orientations"] != 43
            or not case["original_operator_closure"]["original_callable"]
        ):
            raise ValueError("qualification original/native coverage differs")
        maxima = case["maxima"]
        if set(maxima) != {
            "native_S",
            "native_A",
            "native_fg",
            "bin_S",
            "bin_A",
            "bin_fg",
        }:
            raise ValueError("complete native/count-averaged physical gates required")
        for key, value in maxima.items():
            if (
                not np.isfinite(value)
                or value > (1e-12 if key.endswith("fg") else 1e-9)
                or value < 0
            ):
                raise ValueError("full-resolution numerical gate failed: " + key)
        closure = case["original_operator_closure"]
        for key, limit in [("max_abs_S_K", 1e-9), ("max_abs_fg", 1e-12)]:
            if not np.isfinite(closure[key]) or not 0 <= closure[key] <= limit:
                raise ValueError("independent original-operator closure failed")


class QualifiedAcceleration:
    """No numerical consumer is selected from an incomplete or older key."""

    def __init__(self, pins, config, ctx, source):
        self.pins = pins
        workflow_path, workflow = checked(pins["workflow"])
        resource_path, resource = checked(pins["resource"])
        audit_path, audit = checked(pins["array_audit"])
        if (
            resource_path.parent != workflow_path.parent
            or audit_path.parent != workflow_path.parent
            or resource_path.name != "RESOURCE_DONE.json"
        ):
            raise ValueError("qualified numerical/resource/audit run identities differ")
        if (
            not workflow["source_runtime_equal_before_after"]
            or resource["status"] != "PASS"
            or resource["elapsed_s"] > 900
            or resource["peak_sampled_descendant_rss_bytes"] > 8 * 1024**3
        ):
            raise ValueError("closed source/runtime/resource qualification required")
        _, result = checked(workflow["conformance"])
        _, inputs = checked(workflow["inputs"])
        _, qualified_source = checked(workflow["source_snapshot"])
        if (
            not result["passed"]
            or result["transform_backend"] != "guarded-precomputed"
            or result["orientation_count"] != 43
        ):
            raise ValueError("complete guarded transform qualification required")
        ten_cases(result["cases"])
        if (
            audit["status"]
            != "PASS independent10case allnative and originalmask/countaverages"
            or len(audit["cases"]) != 10
            or audit["wrapper_sha256"] != pins["workflow"]["sha256"]
            or audit["result_sha256"] != workflow["conformance"]["sha256"]
        ):
            raise ValueError("independent all-native array audit required")
        for key in PHYSICS:
            if inputs["model"][key] != config["model"][key]:
                raise ValueError("qualified physical input changed: " + key)
        for key in ["field_operands", "raw_beam"]:
            if inputs[key] != config[key]:
                raise ValueError("qualified beam identity changed: " + key)
        for name, digest in qualified_source["files_sha256"].items():
            if name == "fine_night_d5/src/fine_night_d5/four_channel_batch.py":
                continue  # Explicitly reviewed scheduling/integration consumer.
            if source["files_sha256"].get(name) != digest:
                raise ValueError("qualified numerical implementation changed: " + name)
        if result["frame_signature"]["context"] != ctx["identity"]:
            raise ValueError("qualified physical frame context changed")
        self.result = result
        self.ctx = ctx

    def orientation(self, frame):
        if frame.signature != self.result["frame_signature"]:
            raise ValueError("actual accelerated frame differs from qualification")
        drive = DriveOperands(
            self.result["drive_operands"],
            drive_eulers(self.ctx["pointing"]),
            self.ctx["identity"],
        )
        sht = SHTOperands(self.result["transform_operands"], self.ctx["identity"])
        return sht.orientation(
            drive, frame.template.beam_norm, frame.template.sim.lmax + 1
        )


def original_closure(setup, point, values, cmb):
    """Up to128 actual rows of the most-supported original drive group."""
    import jax.numpy as jnp
    from croissant.rotations import rotmat_to_eulerZYZ
    from croissant.simulator import convolve
    from eigsim.rotations import drive_rotation_matrix

    started = time.monotonic()
    groups = orientation_groups(
        point["el_drive_deg"],
        point["az_drive_deg"],
        point["rotations"],
        point["misalignment"],
    )
    el, az, rows = max(groups, key=lambda group: len(group[2]))
    rows = rows[:128]
    euler = jnp.asarray(
        rotmat_to_eulerZYZ(
            drive_rotation_matrix(el, az, misalignment=point["misalignment"])
        ),
        dtype=jnp.float64,
    )
    beam, fg = setup.orient(
        setup.alm, euler, setup.sim, setup.horizon, setup.quad_weights
    )
    signal = np.asarray(
        (
            convolve(beam, setup.sky_alm, setup.phases[rows]) / setup.beam_norm[None, :]
        ).real
    )
    ground = np.broadcast_to(np.asarray(fg), signal.shape)
    np.testing.assert_allclose(values[rows, :, 0], signal, rtol=0, atol=1e-9)
    np.testing.assert_allclose(values[rows, :, 1], ground, rtol=0, atol=1e-12)
    actual_a = values[rows, :, 0] + cmb * (1 - values[rows, :, 1])
    original_a = signal + cmb * (1 - ground)
    np.testing.assert_allclose(actual_a, original_a, rtol=0, atol=1e-9)
    return dict(
        original_callable=True,
        native_positions=rows.tolist(),
        elevation_deg=el,
        azimuth_deg=az,
        max_abs_S_K=float(np.max(abs(values[rows, :, 0] - signal))),
        max_abs_A_K=float(np.max(abs(actual_a - original_a))),
        max_abs_fg=float(np.max(abs(values[rows, :, 1] - ground))),
        elapsed_s=time.monotonic() - started,
    )
