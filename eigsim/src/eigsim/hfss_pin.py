"""Opt-in, fail-closed pins for the dense HFSS authority and derived power.

The complex L180 authority is not the scalar-power schema load_beam consumes.
No default, symmetry, normalization or physical reference plane is changed.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

PIN_DIR = Path(__file__).parent / "pins"
NATIVE_PIN = PIN_DIR / "hfss_1mhz_mwss_v0001.json"
POWER_PIN = PIN_DIR / "eigsep_bowtie_v003_1mhz.json"
RECIPE = "raw-sumE2-directivity-mwss-lmax128-v1"


def sha256(path):
    """Hash bytes without loading the large artifact into memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def checked_artifact(path, manifest):
    """Require an explicit released hash and exact bytes; no fallback."""
    pin = json.loads(Path(manifest).read_text())
    expected = pin.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(
            "artifact has no reviewed checksum; generation/adoption pending"
        )
    path = Path(path)
    if path.stat().st_size != pin["bytes"] or sha256(path) != expected:
        raise ValueError("artifact size/checksum differs from explicit pin")
    return pin


def _axis(row):
    return np.arange(row["start"], row["stop"] + row["step"] / 2, row["step"])


def load_native_fields(path, manifest=NATIVE_PIN):
    """Read the explicitly pinned raw complex MWSS authority unchanged.

    Returns all arrays, with no symmetry, amplitude scaling or SHT. The
    manifest retains frame/excitation/source chain and unknown physical plane.
    A missing/changed artifact is an error, never a request for the old beam.
    """
    pin = checked_artifact(path, manifest)
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    schema = pin["schema"]
    L = schema["L"]
    e = arrays["E_mV"]
    f, theta, phi = (arrays[k] for k in ("freqs_mhz", "theta_deg", "phi_deg"))
    if (
        int(arrays["L"]) != L
        or str(arrays["sampling"]) != "mwss"
        or e.shape != tuple(schema["E_mV_shape"])
        or e.shape != (len(f), 3, L + 1, 2 * L)
        or e.dtype != np.dtype("complex128")
    ):
        raise ValueError("invalid complex MWSS shape/dtype/schema")
    for name, got, expected in [
        ("frequency", f, _axis(schema["frequency_mhz"])),
        ("theta", theta, np.linspace(0, 180, L + 1)),
        ("phi", phi, np.arange(2 * L) * 180 / L),
    ]:
        if not np.array_equal(got, expected):
            raise ValueError(f"invalid native {name} axis")
    if not np.array_equal(theta, _axis(schema["theta_deg"])) or not np.array_equal(
        phi, _axis(schema["phi_deg"])
    ):
        raise ValueError("native angular axes differ from manifest")
    metadata = {
        "product_version": pin["version"],
        "field_quantity": schema["field_quantity"],
        "coordinate_system": schema["coordinate_system"],
        "source_sha256": pin["raw_source_sha256"],
        "source_metadata_sha256": pin["raw_provenance_sha256"],
        "physical_port_plane": schema["physical_port_plane"],
    }
    if any(str(arrays[k]) != value for k, value in metadata.items()):
        raise ValueError("unknown native source/frame/version metadata")
    if not np.isfinite(e).all():
        raise ValueError("nonfinite native field")
    for name in ("S33_50ohm", "S33_75ohm", "Z33_report_ohm"):
        if arrays[name].shape != f.shape or not np.isfinite(arrays[name]).all():
            raise ValueError("invalid native network array")
    return arrays, pin


def load_pinned_power(path, manifest=POWER_PIN):
    """Load a released opt-in scalar-power artifact; reject pending pins."""
    pin = checked_artifact(path, manifest)
    native = json.loads(NATIVE_PIN.read_text())
    if pin["parent_native_sha256"] != native["sha256"]:
        raise ValueError("unknown native parent")
    with np.load(path, allow_pickle=False) as z:
        f, bm, lmax = z["freqs"], z["bm"], int(z["lmax"])
        metadata = {k: str(z[k]) for k in pin["metadata"]}
    schema = pin["schema"]
    if (
        lmax != schema["lmax"]
        or bm.shape != tuple(schema["bm_shape"])
        or bm.shape != (len(f), lmax + 2, 2 * lmax + 2)
        or bm.dtype != np.dtype("float64")
        or not np.array_equal(f, _axis(schema["frequency_hz"]))
        or not np.isfinite(bm).all()
        or metadata != pin["metadata"]
    ):
        raise ValueError("invalid pinned power axes/schema/metadata")
    return f, bm, lmax


def directivity(power, theta_weights):
    """The v002 raw-power normalization; no complex-field symmetrization."""
    integral = np.einsum("tp,t->", power, theta_weights)
    if not np.isfinite(power).all() or not np.isfinite(integral) or integral <= 0:
        raise ValueError("invalid field power/integral")
    return power * (4 * np.pi / integral), float(integral)
