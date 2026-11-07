"""Pins reject changed bytes/schema; new configs never substitute old beams."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from eigsim import data, load_config
from eigsim import hfss_pin as hp


def write_pin(path, arrays, pin):
    np.savez(path, **arrays)
    pin = copy.deepcopy(pin)
    pin.update(sha256=hp.sha256(path), bytes=path.stat().st_size)
    manifest = path.with_suffix(".json")
    manifest.write_text(json.dumps(pin))
    return manifest


@pytest.fixture
def native(tmp_path):
    pin = json.loads(hp.NATIVE_PIN.read_text())
    pin["schema"].update(
        L=4,
        E_mV_shape=[3, 3, 5, 8],
        frequency_mhz=dict(start=50, stop=52, step=1),
        theta_deg=dict(start=0, stop=180, step=45),
        phi_deg=dict(start=0, stop=315, step=45),
    )
    arrays = dict(
        E_mV=np.ones((3, 3, 5, 8), dtype=np.complex128),
        freqs_mhz=np.arange(50.0, 53.0),
        theta_deg=np.linspace(0, 180, 5),
        phi_deg=np.arange(8) * 45.0,
        L=4,
        sampling="mwss",
        product_version=pin["version"],
        field_quantity=pin["schema"]["field_quantity"],
        coordinate_system=pin["schema"]["coordinate_system"],
        source_sha256=pin["raw_source_sha256"],
        source_metadata_sha256=pin["raw_provenance_sha256"],
        physical_port_plane=pin["schema"]["physical_port_plane"],
        S33_50ohm=np.ones(3, complex),
        S33_75ohm=np.ones(3, complex),
        Z33_report_ohm=np.ones(3, complex),
    )
    path = tmp_path / "native.npz"
    return path, arrays, pin, write_pin(path, arrays, pin)


def test_native_preserves_arrays_and_rejects_changed_bytes(native):
    path, arrays, _, manifest = native
    got, _ = hp.load_native_fields(path, manifest)
    for k, v in arrays.items():
        np.testing.assert_array_equal(got[k], v)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        hp.load_native_fields(path, manifest)


@pytest.mark.parametrize("change", ["axis", "dtype", "frame", "source", "finite"])
def test_native_rejects_rehashed_wrong_schema(native, change):
    path, arrays, pin, _ = native
    if change == "axis":
        arrays["theta_deg"][1] += 0.01
    elif change == "dtype":
        arrays["E_mV"] = arrays["E_mV"].astype(np.complex64)
    elif change == "frame":
        arrays["coordinate_system"] = "unknown sky frame"
    elif change == "source":
        arrays["source_sha256"] = "0" * 64
    else:
        arrays["E_mV"][0, 0, 0, 0] = np.nan
    manifest = write_pin(path, arrays, pin)
    with pytest.raises(ValueError):
        hp.load_native_fields(path, manifest)


def test_new_config_is_explicit_and_pending_pin_never_falls_back(tmp_path, monkeypatch):
    cfg = load_config("eigsep_v003_1mhz")
    np.testing.assert_array_equal(cfg["frequencies"], np.arange(50.0, 251.0))
    assert cfg["beam"]["pin_manifest"] == hp.POWER_PIN.name
    pending = json.loads(hp.POWER_PIN.read_text())
    pending.update(sha256=None, bytes=None)
    (tmp_path / hp.POWER_PIN.name).write_text(json.dumps(pending))
    monkeypatch.setattr(hp, "PIN_DIR", tmp_path)
    with pytest.raises(ValueError, match="reviewed checksum"):
        data.load_beam(config="eigsep_v003_1mhz")
    assert load_config()["beam"]["file"] == "eigsep_bowtie_v002_mwss.npz"
    assert (
        load_config("eigsep_1mhz")["beam"]["file"] == "eigsep_bowtie_v002_1mhz_mwss.npz"
    )


def test_pinned_power_schema_parent_and_config_frequency_check(tmp_path):
    pin = json.loads(hp.POWER_PIN.read_text())
    pin["schema"] = dict(
        lmax=1, bm_shape=[1, 3, 4], frequency_hz=dict(start=50e6, stop=50e6, step=1e6)
    )
    arrays = dict(
        freqs=np.array([50e6]), bm=np.ones((1, 3, 4)), lmax=1, **pin["metadata"]
    )
    path = tmp_path / "power.npz"
    manifest = write_pin(path, arrays, pin)
    f, bm, lmax = hp.load_pinned_power(path, manifest)
    assert lmax == 1 and bm.shape == (1, 3, 4)
    np.testing.assert_array_equal(f, [50e6])
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"frequencies: [51.0]\nbeam:\n  file: {path}\n  pin_manifest: {manifest}\n"
    )
    with pytest.raises(ValueError, match="opt-in config"):
        data.load_beam(config=cfg)
    arrays["freqs"] += 1
    changed = write_pin(path, arrays, pin)
    with pytest.raises(ValueError, match="axes/schema"):
        hp.load_pinned_power(path, changed)
    pin["parent_native_sha256"] = "0" * 64
    changed = write_pin(path, arrays, pin)
    with pytest.raises(ValueError, match="parent"):
        hp.load_pinned_power(path, changed)


def test_v002_constant_harmonic_truncation_and_raw_normalization():
    # Reuse the unchanged old transform at tiny L; no real L180 beam/SHT.
    script = Path(__file__).parents[1] / "scripts/make_bowtie_v002.py"
    spec = importlib.util.spec_from_file_location("old_bowtie_recipe", script)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    weights = np.asarray(
        old.s2fft.utils.quadrature_jax.quad_weights(L=4, sampling="mwss")
    )
    power = np.full((5, 8), 3.0)
    normalized, integral = hp.directivity(power, weights)
    np.testing.assert_allclose(integral, 12 * np.pi, rtol=1e-14)
    np.testing.assert_allclose(normalized, 1, rtol=1e-14)
    out, dropped = old.native_to_mwss(normalized[None], 4, 2)
    np.testing.assert_allclose(out, 1, atol=1e-13)
    np.testing.assert_allclose(dropped, 0, atol=1e-14)
