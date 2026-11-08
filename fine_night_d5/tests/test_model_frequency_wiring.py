"""Real producer/reader I/O with explicitly fake physical backends only."""

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from fine_night_d5.join import join_arrays, read_piece
from fine_night_d5.model_frequency import PAYLOAD_FIELDS

from fine_night_d5 import assets, model


@pytest.mark.parametrize("centre_control", [False, True])
def test_producer_backend_frequencies_and_pinned_payload_roundtrip(
    tmp_path, monkeypatch, centre_control
):
    channels = np.array(
        [
            283,
            284,
            285,
            291,
            292,
            293,
            603,
            604,
            605,
            611,
            612,
            613,
            843,
            844,
            845,
            851,
            852,
            853,
        ]
    )
    centres = np.array([70.3125, 148.4375, 207.03125])
    data_f = channels * 250 / 1024
    eval_f = centres if centre_control else data_f
    use = np.ones((4, 18), bool)
    use[:, 0], use[:, 1] = [True, False, False, False], [False, True, False, False]
    use[:, 2], use[:, 3] = [True, True, True, False], False
    bins = np.array([0, 0, 0, 1])
    coords = {
        "source_file": np.full(4, "synthetic-fixture"),
        "row": np.arange(4),
        "input": np.full(4, "4"),
        "time_best": np.array([0.0, 1.0, 3.0, 9.0]),
    }

    def pin_json(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {"path": str(path), "sha256": assets.sha256(path)}

    def stage(name, arrays, metadata):
        path = tmp_path / name
        (path / "sources").mkdir(parents=True)
        source = path / "sources/fine_night.py"
        source.write_text("# synthetic stage producer for operator I/O test\n")
        np.savez(path / "arrays.npz", **arrays)
        record = {
            "schema": 1,
            "status": "COMPLETE",
            "label": "diagnostic",
            "arrays_sha256": assets.sha256(path / "arrays.npz"),
            "producer_source_sha256": assets.sha256(source),
            "producer_sources": {"fine_night.py": assets.sha256(source)},
            "metadata": metadata,
        }
        (path / "STAGE.json").write_text(json.dumps(record))
        return {
            "path": str(path / "STAGE.json"),
            "sha256": assets.sha256(path / "STAGE.json"),
        }

    expected_count = np.array(
        [
            [sum(use[i, j] for i in range(4) if bins[i] == b) for j in range(18)]
            for b in range(3)
        ]
    )
    native = stage(
        "native",
        {
            **{"native__" + k: v for k, v in coords.items()},
            "native__chans": channels,
            "native__accepted": use,
            "grid__native_to_bin": bins,
            "grid__fine_bin_id": np.arange(3),
            "bin__count": expected_count,
        },
        {"input_manifest_sha256": "fixture-input"},
    )
    nominal = pin_json(
        "nominal.json",
        {
            "input_sha256": {"horizon": "fixture-horizon"},
            "iers": {
                "finals2000A_sha256": "fixture-finals",
                "eopc04_sha256": "fixture-eopc",
            },
            "site": {"lat_deg": 1, "lon_deg": 2, "height_m": 3},
            "tau_imu": {"tau_deg": 0},
        },
    )
    pointing = stage(
        "pointing",
        {
            **coords,
            "native_index": np.arange(4),
            "rotation": np.repeat(np.eye(3)[None], 4, axis=0),
            "az_drive_deg": np.zeros(4),
            "el_drive_deg": np.zeros(4),
        },
        {
            "native_stage_sha256": native["sha256"],
            "input_manifest_sha256": "fixture-input",
            "night_tau_deg": 0,
            "mount": {"psi_deg": 0},
            "tilt_authority": {"consumed_scalar_source_sha256": nominal["sha256"]},
        },
    )
    backend_source = tmp_path / "fake_backend.py"
    backend_source.write_text("# No physical backend imported or executed\n")
    dependency_sources = {
        k: assets.sha256(backend_source)
        for k in ("eigsim.simulate", "eigsim.rotations", "croissant.simulator")
    }
    runtime = pin_json(
        "runtime.json",
        {
            "dependency_sources": {
                k: {"sha256": h} for k, h in dependency_sources.items()
            },
            "jax_binaries": {"fake-fixture-only": {"sha256": "not-an-executable"}},
        },
    )
    config = {
        "schema": 1,
        "label": "diagnostic",
        "channels": channels.tolist(),
        "runtime_versions": {
            n: "fake-backend"
            for n in [
                "numpy",
                "scipy",
                "jax",
                "jaxlib",
                "s2fft",
                "croissant-sim",
                "healpy",
                "pygdsm",
                "astropy",
            ]
        },
        "runtime_modules": dependency_sources,
        "runtime_evidence": runtime,
        "native_stage": native,
        "pointing_stage": pointing,
        "nominal034": nominal,
        "provenance034": pin_json(
            "provenance.json", {"params": {"model": {"t_cmb_k": 2.725}}}
        ),
        "beam": {"role": "old_symmetric", "sha256": "fixture-beam"},
        "horizon": {"sha256": "fixture-horizon"},
        "iers": {
            "finals2000A.all": {"sha256": "fixture-finals"},
            "eopc04.1962-now": {"sha256": "fixture-eopc"},
        },
        "gsm": {"fixture": True},
        "chunk_rows": 2,
    }
    if centre_control:
        config["model_frequency_operator"] = {
            "version": 1,
            "kind": "c53-nominal-centres",
            "data_channels": channels.tolist(),
            "evaluation_frequencies_mhz": centres.tolist(),
            "data_to_evaluation": [0] * 6 + [1] * 6 + [2] * 6,
        }
    calls = {}

    def scalar_beam(pin, f):
        calls["beam"] = np.array(f)
        return (
            np.ones((len(f), 1, 1)),
            {"role": pin["role"], "input_sha256": pin["sha256"]},
            {},
        )

    def gsm_maps(config, f, out):
        calls["gsm"] = np.array(f)
        return np.zeros((len(f), 12)), {"fixture": True}

    def sky(maps, f, **kwargs):
        calls["sky"] = np.array(f)
        assert maps.shape[0] == len(f)
        return SimpleNamespace(f=np.array(f))

    def simulate(
        beam, f, sky, horizon, times, el, az, rotations, mis, site, out, **kwargs
    ):
        calls["convolution"] = np.array(f)
        np.testing.assert_array_equal(sky.f, f)
        np.testing.assert_array_equal(times, coords["time_best"])
        # An explicitly fake physical response. Real expansion, CMB,
        # measured reduction, on-disk producer and reader remain unmocked.
        s = times[:, None] ** 2 + f[None, :]
        return np.stack([s, np.broadcast_to(f / 10000, s.shape)], -1), {"fixture": True}

    monkeypatch.setattr(
        model,
        "metadata",
        SimpleNamespace(
            version=lambda n: "fake-backend", distributions=model.metadata.distributions
        ),
    )
    monkeypatch.setattr(
        model,
        "importlib",
        SimpleNamespace(
            import_module=lambda n: SimpleNamespace(__file__=str(backend_source))
        ),
    )
    monkeypatch.setattr(assets, "scalar_beam", scalar_beam)
    monkeypatch.setattr(assets, "gsm_maps", gsm_maps)
    monkeypatch.setattr(assets, "install_iers", lambda value: value)
    monkeypatch.setattr(
        assets, "horizon_weight", lambda value: (np.ones((1, 1)), value)
    )
    monkeypatch.setattr(model, "simulate_native", simulate)
    cro = ModuleType("croissant")
    cro.Sky = sky
    eigsim = ModuleType("eigsim")
    rotations = ModuleType("eigsim.rotations")
    rotations.misalignment_matrix = lambda **kwargs: np.eye(3)
    monkeypatch.setitem(sys.modules, "croissant", cro)
    monkeypatch.setitem(sys.modules, "eigsim", eigsim)
    monkeypatch.setitem(sys.modules, "eigsim.rotations", rotations)
    inputs = pin_json("inputs.json", config)
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys, "argv", ["model", "--inputs", inputs["path"], "--out", str(output)]
    )
    model.main()
    for backend in ("beam", "gsm", "sky", "convolution"):
        np.testing.assert_array_equal(calls[backend], eval_f)
    pin = {
        "done": {
            "path": str(output / "RUN_DONE.json"),
            "sha256": assets.sha256(output / "RUN_DONE.json"),
        },
        "inputs": inputs,
    }
    (arrays, identity), proof, record = read_piece(pin)
    np.testing.assert_array_equal(arrays["channels"], channels)
    np.testing.assert_array_equal(arrays["frequencies_mhz"], data_f)
    np.testing.assert_array_equal(arrays["accepted"], use)
    for key, value in coords.items():
        np.testing.assert_array_equal(arrays[key], value)
    mapped_f = np.repeat(centres, 6) if centre_control else data_f
    expected_s = coords["time_best"][:, None] ** 2 + mapped_f[None, :]
    expected_fg = np.broadcast_to(mapped_f / 10000, expected_s.shape)
    expected_a = expected_s + 2.725 * (1 - expected_fg)
    for name, values in [("S", expected_s), ("fg", expected_fg), ("A", expected_a)]:
        np.testing.assert_array_equal(arrays["native_" + name], values)
        for b in range(3):
            for j in range(18):
                rows = [i for i in range(4) if bins[i] == b and use[i, j]]
                if rows:
                    assert arrays["bin_" + name][b, j] == pytest.approx(
                        sum(values[i, j] for i in rows) / len(rows), abs=1e-12
                    )
                else:
                    assert np.isnan(arrays["bin_" + name][b, j])
    np.testing.assert_array_equal(arrays["bin_count"], expected_count)
    if centre_control:
        assert len(record["model_frequency_operator"]["data_channels"]) == 18
        assert (
            len(record["model_frequency_operator"]["evaluation_frequencies_mhz"]) == 3
        )
        assert (
            identity["configuration"]["model_frequency_operator"]
            == config["model_frequency_operator"]
        )
        with pytest.raises(ValueError, match="cannot be chunk-joined"):
            join_arrays([(arrays, identity)], channels)
        # A rehashed payload/record still cannot substitute other evalν.
        arrays["model_evaluation_frequencies_mhz"][0] += 0.5
        np.savez(output / "columns.npz", **arrays)
        record["columns_sha256"] = assets.sha256(output / "columns.npz")
        (output / "RUN_DONE.json").write_text(json.dumps(record))
        pin["done"]["sha256"] = assets.sha256(output / "RUN_DONE.json")
        with pytest.raises(ValueError, match="model-frequency coordinates"):
            read_piece(pin)
    else:
        assert not PAYLOAD_FIELDS.intersection(arrays)
        assert "model_frequency_operator" not in record
        joined, _ = join_arrays([(arrays, identity)], channels)
        np.testing.assert_array_equal(joined["native_A"], expected_a)
