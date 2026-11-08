"""Independent equation, mask, source split and actual kernel controls."""

import copy
import gzip

import numpy as np
import pytest
from eigsep_analysis import ground_reflect as gr
from eigsep_analysis import s11_hypotheses as obs
from eigsep_analysis import s11_hypothesis_io as intake

from s11_hypotheses import checkpoint, feature, leakage, terrain
from s11_hypotheses.config import DEFAULTS


def test_overlapping_top_hats_reproduce_source_and_zero_native_support():
    source = np.array([46.75e6, 47e6])
    values = np.array([2 + 3j, 7 - 1j])
    f = np.linspace(1e6, 250e6, 1000)
    for width in DEFAULTS["widths_hz"]:
        assert np.allclose(
            feature.width_template(source, values, width)["template"], values
        )
    assert feature.width_template(f, values, 0.05e6)["status"] == "UNOBSERVED"
    assert feature.width_template(f, values, 0.1e6)["status"] == "UNOBSERVED"
    assert feature.width_template(f, values, 0.25e6)["support"].sum() == 2
    assert feature.width_template(f, values, 0.5e6)["support"].sum() == 3


def test_source_estimation_uses_only_explicit_source_rows():
    f = np.array([46.75e6, 47e6])
    r = np.array([[1, 2], [3, 4], [999, 999]], complex)
    a = feature.source_template(f, r, [0, 1])
    assert a["values_re_im"] == [[2, 0], [3, 0]]
    r[2] = -99999j
    assert feature.source_template(f, r, [0, 1]) == a
    assert len(a["leave_one_source_out_re_im"]) == 2


def test_upstream_injection_matches_explicit_refit_and_homogeneity():
    f = np.linspace(1e6, 125e6, 497)
    y = np.exp(-2j * np.pi * f * 30e-9)
    selection = dict(
        support_s=150e-9,
        cutoff=1e-9,
        ridge_fraction=1e-6,
        delays_s=[],
        target_hz=[46e6, 48e6],
    )
    t = feature.width_template(f, [2e-4j, 3e-4], 0.5e6)["template"]
    r = feature.inject_and_recover(f, y, selection, [46e6, 48e6], t, 1, 0)
    other = feature.inject_and_recover(f, y, selection, [46e6, 48e6], t, 2, np.pi / 2)
    assert np.allclose(other["recovered"], 2j * r["recovered"], atol=1e-13)
    assert other["magnitude"] == pytest.approx(r["magnitude"], abs=1e-8)


def test_long_line_native_alias_and_fine_array_separation():
    f = np.linspace(1e6, 250e6, 1000)
    native, fine = terrain.frequency_grids(f, 1e4, 49e-6)
    assert len(fine) == 40001
    assert np.all(np.diff(fine) == 5000)
    assert np.array_equal(native, f[f >= 50e6])
    c = leakage.controls(f)
    period = (len(f) - 1) / (f[-1] - f[0])
    factor = np.exp(-2j * np.pi * f[0] * period)
    assert np.max(abs(c["far_plus_period"] - factor * c["far450m"])) < 1e-10
    signed = obs.signed_prediction(f, c["far450m"], c["far450m"])
    peak = signed["tau_s"][np.argmax(abs(signed["data"]))]
    assert peak < -0.9e-6 and peak > -1.1e-6


def test_r0_reference_pixel_split_and_diffuse_area_convention():
    # Each64 pixel split into16 equal rays at256: historical sum16x,
    # common-reference coherent amplitude unchanged. sqrt-area diffuse
    # expected power is separately invariant and gets no coherent factor.
    assert 16 * terrain.r0_normalization(256) == terrain.r0_normalization(64)
    omega64 = 4 * np.pi / (12 * 64**2)
    omega256 = 4 * np.pi / (12 * 256**2)
    assert 16 * (np.sqrt(omega256) ** 2) == pytest.approx(omega64)


def tiny_geometry():
    return dict(
        E=np.array([-1.0, 1.0, 0.0]),
        N=np.array([0.0, 0.0, 2.0]),
        z=np.array([0.0, 0.0, 0.0]),
        dzdE=np.zeros(3),
        dzdN=np.zeros(3),
        area_m2=np.ones(3),
        grid_index=np.arange(3),
    )


def isotropic_beam():
    return dict(
        D=np.ones((3, 4, 2)),
        E=np.ones((3, 4, 2, 3), complex),
        f_hz=np.array([50e6, 250e6]),
    )


def test_existing_po_kernel_and_native_fine_same_physical_draw():
    f = np.linspace(50e6, 250e6, 33)
    c = tiny_geometry()
    ant = np.array([0.0, 0.0, 100.0])
    choices = copy.deepcopy(DEFAULTS)
    choices.update(cell_chunk=2, frequency_chunk=8, terminal_taper_fraction=0.0)
    a = terrain.facet_spectra(
        f,
        c,
        ant,
        isotropic_beam(),
        np.eye(3),
        choices,
        600,
        0.01,
        member=2,
        bin_deg=None,
        covariance=True,
    )
    choices["cell_chunk"] = 3
    b = terrain.facet_spectra(
        f,
        c,
        ant,
        isotropic_beam(),
        np.eye(3),
        choices,
        600,
        0.01,
        member=2,
        bin_deg=None,
        covariance=False,
    )
    assert np.allclose(a["po"], b["po"])
    assert np.allclose(a["diffuse"], b["diffuse"])
    geom = gr.member_geometry(c, ant)
    dn, wn = gr.facet_beam(isotropic_beam(), np.eye(3), c, ant)
    eps = gr.eps_complex(f, choices["material_eps"], choices["material_sigma_s_m"])
    rs, rp = gr.fresnel_rs_rp(geom["cos_inc"][None], eps[:, None])
    direct = gr.po_direct(
        f,
        geom,
        gr.r_mono(rs, rp, wn[:, 0][None])
        * gr.ament_chi(f[:, None], 0.01, geom["cos_inc"][None]),
        1.0,
    )
    assert np.allclose(a["po"].sum(axis=0), direct, rtol=1e-6, atol=1e-12)
    cov = a["diffuse_delay_covariance"].sum(axis=0)
    assert np.linalg.eigvalsh(cov).min() > -1e-15
    assert a["diffuse_delay_expected_power"].min() >= 0


def test_no_hidden_220m_cut_and_shell_coherent_cross_terms():
    assert terrain.terminal_taper(np.array([450.0]), 600)[0] == 1
    assert gr.slant_taper(np.array([450.0]))[0] == 0
    assert terrain.shell_index([220, 301, 450, 601], [0, 220, 301, 600]).tolist() == [
        0,
        1,
        2,
        -1,
    ]
    coherent = np.array([1.0, -1.0], complex)
    assert abs(coherent.sum()) ** 2 == 0
    assert sum(abs(coherent) ** 2) == 2


def test_covariance_operator_mask_and_negative_line_absorption():
    f = np.linspace(1e6, 125e6, 100)
    a = obs.causal_basis(f, 0.15e-6)
    mask = np.ones(100, bool)
    mask[10:13] = False
    imp = leakage.influence(f, a, mask, 1e-6, mask, "bh7")
    assert np.all(imp["residual_operator"][:, ~mask] == 0)
    cov = leakage.covariance_through(imp["residual_operator"], np.eye(100))
    assert np.linalg.eigvalsh(cov).min() > -1e-12
    line = leakage.controls(f)["negative_candidate"]
    assert np.linalg.norm(imp["residual_operator"] @ line) > 0


def test_checkpoint_mismatch_and_corruption_fail_closed(tmp_path):
    unit = checkpoint.Unit(tmp_path, "case", "good")
    checkpoint.write_h5(
        unit.path / "values.h5", dict(x=np.arange(3), mask=[True, False, True])
    )
    unit.finish(dict(scope="test"))
    assert checkpoint.Unit(tmp_path, "case", "good").completed()
    with pytest.raises(ValueError, match="identity"):
        checkpoint.Unit(tmp_path, "case", "different").completed()
    (unit.path / "values.h5").write_bytes(b"changed")
    with pytest.raises(ValueError, match="result"):
        unit.completed()


def test_streaming_raw_reader_preserves_complex_and_clock(tmp_path):
    path = tmp_path / "raw.gz"
    with gzip.open(path, "wt") as h:
        h.write("0 2024 10 19 17 49 0 10 46.75 47.00\n")
        h.write("0 (2024+0j) 10 19 17 49 1 (12+0j) 1+-2j 3+4j\n")
        h.write("0 2024 10 19 17 49 2 31 not-parsed-PSD\n")
    data = intake.read_mistic_vna(path)
    assert np.array_equal(data[10]["values"][0], [46.75e6, 47e6])
    assert np.array_equal(data[12]["values"][0], [1 - 2j, 3 + 4j])
    assert data[12]["time_utc_s"][0] - data[10]["time_utc_s"][0] == 1


def test_gate_rejects_unapproved_scope_and_does_not_run(tmp_path):
    import json

    from s11_hypotheses.run import gate, physical_units

    config = dict(
        resource_gate=str(tmp_path / "gate.json"), choices=copy.deepcopy(DEFAULTS)
    )
    identity = {"source": "test"}
    with pytest.raises(ValueError, match="resource gate"):
        gate(config, "probe", identity, "manifest")
    record = dict(
        allow=["probe"],
        source_fingerprint=checkpoint.fingerprint(identity),
        manifest_sha256="manifest",
        configuration_fingerprint=checkpoint.fingerprint(
            dict(choices=config["choices"], units=physical_units(config, True))
        ),
    )
    (tmp_path / "gate.json").write_text(json.dumps(record))
    assert gate(config, "probe", identity, "manifest") == record
    with pytest.raises(ValueError, match="authorize"):
        gate(config, "score", identity, "manifest")


def test_freeze_source_split_and_positive_target_blinding():
    from eigsep_analysis import s11_hypothesis_pipeline as pipeline

    f = np.arange(1, 125.01, 0.25) * 1e6
    values = np.array([np.exp(-2j * np.pi * f * 20e-9) for _ in range(4)])
    values[:2, f == 46.75e6] += 2e-4j
    values[:2, f == 47e6] += 3e-4
    group = dict(
        frequency_hz=f,
        values=values,
        capture_ids=np.array([f"capture{i}" for i in range(4)]),
    )
    choices = copy.deepcopy(DEFAULTS)
    choices.update(
        source_canyon_indices=[0, 1],
        positive_control_canyon_indices=[2, 3],
        dpss_supports_s=[0.15e-6],
        ridge_fractions=[1e-6],
        control_start_mhz=[10, 14, 18, 22],
    )
    a, _ = pipeline.freeze_design(
        None, {"MISTIC:canyon:short": group}, choices, feature, "source", "input"
    )
    values[2:, (f >= 46e6) & (f < 48e6)] = 1000j
    b, _ = pipeline.freeze_design(
        None, {"MISTIC:canyon:short": group}, choices, feature, "source", "input"
    )
    assert a == b
    assert set(a["source_capture_ids"]).isdisjoint(a["positive_control_capture_ids"])


def test_independent_fine_native_and_exact_diffuse_expected_power():
    f = np.linspace(50e6, 250e6, 33)
    fine = np.linspace(50e6, 250e6, 65)
    c, ant = tiny_geometry(), np.array([0.0, 0.0, 100.0])
    choices = copy.deepcopy(DEFAULTS)
    choices.update(cell_chunk=2, frequency_chunk=8, terminal_taper_fraction=0.0)
    a = terrain.facet_spectra(
        f,
        c,
        ant,
        isotropic_beam(),
        np.eye(3),
        choices,
        600,
        0.01,
        member=2,
        bin_deg=None,
    )
    b = terrain.facet_spectra(
        fine,
        c,
        ant,
        isotropic_beam(),
        np.eye(3),
        choices,
        600,
        0.01,
        member=2,
        bin_deg=None,
    )
    assert np.allclose(a["po"], b["po"][:, ::2], rtol=1e-12, atol=1e-14)
    assert np.allclose(a["diffuse"], b["diffuse"][:, ::2], rtol=1e-12, atol=1e-14)
    geom = gr.member_geometry(c, ant)
    _, ws = gr.facet_beam(isotropic_beam(), np.eye(3), c, ant)
    eps = gr.eps_complex(f, choices["material_eps"], choices["material_sigma_s_m"])
    rs, rp = gr.fresnel_rs_rp(geom["cos_inc"][None], eps[:, None])
    gamma = ws[:, 0][None] * abs(rs) ** 2 + (1 - ws[:, 0][None]) * abs(rp) ** 2
    share = gr.ament_share(f[:, None], 0.01, geom["cos_inc"][None])
    expected = gr.incoherent_power(f, geom, np.ones_like(gamma), gamma, share).sum(
        axis=1
    )
    assert np.allclose(
        a["diffuse_frequency_expected_power"].sum(axis=0),
        expected,
        rtol=1e-12,
        atol=1e-16,
    )


def test_complete_power_report_executes_ac5_and_signed_export(tmp_path):
    """Exercise real041 operators, fit, all epoch scores and report I/O."""
    import h5py
    import pandas as pd
    from eigsep_analysis import s11_hypothesis_pipeline as pipeline

    f_all = np.linspace(1e6, 250e6, 1000)
    f = f_all[f_all >= 50e6]
    times = np.arange(14, dtype=float) + 1000
    base = np.exp(-2j * np.pi * f_all * 600e-9)
    with h5py.File(tmp_path / "ant.h5", "w") as h:
        h["freqs"], h["timestamps"] = f_all, times
        for i in range(14):
            h[f"t{i}/lna"] = base * (1 + 0.01j * i)
    b0 = pd.DataFrame(
        dict(
            t=times,
            n_terms=4,
            floor_kind="F_plus",
            is_reference=[True] * 12 + [False, False],
            epoch_local=[f"night{i}" for i in range(12)]
            + ["07-17 13:10", "07-17 19:17"],
        )
    )
    b0.to_csv(tmp_path / "b0.csv", index=False)
    pd.DataFrame(dict(dut="ant", t=times, osl_t=times // 3)).to_csv(
        tmp_path / "donor.csv", index=False
    )
    echo = {}
    for epoch in ["night", "1310", "1917"]:
        op = gr.operator_O(f, 600e-9)
        st = gr.apply_O(op, base[f_all >= 50e6])
        env, pb = st["env"], st["pb"]
        d = np.r_[np.log(env / env.sum()), np.log(pb)]
        for key, value in dict(
            d=d,
            var_th=np.ones_like(d) * 0.01,
            var_jit=np.ones_like(d) * 0.01,
            nbar_env=env * 1e-6,
            nbar_pb=pb * 1e-6,
        ).items():
            echo[f"{epoch}_{key}"] = value
        echo[f"tau_lead_{epoch}"] = 600e-9
    np.savez(tmp_path / "echo.npz", **echo)

    class Store:
        def suffix(self, ending):
            if ending.endswith("ant_calibrated.h5"):
                return tmp_path / "ant.h5"
            if ending.endswith("echo_data.npz"):
                return tmp_path / "echo.npz"
            if ending.endswith("b0_features.csv"):
                return tmp_path / "b0.csv"
            if ending.endswith("D_epoch_osl.csv"):
                return tmp_path / "donor.csv"
            raise AssertionError(ending)

    exports = []
    for epoch in ["night", "1310", "1917"]:
        for member in range(2):
            for grid, ff in [("native", f), ("fine", np.linspace(50e6, 250e6, 2001))]:
                name = f"facet_tiny_{epoch}_m{member:03d}_{grid}"
                near = np.exp(-2j * np.pi * ff * 600e-9)
                far = 0.01 * np.exp(-2j * np.pi * ff * 3e-6)
                arrays = dict(
                    frequency_hz=ff,
                    po=np.array([near, far]),
                    diffuse=np.zeros((2, len(ff)), complex),
                    legacy_near_po=near,
                    legacy_near_diffuse=np.zeros_like(near),
                    diffuse_delay_expected_power=np.ones((2, len(ff))) * 1e-10,
                    diffuse_delay_covariance=np.ones((2, 2, 2)) * 1e-12,
                    tau_s=np.fft.fftshift(np.fft.fftfreq(len(ff), ff[1] - ff[0])),
                )
                path = tmp_path / "units" / name / "spectra.h5"
                checkpoint.write_h5(path, arrays)
                if grid == "native":
                    exports.append(
                        dict(
                            configuration="tiny",
                            radius_m=600.0,
                            stride=4,
                            bin_deg=1.0,
                            epoch=epoch,
                            member=member,
                            path=str(path),
                        )
                    )
    choices = copy.deepcopy(DEFAULTS)
    choices["tau0_grid_ns"] = [-1, 1, 1]
    rows = pipeline.terrain_power_report(
        Store(),
        exports,
        choices,
        checkpoint.write_h5,
        checkpoint.write_json,
        tmp_path / "report",
    )
    convergence = terrain.convergence_report(exports, choices)
    assert not convergence["comparisons"]
    assert "limited" in convergence["status"]
    assert len(convergence["last_shells"]) == 3
    assert len(rows) == 1
    assert rows[0]["ac5"]["predicted_power"] > 0
    assert rows[0]["ac5"]["consecutive_difference_proxy"] > 0
    assert {r["epoch"] for r in rows[0]["scores"]} == {"night", "1310", "1917"}
    with h5py.File(tmp_path / "report/signed_power.h5", "r") as h:
        assert len(h["fine_tau_s"]) > len(h["tau_s"])
        assert np.any(h["tau_s"][:] < 0)


def test_fingerprint_algorithm_source_mutation_invalidates_identity(
    tmp_path, monkeypatch
):
    mock = tmp_path / "mock"
    code = tmp_path / "code"
    (mock / "s11_hypotheses/src/s11_hypotheses").mkdir(parents=True)
    (mock / "pyproject.toml").write_text("# test package\n")
    (mock / "s11_hypotheses/pyproject.toml").write_text("# adapter\n")
    (mock / "s11_hypotheses/src/s11_hypotheses/__init__.py").write_text("")
    (mock / "s11_hypotheses/src/s11_hypotheses/runtime_contract.json").write_text("{}")

    def manifest(root, entrypoints):
        for name in entrypoints:
            path = root / name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# frozen algorithm source\n")
        return dict(sources={name: intake.digest(root / name) for name in entrypoints})

    monkeypatch.setattr(checkpoint.contract, "code_manifest", manifest)
    monkeypatch.setattr(checkpoint.metadata, "version", lambda _: "test")
    first = checkpoint.source_identity(mock, code)
    algorithm = "src/eigsep_analysis/run_contract.py"
    assert algorithm in first["code"]["sources"]
    (code / algorithm).write_text("# hashing algorithm changed\n")
    second = checkpoint.source_identity(mock, code)
    assert checkpoint.fingerprint(first) != checkpoint.fingerprint(second)
    policy = mock / "s11_hypotheses/src/s11_hypotheses/runtime_contract.json"
    policy.write_text('{"analysis_commit": "different-source"}')
    third = checkpoint.source_identity(mock, code)
    assert checkpoint.fingerprint(second) != checkpoint.fingerprint(third)
