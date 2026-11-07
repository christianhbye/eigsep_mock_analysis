"""Small synthetic checks only; no real map/correlator input is opened."""

import healpy as hp
import numpy as np
import pytest

from compact_source_d5.identity import (
    finalize,
    producer_identity,
    read_product,
    sha256,
    snapshot_producer,
    write_json,
)
from compact_source_d5.math import (
    KB,
    METHODS,
    T_CMB,
    C,
    H,
    account_patch,
    area_sr,
    catalogue_flux,
    degrade_sparse,
    kcmb_sr_per_jy,
    neighbour_mask,
    nominal_profile,
    patch,
    sample_sparse,
)
from compact_source_d5.run import closure, coordinates, offline_parent


@pytest.mark.parametrize("nu", [50e6, 109.375e6, 234.375e6])
def test_conversion_against_planck_finite_difference(nu):
    def planck(temp):
        return 2 * H * nu**3 / C**2 / np.expm1(H * nu / (KB * temp))

    dt = 1e-3
    derivative = (planck(T_CMB + dt) - planck(T_CMB - dt)) / (2 * dt)
    assert np.isclose(kcmb_sr_per_jy(nu), 1e-26 / derivative, rtol=2e-9)
    assert 1 < kcmb_sr_per_jy(nu) / (1e-26 * C**2 / (2 * KB * nu**2)) < 1.00001


@pytest.mark.parametrize("nu", [0, -1, float("nan"), float("inf")])
def test_units_reject_invalid_frequency(nu):
    with pytest.raises(ValueError):
        kcmb_sr_per_jy(nu)


def test_catalogue_units_and_published_domain():
    r = dict(
        valid_frequency_mhz=[50, 12000],
        coefficients=[3.3498, -1.0022, -0.225, 0.023, 0.043],
        convention=dict(frequency_unit="GHz", flux_unit="Jy"),
    )
    assert catalogue_flux(r, 100) == pytest.approx(14028.13704561959)
    with pytest.raises(ValueError, match="outside catalogue"):
        catalogue_flux(r, 49.560546875)
    r = dict(
        valid_frequency_mhz=[20, 2000],
        coefficients=[4.695, 0.085, -0.178],
        convention=dict(frequency_unit="MHz", flux_unit="Jy"),
    )
    assert catalogue_flux(r, 100) == pytest.approx(14223.287871228213)
    assert catalogue_flux(r, 49.560546875) > 0


@pytest.mark.parametrize("nside", [0, 3, -2, 1.5])
def test_bad_nside(nside):
    with pytest.raises(ValueError):
        area_sr(nside)


def test_sparse_degradation_against_independent_healpy_full_map():
    rng = np.random.default_rng(159)
    pixels = rng.choice(12 * 32**2, 500, replace=False)
    values = rng.random(len(pixels))
    full = np.zeros(12 * 32**2)
    full[pixels] = values
    reduced = degrade_sparse(pixels, values, 32, 8)
    np.testing.assert_allclose(reduced, hp.ud_grade(full, 8), rtol=1e-15, atol=1e-18)
    assert reduced.sum() * area_sr(8) == pytest.approx(full.sum() * area_sr(32))
    two_step = degrade_sparse(np.arange(len(reduced)), reduced, 8, 4)
    np.testing.assert_allclose(two_step, hp.ud_grade(full, 4), rtol=1e-15, atol=1e-18)


@pytest.mark.parametrize(
    "pixels,values",
    [([-1], [1]), ([123456789], [1]), ([1, 1], [1, 1]), ([1], [float("nan")])],
)
def test_sparse_identity_rejection(pixels, values):
    with pytest.raises(ValueError):
        degrade_sparse(np.array(pixels), np.array(values), 32, 8)


@pytest.fixture
def toy():
    nside = 64  # small synthetic grid, never the real034 map
    centre = hp.ang2vec(76, 6, lonlat=True)
    geometry = patch(nside, centre)
    pixels, values = nominal_profile(nside, centre)
    q = sample_sparse(pixels, values, geometry["pixel"])
    return nside, geometry, q


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("flux", [12345.0, -2345.0])
def test_noiseless_injection_one_percent_each_background(toy, method, flux):
    nside, g, q = toy
    x, y = g["x_deg"] / 10, g["y_deg"] / 10
    bg = np.full(len(x), 200.0)
    if method != METHODS[0]:
        bg += 3 * x - 4 * y
    if method == METHODS[2]:
        bg += 2 * x * x + 0.7 * x * y - 1.2 * y * y
    data = bg + flux * kcmb_sr_per_jy(109.375e6) * q
    r, a = account_patch(data, q, g, 109.375, nside, method)
    assert r["status"] == "conditional"
    assert r["formal_flux_jy"] == pytest.approx(flux, rel=0.01)
    np.testing.assert_allclose(a["model_k"], data, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(a["residual_k"], data - a["model_k"])
    assert r["negative_formal_warning"] == (flux < 0)
    assert r["compatible_interval_jy"] is None and r["covariance_jy2"] is None
    assert q.sum() * area_sr(nside) == pytest.approx(1, rel=0.01)


def test_mask_changes_estimator_and_retains_excluded_pixels(toy):
    nside, g, q = toy
    data = 200 + 10000 * kcmb_sr_per_jy(109.375e6) * q
    bad = (g["radius_deg"] > 3) & (g["x_deg"] > 1)
    data[bad] += 4000
    corrected, a = account_patch(data, q, g, 109.375, nside, "plane8", bad)
    unmasked, _ = account_patch(data, q, g, 109.375, nside, "plane8")
    assert corrected["formal_flux_jy"] == pytest.approx(10000, rel=0.01)
    assert abs(unmasked["formal_flux_jy"] - 10000) > 100
    assert np.array_equal(a["data_k"], data)
    assert not a["fit_mask"][bad].any()
    original, _ = account_patch(data, q, g, 109.375, nside, METHODS[0], bad)
    assert original["neighbour_mask_applied"] is False


def test_rank_or_sparse_support_is_explicit_not_dropped(toy):
    nside, g, q = toy
    r, a = account_patch(
        np.ones(len(q)), q, g, 109.375, nside, "quadratic10", np.ones(len(q), bool)
    )
    assert r["status"] == "unsupported" and "fewer pixels" in r["reason"]
    assert len(a["data_k"]) == len(q)


def test_neighbour_mask_and_sparse_coordinate_join():
    v = np.eye(3)
    np.testing.assert_array_equal(neighbour_mask(v, [v[0]]), [True, False, False])
    np.testing.assert_array_equal(
        sample_sparse([8, 2], [10.0, 5.0], [2, 3, 8]), [5, 0, 10]
    )


def test_icrs_frame_and_independent_galactic_centre_check():
    r = dict(
        frame="ICRS",
        catalogue_epoch_label="ep=J2000",
        ra_deg=266.4051,
        dec_deg=-28.936175,
    )
    vec, gal = coordinates(r)
    assert np.linalg.norm(vec) == pytest.approx(1)
    assert min(gal["l_deg"], 360 - gal["l_deg"]) < 0.001 and abs(gal["b_deg"]) < 0.001
    with pytest.raises(ValueError):
        coordinates(r | {"frame": "FK5"})


def test_closure_catches_single_pixel_change():
    maps = {64: np.arange(12.0, dtype="float32"), 8: np.ones(12, dtype="float32")}
    a, b = maps[64].astype(float), maps[8].astype(float)
    assert closure(maps, a, b, 109.375)["pass"]
    a[3] += 1e-9
    assert not closure(maps, a, b, 109.375)["pass"]


def test_offline_cache_is_copy_and_preserves_input(tmp_path):
    parent = tmp_path / "parent"
    parent.write_bytes(b"tiny synthetic cache fixture; not a GSM HDF")
    digest = sha256(parent)
    with offline_parent(parent, tmp_path / "private-cache", digest) as cached:
        assert cached != parent and cached.read_bytes() == parent.read_bytes()
    assert sha256(parent) == digest


def test_historical_product_snapshot_and_tamper(tmp_path):
    identity = producer_identity()
    identity["git"]["commit"] = "historical-declared-producer"
    out = tmp_path / "product"
    out.mkdir()
    snapshot_producer(out, identity)
    write_json(out / "RUN_STARTED.json", dict(producer=identity, inputs={}))
    finalize(out, identity, {}, {"label": "synthetic identity check"})
    manifest, _ = read_product(out)
    assert manifest["producer"]["git"]["commit"] == "historical-declared-producer"
    (out / "producer/math.py").write_text("changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        read_product(out)


def test_conversion_matches_actual_pinned_loader_helper():
    from pygdsm.gsm16 import K_CMB2MJysr

    nu = np.array([50e6, 109.375e6, 234.375e6])
    np.testing.assert_allclose(
        kcmb_sr_per_jy(nu), 1e-6 / K_CMB2MJysr(1.0, nu), rtol=1e-11
    )


@pytest.mark.parametrize("method", METHODS)
def test_runtime_injection_check_on_toy_operator(toy, method):
    from compact_source_d5.run import injection_check

    nside, geometry, q = toy
    result = injection_check(
        q, geometry, 109.375, nside, method, np.zeros(q.shape, bool)
    )
    assert result["pass_one_percent"] and result["fractional_error"] < 0.01


def test_figure_decimal_frequency_does_not_overwrite_methods(tmp_path):
    from compact_source_d5.plots import comparison

    g = dict(x_deg=np.array([0.0, 1.0, -1.0]), y_deg=np.array([0.0, -1.0, 1.0]))
    for method in ["plane8", "quadratic10"]:
        target = tmp_path / ("CygA_109.375_" + method)
        comparison(
            g,
            np.array([2.0, 3.0, 4.0]),
            np.array([2.1, 2.9, 4.0]),
            target,
            "Synthetic plot fixture",
            "Toy coordinate/filename check.",
        )
        assert (tmp_path / (target.name + ".png")).is_file()
        assert (tmp_path / (target.name + ".txt")).is_file()
    assert len(list(tmp_path.glob("*.png"))) == 2
