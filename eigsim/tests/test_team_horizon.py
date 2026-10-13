"""The team's D5 horizon (horizon_profiles_v0003) as an eigsim curve.

The team's profile gives elevation against UTM *grid* bearings measured
counter-clockwise from grid East.  eigsim's ``open_sky_weight`` wants
``alpha_h(az)`` against compass azimuth ``az = atan2(E, N)`` in the true
(geographic) frame, ascending on ``[0, 2 pi)``.  The conversion is
``A_true = 90 deg - theta_grid - gamma`` with the grid convergence
``gamma = 1.5208 deg`` at the site (eigsep_analysis issue #8; the team's
memo 008 adds the same ``gamma`` to its bearings).
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from eigsim.horizon import (  # noqa: E402
    GRID_CONVERGENCE_DEG,
    curve_from_grid_bearings,
    grid_bearings_to_az,
    mwss_grid,
    open_sky_weight,
)

import eigsim.data  # noqa: E402

TEAM_FILE = eigsim.data._DATA_DIR / "horizon_d5_v0003.npz"
needs_team_file = pytest.mark.skipif(
    not TEAM_FILE.exists(),
    reason="team horizon absent (run eigsim/scripts/make_horizon_d5_v0003.py)",
)


def test_convergence_is_the_site_value():
    assert GRID_CONVERGENCE_DEG == pytest.approx(1.5208)


def test_grid_east_maps_to_compass_east_less_convergence():
    az = grid_bearings_to_az(np.array([0.0]))
    assert np.degrees(az[0]) == pytest.approx(90.0 - 1.5208)


def test_grid_north_wraps_into_range():
    az = grid_bearings_to_az(np.array([90.0]))
    assert 0.0 <= az[0] < 2 * np.pi
    assert np.degrees(az[0]) == pytest.approx(360.0 - 1.5208)


def test_curve_is_ascending_and_keeps_values_with_their_bearings():
    bearings = np.arange(3600) * 0.1
    elev = np.full(3600, 0.1)
    elev[900] = 0.5  # grid bearing 90 deg, i.e. grid north
    az, alpha = curve_from_grid_bearings(bearings, elev)
    assert np.all(np.diff(az) > 0)
    assert az[0] >= 0.0 and az[-1] < 2 * np.pi
    i = np.argmax(alpha)
    assert alpha[i] == 0.5
    assert np.degrees(az[i]) == pytest.approx(360.0 - 1.5208)


def test_open_sky_weight_accepts_the_converted_curve():
    bearings = np.arange(3600) * 0.1
    az, alpha = curve_from_grid_bearings(bearings, np.full(3600, 0.2))
    W = np.asarray(open_sky_weight(alpha, az, 32))
    # a flat horizon gives the same weight at every azimuth
    assert np.allclose(W, W[:, :1], atol=1e-12)


def test_load_horizon_curve_reads_the_built_format(tmp_path):
    az = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    path = tmp_path / "h.npz"
    np.savez(
        path,
        az=az,
        alpha_h_30m=np.full(8, 0.3),
        alpha_h_87p5m=np.full(8, 0.2),
        alpha_h_91m=np.full(8, 0.1),
        stamp=np.array("marjum-2026-07/horizon_profiles@v0003+a265aae"),
        source_sha256=np.array("0" * 64),
        convergence_deg=np.array(1.5208),
    )
    got_az, got_alpha, info = eigsim.data.load_horizon_curve("87.5m", path=path)
    assert np.array_equal(got_az, az)
    assert np.allclose(got_alpha, 0.2)
    assert info["stamp"].endswith("v0003+a265aae")
    assert info["era"] == "87.5m"
    with pytest.raises(ValueError, match="era"):
        eigsim.data.load_horizon_curve("50m", path=path)


def _load_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "make_horizon_d5_v0003.py"
    spec = importlib.util.spec_from_file_location("make_horizon_d5_v0003", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_team_file(tmp_path):
    bearings = np.arange(3600) * 0.1
    src = tmp_path / "horizon_profiles_v0003.npz"
    np.savez(
        src,
        bearings_deg=bearings,
        elev_rad_30m=np.full(3600, 0.3),
        **{"elev_rad_87.5m": np.full(3600, 0.2)},
        elev_rad_91m=np.where(bearings == 90.0, 0.5, 0.1),
    )
    import json

    (tmp_path / "horizon_profiles_v0003.json").write_text(
        json.dumps(
            {
                "stamp": "marjum-2026-07/horizon_profiles@v0003+a265aae",
                "frame": "UTM raster grid ...; bearings are atan2(dN,dE) degrees",
            }
        )
    )
    return src


def test_build_converts_every_era_and_records_provenance(tmp_path):
    mod = _load_script()
    src = _fake_team_file(tmp_path)
    out = tmp_path / "built.npz"
    mod.build(src, out, expect_sha256=None)
    az, alpha, info = eigsim.data.load_horizon_curve("91m", path=out)
    assert np.all(np.diff(az) > 0)
    assert np.degrees(az[np.argmax(alpha)]) == pytest.approx(360.0 - 1.5208)
    assert np.allclose(eigsim.data.load_horizon_curve("30m", path=out)[1], 0.3)
    assert info["stamp"] == "marjum-2026-07/horizon_profiles@v0003+a265aae"
    assert len(info["source_sha256"]) == 64
    d = np.load(out)
    assert "provenance" in d.files and "description" in d.files


def test_build_refuses_a_different_source(tmp_path):
    mod = _load_script()
    src = _fake_team_file(tmp_path)
    with pytest.raises(ValueError, match="sha256"):
        mod.build(src, tmp_path / "built.npz", expect_sha256="0" * 64)


@needs_team_file
def test_team_curves_are_physical_and_order_by_height():
    thetas, _ = mwss_grid(128)
    w = np.sin(thetas)[:, None]
    open_frac = {}
    for era in ("30m", "87.5m", "91m"):
        az, alpha, info = eigsim.data.load_horizon_curve(era)
        assert az.shape == alpha.shape == (3600,)
        assert np.all(np.abs(alpha) <= np.pi / 2)
        assert info["stamp"] == "marjum-2026-07/horizon_profiles@v0003+a265aae"
        W = np.asarray(open_sky_weight(alpha, az, 128))
        open_frac[era] = float((w * W).sum() / (w * np.ones_like(W)).sum())
    # a higher antenna sees more open sky
    assert open_frac["30m"] < open_frac["87.5m"] < open_frac["91m"]
