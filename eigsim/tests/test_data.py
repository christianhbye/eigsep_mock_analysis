"""Tests for the packaged configs and the beam files they name."""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import s2fft  # noqa: E402

# Import through the package: ruff sorts `eigsim.data` differently when the
# gitignored eigsim/data/ directory exists, so local and CI lint disagree.
import eigsim  # noqa: E402
from eigsim import load_beam, load_config, load_horizon  # noqa: E402

D5_CHANNEL_MHZ = 250 / 1024
V001_FILES = [
    eigsim.data._DATA_DIR / f
    for f in ("eigsep_bowtie_v001_mwss.npz", "eigsep_bowtie_v001_1mhz_mwss.npz")
]

needs_v001 = pytest.mark.skipif(
    not all(f.exists() for f in V001_FILES),
    reason="v001 beam data absent (run eigsim/scripts/make_bowtie_v001.py)",
)


# ── packaged configs ─────────────────────────────────────────────────────


class TestPackagedConfigs:
    def test_default_is_the_hfss_channels(self):
        """The default grid is the 52 D5 channels the HFSS beam exists at."""
        cfg = load_config()
        chan = np.array(cfg["frequencies"]) / D5_CHANNEL_MHZ
        np.testing.assert_array_equal(chan, np.arange(192, 1009, 16))
        assert cfg["beam"]["file"] == "eigsep_bowtie_v001_mwss.npz"

    def test_default_by_name(self):
        assert load_config("eigsep") == load_config()

    def test_1mhz(self):
        cfg = load_config("eigsep_1mhz")
        np.testing.assert_array_equal(cfg["frequencies"], np.arange(50.0, 247.0))
        assert cfg["beam"]["file"] == "eigsep_bowtie_v001_1mhz_mwss.npz"

    def test_v000_keeps_the_previous_default(self):
        cfg = load_config("eigsep_v000")
        np.testing.assert_array_equal(cfg["frequencies"], np.arange(50.0, 251.0))
        assert cfg["beam"]["file"] == "eigsep_bowtie_v000_mwss.npz"

    @pytest.mark.parametrize("name", ["eigsep_1mhz", "eigsep_v000"])
    def test_only_frequencies_and_beam_differ(self, name):
        base, other = load_config(), load_config(name)
        assert base.keys() == other.keys()
        for key in base.keys() - {"frequencies", "beam"}:
            assert other[key] == base[key], key

    def test_path_still_accepted(self, tmp_path):
        cfg_file = tmp_path / "custom.yaml"
        cfg_file.write_text("frequencies: [100.0]\n")
        assert load_config(cfg_file)["frequencies"] == [100.0]
        assert load_config(str(cfg_file))["frequencies"] == [100.0]

    def test_unknown_name_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("no_such_config")


# ── load_beam ────────────────────────────────────────────────────────────


def _write_beam(path, value):
    np.savez(path, freqs=np.array([1e8]), bm=np.full((1, 3, 4), value), lmax=1)


class TestLoadBeam:
    def test_reads_the_file_the_config_names(self, tmp_path):
        beam = tmp_path / "beam.npz"
        _write_beam(beam, 7.0)
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(f"beam:\n  file: {beam}\n")

        freqs, bm, lmax = load_beam(config=cfg_file)

        np.testing.assert_array_equal(freqs, [1e8])
        assert bm.shape == (1, 3, 4) and np.all(bm == 7.0)
        assert lmax == 1

    def test_path_overrides_config(self, tmp_path):
        named, given = tmp_path / "named.npz", tmp_path / "given.npz"
        _write_beam(named, 1.0)
        _write_beam(given, 2.0)
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(f"beam:\n  file: {named}\n")

        _, bm, _ = load_beam(given, config=cfg_file)

        assert np.all(bm == 2.0)


# ── v001 beam files (need local data) ────────────────────────────────────


@needs_v001
class TestBeamV001:
    @pytest.mark.parametrize("name", ["eigsep", "eigsep_1mhz"])
    def test_frequencies_match_the_config(self, name):
        freqs_hz, bm, _ = load_beam(config=name)
        np.testing.assert_array_equal(freqs_hz / 1e6, load_config(name)["frequencies"])
        assert bm.shape[0] == freqs_hz.size

    @pytest.mark.parametrize("name", ["eigsep", "eigsep_1mhz"])
    def test_grid_matches_the_horizon(self, name):
        _, bm, lmax = load_beam(config=name)
        horizon, h_lmax = load_horizon()
        assert lmax == h_lmax
        assert bm.shape[1:] == horizon.shape

    @pytest.mark.parametrize("name", ["eigsep", "eigsep_1mhz"])
    def test_normalised_to_directivity(self, name):
        """Each channel integrates to 4 pi over the sphere."""
        _, bm, lmax = load_beam(config=name)
        w = np.asarray(
            s2fft.utils.quadrature_jax.quad_weights(L=lmax + 1, sampling="mwss")
        )
        integral = np.einsum("ftp,t->f", bm, w)
        np.testing.assert_allclose(integral, 4 * np.pi, rtol=1e-3)

    def test_spline_passes_through_the_channels(self):
        """125 MHz is on both grids; the interpolant must reproduce it."""
        f_nat, bm_nat, _ = load_beam(config="eigsep")
        f_1mhz, bm_1mhz, _ = load_beam(config="eigsep_1mhz")
        i, j = np.flatnonzero(f_nat == 125e6)[0], np.flatnonzero(f_1mhz == 125e6)[0]
        np.testing.assert_allclose(bm_1mhz[j], bm_nat[i], rtol=0, atol=1e-10)
