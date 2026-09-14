"""Tests for simulate_path, eigsim's D5 path mode (interface spec § 7)."""

import jax

jax.config.update("jax_enable_x64", True)

import croissant as cro  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import s2fft  # noqa: E402
from astropy.time import Time  # noqa: E402
from eigsim.config import load_config  # noqa: E402
from eigsim.simulate import (  # noqa: E402
    _run_orientation,
    _setup,
    simulate,
    simulate_path,
)

LMAX = 16
L = LMAX + 1
SAMPLING = "mwss"
NTHETA = L + 1
NPHI = 2 * L
# D5 channels k = 192, 208, 528: non-uniform on purpose.
FREQS_MHZ = np.array([192, 208, 528]) * 250 / 1024
RCVR_TEMP = load_config()["receiver"]["temperature"]


def _grids():
    thetas = s2fft.sampling.s2_samples.thetas(L=L, sampling=SAMPLING)
    phis = s2fft.sampling.s2_samples.phis_equiang(L=L, sampling=SAMPLING)
    return np.meshgrid(thetas, phis, indexing="ij")


def _beam():
    """Chromatic beam with azimuthal structure."""
    th, ph = _grids()
    pattern = 0.5 + 0.3 * np.cos(th) + 0.2 * np.sin(th) * np.cos(ph)
    scale = 1.0 + 0.1 * np.arange(FREQS_MHZ.size)
    return pattern[None] * scale[:, None, None]


def _sky():
    th, ph = _grids()
    pattern = 1000 + 500 * np.cos(th) + 200 * np.sin(th) * np.cos(ph)
    data = np.broadcast_to(pattern[None], (FREQS_MHZ.size, NTHETA, NPHI)).copy()
    return cro.Sky(data, FREQS_MHZ, sampling=SAMPLING, coord="equatorial")


def _times(n):
    """n samples 10 minutes apart on the night of Jul 16/17."""
    t0 = Time("2026-07-17 04:00:00", scale="utc").jd
    return t0 + np.arange(n) * 600.0 / 86400.0


def _croissant_t_ant(beam, sky, times):
    """Direct croissant antenna temperature for the unrotated beam."""
    cfg = load_config()
    loc = cfg["location"]
    sim = cro.Simulator(
        cro.Beam(beam, FREQS_MHZ, sampling=SAMPLING),
        sky,
        times,
        FREQS_MHZ,
        lon=loc["lon"],
        lat=loc["lat"],
        alt=loc["alt"],
        world=cfg["world"],
        Tgnd=cfg["ground"]["temperature"],
    )
    return np.asarray(sim.sim())


class TestSimulateUnchanged:
    def test_unrotated_is_croissant_plus_receiver(self):
        """simulate() keeps its grid output and receiver term (spec § 7)."""
        beam, sky, times = _beam(), _sky(), _times(2)
        got = np.asarray(simulate(beam, FREQS_MHZ, sky, times, [0.0], [0.0]))
        assert got.shape == (1, 2, FREQS_MHZ.size)
        np.testing.assert_allclose(
            got[0], _croissant_t_ant(beam, sky, times) + RCVR_TEMP, rtol=0, atol=1e-10
        )


class TestSimulatePath:
    def test_matches_grid_mode_minus_receiver(self):
        """At matching (orientation, time) samples, path = grid - receiver."""
        beam, sky, times = _beam(), _sky(), _times(4)
        els = np.array([0.0, 30.0, -60.0])
        azs = np.array([0.0, 45.0, 150.0])
        grid = np.asarray(simulate(beam, FREQS_MHZ, sky, times, els, azs))
        ori_of_sample = np.array([2, 0, 1, 2])

        path = np.asarray(
            simulate_path(
                beam, FREQS_MHZ, sky, times, els[ori_of_sample], azs[ori_of_sample]
            )
        )

        want = grid[ori_of_sample, np.arange(times.size)] - RCVR_TEMP
        np.testing.assert_allclose(path, want, rtol=0, atol=1e-10)

    def test_grouping_matches_per_sample_runs(self):
        """Grouping by unique orientation equals evaluating each sample alone."""
        beam, sky, times = _beam(), _sky(), _times(5)
        els = np.array([0.0, 30.0, 0.0, 30.0, 0.0])
        azs = np.array([0.0, 45.0, 0.0, 45.0, 0.0])

        batched = np.asarray(simulate_path(beam, FREQS_MHZ, sky, times, els, azs))
        single = np.stack(
            [
                np.asarray(simulate(beam, FREQS_MHZ, sky, times, [els[i]], [azs[i]]))[
                    0, i
                ]
                for i in range(times.size)
            ]
        )

        np.testing.assert_allclose(batched, single - RCVR_TEMP, rtol=0, atol=1e-10)

    def test_no_receiver_term(self):
        """Uniform sky, open horizon: T_ant is the sky, with nothing added."""
        t0 = 1000.0
        beam = np.ones((FREQS_MHZ.size, NTHETA, NPHI))
        data = np.full((FREQS_MHZ.size, NTHETA, NPHI), t0)
        sky = cro.Sky(data, FREQS_MHZ, sampling=SAMPLING, coord="equatorial")
        open_sky = np.ones((NTHETA, NPHI), dtype=bool)

        got = np.asarray(
            simulate_path(
                beam,
                FREQS_MHZ,
                sky,
                _times(2),
                [0.0, 30.0],
                [0.0, 45.0],
                beam_kw={"horizon": open_sky},
            )
        )

        np.testing.assert_allclose(got, t0, rtol=1e-8)

    def test_shape_and_dtype(self):
        got = simulate_path(_beam(), FREQS_MHZ, _sky(), _times(3), [0.0] * 3, [0.0] * 3)
        assert got.shape == (3, FREQS_MHZ.size)
        assert got.dtype == np.float64

    @pytest.mark.parametrize(
        "els, azs", [([0.0, 0.0], [0.0, 0.0, 0.0]), ([0.0, 0.0, 0.0], [0.0])]
    )
    def test_length_mismatch_raises(self, els, azs):
        with pytest.raises(ValueError, match="one orientation per time"):
            simulate_path(_beam(), FREQS_MHZ, _sky(), _times(3), els, azs)

    def test_empty_input_raises(self):
        empty = np.array([], dtype=np.float64)
        with pytest.raises(ValueError, match="at least one"):
            simulate_path(_beam(), FREQS_MHZ, _sky(), empty, empty, empty)

    @pytest.mark.parametrize(
        "times_jd, elevations_deg, azimuths_deg",
        [
            ([np.nan, 1.0], [0.0, 0.0], [0.0, 0.0]),
            ([0.0, 1.0], [np.nan, 0.0], [0.0, 0.0]),
            ([0.0, 1.0], [0.0, 0.0], [0.0, np.inf]),
        ],
        ids=["nan-times_jd", "nan-elevations_deg", "inf-azimuths_deg"],
    )
    def test_non_finite_input_raises(self, times_jd, elevations_deg, azimuths_deg):
        with pytest.raises(ValueError, match="finite"):
            simulate_path(
                _beam(), FREQS_MHZ, _sky(), times_jd, elevations_deg, azimuths_deg
            )

    def test_orientation_graph_compiles_once_across_group_sizes(self):
        """The orientation graph must not retrace per group size.

        simulate_path() calls _run_orientation() once per group, and
        groups can have different numbers of times. The orientation
        graph itself must be time-independent so it compiles once per
        call regardless of group size; only croissant's convolve()
        should specialise on the number of times.
        """
        setup = _setup(
            _beam(), FREQS_MHZ, _sky(), _times(6), None, "mwss", None, None, {}
        )
        for n in (1, 2, 3):
            _run_orientation(setup, 0.0, 0.0, setup.phases[:n])

        # _cache_size() is JAX's private jit cache counter (number of
        # distinct traces for this jitted function); there is no public
        # API for this, but it is the simplest way to assert "compiled
        # once" from a test.
        assert setup.orient._cache_size() == 1
