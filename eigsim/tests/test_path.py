"""Tests for simulate_path, eigsim's D5 path mode (interface spec § 7)."""

import jax

jax.config.update("jax_enable_x64", True)

import croissant as cro  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402, F401
import s2fft  # noqa: E402
from astropy.time import Time  # noqa: E402
from eigsim.config import load_config  # noqa: E402
from eigsim.simulate import simulate  # noqa: E402

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
