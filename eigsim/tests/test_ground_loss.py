"""Tests for ground-fraction computation and ground-loss correction."""

import jax

jax.config.update("jax_enable_x64", True)

import croissant as cro  # noqa: E402
import numpy as np  # noqa: E402
import s2fft  # noqa: E402
from astropy.time import Time  # noqa: E402
from eigsim.config import load_config  # noqa: E402
from eigsim.simulate import (  # noqa: E402
    compute_fgnd,
    correct_ground_loss,
    simulate,
)

# ── constants ────────────────────────────────────────────────────────────

LMAX = 16
L = LMAX + 1
SAMPLING = "mwss"
NTHETA = L + 1
NPHI = 2 * L
FREQS_MHZ = np.array([100.0])
CFG = load_config()
RCVR_TEMP = CFG["receiver"]["temperature"]
TGND = CFG["ground"]["temperature"]

# ── helpers ──────────────────────────────────────────────────────────────


def _make_grids():
    thetas = s2fft.sampling.s2_samples.thetas(L=L, sampling=SAMPLING)
    phis = s2fft.sampling.s2_samples.phis_equiang(L=L, sampling=SAMPLING)
    return np.meshgrid(thetas, phis, indexing="ij")


def _dipole_beam(nfreqs=1):
    theta_grid, phi_grid = _make_grids()
    pattern = (
        0.5 + 0.3 * np.cos(theta_grid) + 0.2 * np.sin(theta_grid) * np.cos(phi_grid)
    )
    return np.broadcast_to(pattern[None], (nfreqs, NTHETA, NPHI)).copy()


def _ring_horizon(theta_max_deg=60.0):
    """Open above theta_max_deg, blocked below (True = open sky)."""
    theta_grid, _ = _make_grids()
    return theta_grid <= np.radians(theta_max_deg)


def _uniform_sky(t0=1000.0, freqs_mhz=FREQS_MHZ):
    data = np.full((len(freqs_mhz), NTHETA, NPHI), t0)
    return cro.Sky(data, freqs_mhz, sampling=SAMPLING, coord="equatorial")


def _sim_defaults():
    loc = CFG["location"]
    return dict(
        lon=loc["lon"],
        lat=loc["lat"],
        alt=loc["alt"],
        world=CFG["world"],
        Tgnd=TGND,
    )


def _single_time():
    return np.array([Time("2026-01-01 00:00:00", scale="utc").jd])


# ── compute_fgnd ─────────────────────────────────────────────────────────


class TestComputeFgnd:
    def test_unrotated_matches_croissant(self):
        """At (0, 0) drive angles, fgnd must equal croissant's."""
        beam_data = _dipole_beam()
        horizon = _ring_horizon()

        fgnd = compute_fgnd(
            beam_data,
            FREQS_MHZ,
            [0.0],
            [0.0],
            beam_kw={"horizon": horizon},
        )

        beam = cro.Beam(beam_data, FREQS_MHZ, sampling=SAMPLING, horizon=horizon)
        fgnd_cro = beam.compute_fgnd()

        np.testing.assert_allclose(
            np.asarray(fgnd[0]), np.asarray(fgnd_cro), atol=1e-10
        )

    def test_all_open_horizon_is_zero(self):
        """With no horizon (all open sky), the ground fraction vanishes."""
        beam_data = _dipole_beam()
        horizon = np.ones((NTHETA, NPHI), dtype=bool)

        fgnd = compute_fgnd(
            beam_data,
            FREQS_MHZ,
            [0.0, 30.0],
            [0.0, 45.0],
            beam_kw={"horizon": horizon},
        )

        np.testing.assert_allclose(np.asarray(fgnd), 0.0, atol=1e-10)

    def test_output_shape(self):
        freqs = np.array([50.0, 100.0, 150.0])
        beam_data = _dipole_beam(nfreqs=3)

        fgnd = compute_fgnd(beam_data, freqs, [0.0, 10.0], [0.0, 90.0])

        assert fgnd.shape == (2, 3)

    def test_tilt_increases_ground_fraction(self):
        """Tilting a zenith-pointing beam toward the horizon raises fgnd."""
        beam_data = _dipole_beam()
        horizon = _ring_horizon(90.0)

        fgnd = compute_fgnd(
            beam_data,
            FREQS_MHZ,
            [0.0, 60.0],
            [0.0, 0.0],
            beam_kw={"horizon": horizon},
        )

        assert float(fgnd[1, 0]) > float(fgnd[0, 0])


# ── correct_ground_loss ──────────────────────────────────────────────────


class TestCorrectGroundLoss:
    def test_uniform_sky_recovery(self):
        """Correcting simulate() output recovers a uniform sky exactly."""
        t0 = 1000.0
        beam_data = _dipole_beam()
        horizon = _ring_horizon()
        sky = _uniform_sky(t0)
        times = _single_time()

        t_sys = simulate(
            beam_data,
            FREQS_MHZ,
            sky,
            times,
            [0.0, 30.0],
            [0.0, 45.0],
            beam_kw={"horizon": horizon},
            **_sim_defaults(),
        )
        fgnd = compute_fgnd(
            beam_data,
            FREQS_MHZ,
            [0.0, 30.0],
            [0.0, 45.0],
            beam_kw={"horizon": horizon},
        )

        t_sky = correct_ground_loss(t_sys, fgnd)

        np.testing.assert_allclose(np.asarray(t_sky), t0, rtol=1e-8)

    def test_matches_croissant_formula(self):
        """The wrapper applies croissant's formula after removing t_rcvr."""
        t_sys = np.array([[[500.0, 600.0]]])  # (1 ori, 1 time, 2 freqs)
        fgnd = np.array([[0.1, 0.2]])  # (1 ori, 2 freqs)

        t_sky = correct_ground_loss(t_sys, fgnd, Tgnd=300.0, t_rcvr=50.0)

        expected = (t_sys[:, 0] - np.array([0.1, 0.2]) * 300.0 - 50.0) / np.array(
            [0.9, 0.8]
        )
        np.testing.assert_allclose(np.asarray(t_sky[:, 0]), expected, atol=1e-10)

    def test_defaults_from_config(self):
        """Tgnd and t_rcvr default to the config values."""
        t_sys = np.array([[[500.0]]])
        fgnd = np.array([[0.1]])

        t_default = correct_ground_loss(t_sys, fgnd)
        t_explicit = correct_ground_loss(t_sys, fgnd, Tgnd=TGND, t_rcvr=RCVR_TEMP)

        np.testing.assert_allclose(np.asarray(t_default), np.asarray(t_explicit))
