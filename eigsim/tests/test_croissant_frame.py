"""Regression for croissant's fixed-pole frame error (christianhbye/croissant#147).

croissant used to compute the sky frame at a call's first time and then
advance time about the J2000 pole instead of the pole of date, so samples
far from ``times_jd[0]`` were simulated in a slightly tilted frame. Since
v5.3.0.dev2 it turns about the pole of date. A correct simulator gives the
same temperature at a time whether that time starts a call or ends a long
one; these tests compare the two.
"""

import jax

jax.config.update("jax_enable_x64", True)

import croissant as cro  # noqa: E402
import numpy as np  # noqa: E402
import s2fft  # noqa: E402
from astropy.time import Time  # noqa: E402
from eigsim.simulate import simulate  # noqa: E402

LMAX = 16
L = LMAX + 1
SAMPLING = "mwss"
FREQS_MHZ = np.array([100.0])
SIDEREAL_DAY_S = cro.constants.sidereal_day["earth"]
ONE_ARCMIN_S = SIDEREAL_DAY_S / (360 * 60)  # Earth turns 1' in ~4 s
T0_JD = Time("2026-07-17 04:00:00", scale="utc").jd


def _grids():
    thetas = s2fft.sampling.s2_samples.thetas(L=L, sampling=SAMPLING)
    phis = s2fft.sampling.s2_samples.phis_equiang(L=L, sampling=SAMPLING)
    return np.meshgrid(thetas, phis, indexing="ij")


def _beam():
    """30 deg Gaussian with azimuthal structure."""
    th, ph = _grids()
    sigma = np.radians(30.0)
    return (np.exp(-(th**2) / (2 * sigma**2)) * (1 + 0.2 * np.cos(2 * ph)))[None]


def _sky():
    """Seeded random sky, so the beam sees structure on all scales."""
    th, _ = _grids()
    rng = np.random.default_rng(0)
    data = 1000 + 100 * rng.standard_normal((FREQS_MHZ.size, *th.shape))
    return cro.Sky(data, FREQS_MHZ, sampling=SAMPLING, coord="equatorial")


def _t_ant(times_jd):
    """Zenith-pointing temperature, shape (n_time, n_freq)."""
    t = simulate(_beam(), FREQS_MHZ, _sky(), np.asarray(times_jd), [0.0], [0.0])
    return np.asarray(t)[0]


def _frame_error(span_s):
    """Late-sample error of a long call, in units of 1' of Earth rotation.

    Compares the temperature at ``T0_JD + span_s`` from a call starting at
    ``T0_JD`` with a fresh call starting at that time, and divides by the
    temperature change over 1' of Earth rotation. The unit is an order of
    magnitude only: a tilt and a rotation of the same angle sweep the beam
    across different sky gradients.
    """
    t1_jd = T0_JD + span_s / 86400.0
    late = _t_ant([T0_JD, t1_jd])[-1]
    fresh = _t_ant([t1_jd, t1_jd + ONE_ARCMIN_S / 86400.0])
    return np.max(np.abs(late - fresh[0])) / np.max(np.abs(fresh[1] - fresh[0]))


class TestCroissantFrame:
    def test_full_sidereal_day_has_no_frame_error(self):
        """Control: one sidereal day is a full turn about any pole.

        Measured 0.035 with croissant 5.2.1.
        """
        assert _frame_error(SIDEREAL_DAY_S) < 1.0

    def test_no_frame_error_after_4_hours(self):
        """Regression for christianhbye/croissant#147.

        Measured 25 with croissant 5.2.1 (8.7' frame error at D5), which
        turned the sky about the J2000 pole; croissant v5.3.0.dev2 turns
        it about the pole of date and this is now well under 1.
        """
        assert _frame_error(4 * 3600) < 1.0
