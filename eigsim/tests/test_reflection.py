"""Tests for the flat-mirror reflected-sky column (eigsim.reflection)."""

import jax

jax.config.update("jax_enable_x64", True)

import croissant as cro  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import s2fft  # noqa: E402
from astropy.time import Time  # noqa: E402
from eigsim.config import load_config  # noqa: E402
from eigsim.horizon import open_sky_weight  # noqa: E402
from eigsim.reflection import (  # noqa: E402
    P_MIRROR,
    fresnel_normal_incidence,
    mirror_beam,
    mirror_misalignment,
    mirror_open_sky_weight,
    reflected_sky_path,
)
from eigsim.rotations import (  # noqa: E402
    drive_rotation_matrix,
    misalignment_matrix,
)
from eigsim.simulate import simulate_path  # noqa: E402

LMAX = 47
L = LMAX + 1
FREQS = np.array([60.0, 150.0])
LAT = load_config()["location"]["lat"]


def _grid(L_=L):
    th = s2fft.sampling.s2_samples.thetas(L=L_, sampling="mwss")
    ph = s2fft.sampling.s2_samples.phis_equiang(L=L_, sampling="mwss")
    return np.meshgrid(th, ph, indexing="ij")


def _beam_fn(b):
    """Body-frame power beam, azimuthally symmetric about body z."""
    return 1.0 + 0.6 * b[..., 2] + 0.3 * b[..., 2] ** 2


def _beam():
    th, _ = _grid()
    bz = np.cos(th)
    one = _beam_fn(np.stack([0 * bz, 0 * bz, bz], axis=-1))
    return np.stack([one, 1.2 * one])


T0, T1 = 1000.0, 400.0


def _pole_sky():
    """T = T0 + T1 cos(theta_eq): a dipole along the celestial pole."""
    th, _ = _grid()
    d = T0 + T1 * np.cos(th)
    return cro.Sky(np.stack([d, d]), FREQS, sampling="mwss", coord="equatorial")


def _times(n=2):
    t0 = Time("2026-07-17 06:00:00", scale="utc").jd
    return t0 + np.arange(n) * 3600.0 / 86400.0


def _brute(R, mirror, n_l=400):
    """Direct quadrature in ENU of the (reflected) pole-dipole sky.

    The pole is (0, cos lat, sin lat) in ENU at any time. Returns
    (T, f): the beam-weighted temperature and beam fraction over the
    selected directions, normalised by the full-sphere beam integral.
    """
    th = (np.arange(n_l) + 0.5) * np.pi / n_l
    ph = (np.arange(2 * n_l) + 0.5) * np.pi / n_l
    T, PH = np.meshgrid(th, ph, indexing="ij")
    n = np.stack([np.sin(T) * np.cos(PH), np.sin(T) * np.sin(PH), np.cos(T)], -1)
    w = np.sin(T) * (np.pi / n_l) ** 2
    B = _beam_fn(n @ R)  # B(R^-1 n): rows of n @ R are R^T n
    pole = np.array([0.0, np.cos(np.radians(LAT)), np.sin(np.radians(LAT))])
    if mirror:
        sel = n[..., 2] < 0
        src = n @ P_MIRROR
    else:
        sel = n[..., 2] > 0
        src = n
    tsky = T0 + T1 * (src @ pole)
    norm = np.sum(w * B)
    return np.sum(w * B * sel * tsky) / norm, np.sum(w * B * sel) / norm


def _open_horizon():
    az = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    return az, np.zeros_like(az)


def test_mirror_beam_flips_theta():
    th, ph = _grid()
    b = np.stack([np.cos(th) + 0.1 * np.sin(th) * np.cos(ph)])
    f = mirror_beam(b)
    want = -np.cos(th) + 0.1 * np.sin(th) * np.cos(ph)
    np.testing.assert_allclose(f[0], want, atol=1e-15)
    np.testing.assert_array_equal(mirror_beam(f), b)


def test_mirror_beam_rejects_other_sampling():
    with pytest.raises(ValueError):
        mirror_beam(np.ones((1, 10, 20)), sampling="mw")


def test_mirror_identity_for_the_drive():
    """R_d^-1 P = P (P R_d P)^-1 with P R_d P the drive at -el, P mis P."""
    mis = misalignment_matrix(4.0, 140.0)
    R = drive_rotation_matrix(25.0, 70.0, misalignment=mis)
    Rp = drive_rotation_matrix(-25.0, 70.0, misalignment=mirror_misalignment(mis))
    np.testing.assert_allclose(Rp, P_MIRROR @ R @ P_MIRROR, atol=1e-15)
    np.testing.assert_allclose(np.linalg.det(Rp), 1.0)
    assert mirror_misalignment(None) is None


def test_mirror_weight_uses_abs_horizon():
    az = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    alpha = np.radians(5.0 * np.sin(az))
    np.testing.assert_allclose(
        mirror_open_sky_weight(alpha, az, 23),
        open_sky_weight(np.abs(alpha), az, 23),
    )


def test_fresnel_limits():
    # lossless: ((sqrt(e) - 1)/(sqrt(e) + 1))^2
    np.testing.assert_allclose(
        fresnel_normal_incidence([100.0], 9.0, 0.0), [0.25], rtol=1e-12
    )
    r = fresnel_normal_incidence([50.0, 150.0], 7.0, 3.3e-3)
    lossless = ((np.sqrt(7.0) - 1) / (np.sqrt(7.0) + 1)) ** 2
    assert np.all(r > lossless) and r[0] > r[1]  # loss raises R, most at low nu
    assert np.all(r < 1)


@pytest.mark.parametrize("el, az, tilt", [(0.0, 0.0, None), (30.0, 40.0, (5.0, 20.0))])
def test_direct_path_matches_brute_force(el, az, tilt):
    """The convention check: eigsim's direct sky equals the ENU quadrature."""
    mis = None if tilt is None else misalignment_matrix(*tilt)
    a_az, a_al = _open_horizon()
    W = open_sky_weight(a_al, a_az, LMAX)
    got = np.asarray(
        simulate_path(
            _beam(),
            FREQS,
            _pole_sky(),
            _times(),
            [el, el],
            [az, az],
            beam_kw={"horizon": W},
            misalignment=mis,
            Tgnd=0.0,
        )
    )
    want, _ = _brute(drive_rotation_matrix(el, az, misalignment=mis), mirror=False)
    np.testing.assert_allclose(got, want, rtol=2e-3)


@pytest.mark.parametrize("el, az, tilt", [(0.0, 0.0, None), (30.0, 40.0, (5.0, 20.0))])
def test_reflected_path_matches_brute_force(el, az, tilt):
    mis = None if tilt is None else misalignment_matrix(*tilt)
    a_az, a_al = _open_horizon()
    W = mirror_open_sky_weight(a_al, a_az, LMAX)
    s, f = reflected_sky_path(
        _beam(),
        FREQS,
        _pole_sky(),
        _times(),
        [el, el],
        [az, az],
        W,
        misalignment=mis,
    )
    R = drive_rotation_matrix(el, az, misalignment=mis)
    want_t, want_f = _brute(R, mirror=True)
    np.testing.assert_allclose(f, want_f, rtol=2e-3)
    np.testing.assert_allclose(s, want_t, rtol=2e-3)
    # the test discriminates: the ground directions' own (unmirrored) sky
    # differs from the reflected one by far more than the tolerance
    pole = np.array([0.0, np.cos(np.radians(LAT)), np.sin(np.radians(LAT))])
    naive = want_t - T1 * 2 * pole[2] * _mean_up(R) * want_f
    assert abs(naive - want_t) > 20 * 2e-3 * want_t


def _mean_up(R, n_l=200):
    """Beam-weighted mean of -n_U over the lower hemisphere."""
    th = (np.arange(n_l) + 0.5) * np.pi / n_l
    ph = (np.arange(2 * n_l) + 0.5) * np.pi / n_l
    T, PH = np.meshgrid(th, ph, indexing="ij")
    n = np.stack([np.sin(T) * np.cos(PH), np.sin(T) * np.sin(PH), np.cos(T)], -1)
    w = np.sin(T) * _beam_fn(n @ R) * (n[..., 2] < 0)
    return np.sum(w * -n[..., 2]) / np.sum(w)


def test_uniform_sky_gives_the_mirror_fraction():
    th, _ = _grid()
    d = np.full_like(th, 7.0)
    sky = cro.Sky(np.stack([d, d]), FREQS, sampling="mwss", coord="equatorial")
    a_az, a_al = _open_horizon()
    W = mirror_open_sky_weight(a_al, a_az, LMAX)
    s, f = reflected_sky_path(_beam(), FREQS, sky, _times(), [10.0, 10.0], [0, 0], W)
    np.testing.assert_allclose(s, 7.0 * f, rtol=1e-6)
