"""Regression tests comparing eigsim against direct croissant-sim results."""

import jax

jax.config.update("jax_enable_x64", True)

import croissant as cro  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import s2fft  # noqa: E402
from astropy.time import Time  # noqa: E402
from eigsim.config import load_config  # noqa: E402
from eigsim.rotations import beam_to_alm, rotate_alm_to_beam  # noqa: E402
from eigsim.simulate import make_beam, simulate  # noqa: E402

# ── constants ────────────────────────────────────────────────────────────

LMAX = 16
L = LMAX + 1
SAMPLING = "mwss"
NTHETA = L + 1  # 18 for lmax=16
NPHI = 2 * L  # 34 for lmax=16
FREQS_MHZ = np.array([100.0])
RCVR_TEMP = load_config()["receiver"]["temperature"]

# ── helpers ──────────────────────────────────────────────────────────────


def _make_grids():
    """Return (theta_grid, phi_grid) for MWSS at LMAX."""
    thetas = s2fft.sampling.s2_samples.thetas(L=L, sampling=SAMPLING)
    phis = s2fft.sampling.s2_samples.phis_equiang(L=L, sampling=SAMPLING)
    return np.meshgrid(thetas, phis, indexing="ij")


def _symmetric_beam(nfreqs=1):
    """Beam that depends only on theta (azimuthally symmetric)."""
    theta_grid, _ = _make_grids()
    pattern = np.cos(theta_grid / 2) ** 2
    return np.broadcast_to(pattern[None], (nfreqs, NTHETA, NPHI)).copy()


def _dipole_beam(nfreqs=1):
    """Beam with azimuthal structure."""
    theta_grid, phi_grid = _make_grids()
    pattern = (
        0.5 + 0.3 * np.cos(theta_grid) + 0.2 * np.sin(theta_grid) * np.cos(phi_grid)
    )
    return np.broadcast_to(pattern[None], (nfreqs, NTHETA, NPHI)).copy()


def _make_sky(freqs_mhz=FREQS_MHZ):
    """Non-uniform sky in MWSS, equatorial coords."""
    theta_grid, phi_grid = _make_grids()
    pattern = (
        1000 + 500 * np.cos(theta_grid) + 200 * np.sin(theta_grid) * np.cos(phi_grid)
    )
    data = np.broadcast_to(pattern[None], (len(freqs_mhz), NTHETA, NPHI)).copy()
    return cro.Sky(data, freqs_mhz, sampling=SAMPLING, coord="equatorial")


def _sim_defaults():
    """Simulator kwargs matching eigsim's default config."""
    cfg = load_config()
    loc = cfg["location"]
    return dict(
        lon=loc["lon"],
        lat=loc["lat"],
        alt=loc["alt"],
        world=cfg["world"],
        Tgnd=cfg["ground"]["temperature"],
    )


def _single_time():
    """Single observation time as JD array."""
    return np.array([Time("2026-01-01 00:00:00", scale="utc").jd])


# ── Test 1: unrotated eigsim == direct croissant ─────────────────────────


class TestUnrotatedMatch:
    """eigsim with (0, 0) must exactly match a direct croissant sim."""

    def test_uniform_beam(self):
        beam_data = np.ones((1, NTHETA, NPHI))
        sky = _make_sky()
        times = _single_time()
        defaults = _sim_defaults()

        result = simulate(beam_data, FREQS_MHZ, sky, times, [0.0], [0.0], **defaults)
        t_ant_eigsim = np.asarray(result[0] - RCVR_TEMP)

        beam = cro.Beam(beam_data, FREQS_MHZ, sampling=SAMPLING)
        sim = cro.Simulator(beam, sky, times, FREQS_MHZ, **defaults)
        t_ant_cro = np.asarray(sim.sim())

        np.testing.assert_allclose(t_ant_eigsim, t_ant_cro, atol=1e-10)

    def test_structured_beam(self):
        beam_data = _dipole_beam()
        sky = _make_sky()
        times = _single_time()
        defaults = _sim_defaults()

        result = simulate(beam_data, FREQS_MHZ, sky, times, [0.0], [0.0], **defaults)
        t_ant_eigsim = np.asarray(result[0] - RCVR_TEMP)

        beam = cro.Beam(beam_data, FREQS_MHZ, sampling=SAMPLING)
        sim = cro.Simulator(beam, sky, times, FREQS_MHZ, **defaults)
        t_ant_cro = np.asarray(sim.sim())

        np.testing.assert_allclose(t_ant_eigsim, t_ant_cro, atol=1e-10)

    def test_multifreq(self):
        freqs = np.array([50.0, 100.0, 150.0])
        beam_data = _dipole_beam(nfreqs=3)
        sky = _make_sky(freqs_mhz=freqs)
        times = _single_time()
        defaults = _sim_defaults()

        result = simulate(beam_data, freqs, sky, times, [0.0], [0.0], **defaults)
        assert result.shape == (1, 1, 3)

        t_ant_eigsim = np.asarray(result[0] - RCVR_TEMP)

        beam = cro.Beam(beam_data, freqs, sampling=SAMPLING)
        sim = cro.Simulator(beam, sky, times, freqs, **defaults)
        t_ant_cro = np.asarray(sim.sim())

        np.testing.assert_allclose(t_ant_eigsim, t_ant_cro, atol=1e-10)

    def test_receiver_temp_added(self):
        beam_data = _dipole_beam()
        sky = _make_sky()
        times = _single_time()
        defaults = _sim_defaults()

        result = simulate(beam_data, FREQS_MHZ, sky, times, [0.0], [0.0], **defaults)

        beam = cro.Beam(beam_data, FREQS_MHZ, sampling=SAMPLING)
        sim = cro.Simulator(beam, sky, times, FREQS_MHZ, **defaults)
        t_ant_cro = np.asarray(sim.sim())

        # eigsim adds receiver temp on top of antenna temp
        np.testing.assert_allclose(
            np.asarray(result[0]), t_ant_cro + RCVR_TEMP, atol=1e-10
        )


# ── Test 2: azimuthally symmetric beam invariant under az rotation ───────


class TestAzimuthalSymmetry:
    """A phi-independent beam gives identical output for any azimuth."""

    @pytest.mark.parametrize("az_deg", [30, 90, 180, 270])
    def test_az_invariance(self, az_deg):
        beam_data = _symmetric_beam()
        sky = _make_sky()
        times = _single_time()
        defaults = _sim_defaults()

        ref = simulate(beam_data, FREQS_MHZ, sky, times, [0.0], [0.0], **defaults)
        rotated = simulate(
            beam_data, FREQS_MHZ, sky, times, [0.0], [az_deg], **defaults
        )

        np.testing.assert_allclose(np.asarray(rotated), np.asarray(ref), atol=1e-8)


# ── Test 3: SHT round-trip precision ────────────────────────────────────


class TestSHTRoundTrip:
    """Quantify numerical cost of the SHT forward+inverse pipeline."""

    def test_near_identity_rotation(self):
        """Tiny rotation vs short-circuit should be close."""
        beam_data = _dipole_beam()
        sky = _make_sky()
        times = _single_time()
        defaults = _sim_defaults()

        ref = simulate(beam_data, FREQS_MHZ, sky, times, [0.0], [0.0], **defaults)
        # tiny elevation triggers the SHT path
        rotated = simulate(beam_data, FREQS_MHZ, sky, times, [1e-6], [0.0], **defaults)

        np.testing.assert_allclose(np.asarray(rotated), np.asarray(ref), atol=1e-6)

    def test_beam_data_round_trip(self):
        """beam_to_alm -> rotate_alm_to_beam(~identity) ≈ original."""
        beam_data = _dipole_beam()
        alm = beam_to_alm(beam_data, LMAX, SAMPLING)
        reconstructed = rotate_alm_to_beam(alm, LMAX, SAMPLING, 1e-6, 0.0)

        np.testing.assert_allclose(np.asarray(reconstructed), beam_data, atol=1e-10)


# ── Test 4: eigsim azimuth rotation == croissant beam_rot ────────────────


@pytest.mark.xfail(reason="requires croissant gimbal lock fix (croissant#120)")
class TestAzimuthVsBeamRot:
    """eigsim Wigner-D azimuth rotation should match croissant beam_rot.

    s2fft rotate_flms applies exp(-i*m*alpha) * exp(-i*m*gamma).
    For a pure z-rotation alpha + gamma = azimuth_rad, so the net
    phase is exp(-i*m*azimuth_rad).

    croissant beam_rot applies exp(+i*m*beam_rot_rad).

    Therefore eigsim azimuth = X is equivalent to beam_rot = -X.
    """

    @pytest.mark.parametrize("angle_deg", [30, 45, 90])
    def test_az_vs_beam_rot(self, angle_deg):
        beam_data = _dipole_beam()
        sky = _make_sky()
        times = _single_time()
        defaults = _sim_defaults()

        # eigsim path: rotate beam, then simulate
        result = simulate(
            beam_data,
            FREQS_MHZ,
            sky,
            times,
            [0.0],
            [angle_deg],
            **defaults,
        )
        t_ant_eigsim = np.asarray(result[0] - RCVR_TEMP)

        # croissant path: use beam_rot (opposite sign)
        beam = cro.Beam(beam_data, FREQS_MHZ, sampling=SAMPLING, beam_rot=-angle_deg)
        sim = cro.Simulator(beam, sky, times, FREQS_MHZ, **defaults)
        t_ant_cro = np.asarray(sim.sim())

        np.testing.assert_allclose(t_ant_eigsim, t_ant_cro, atol=1e-4)

    def test_sign_convention(self):
        """Verify that beam_rot = -azimuth matches, not +azimuth."""
        beam_data = _dipole_beam()
        sky = _make_sky()
        times = _single_time()
        defaults = _sim_defaults()
        angle = 45.0

        result = simulate(
            beam_data,
            FREQS_MHZ,
            sky,
            times,
            [0.0],
            [angle],
            **defaults,
        )
        t_ant_eigsim = np.asarray(result[0] - RCVR_TEMP)

        beam_neg = cro.Beam(beam_data, FREQS_MHZ, sampling=SAMPLING, beam_rot=-angle)
        t_neg = np.asarray(
            cro.Simulator(beam_neg, sky, times, FREQS_MHZ, **defaults).sim()
        )

        beam_pos = cro.Beam(beam_data, FREQS_MHZ, sampling=SAMPLING, beam_rot=angle)
        t_pos = np.asarray(
            cro.Simulator(beam_pos, sky, times, FREQS_MHZ, **defaults).sim()
        )

        err_neg = np.max(np.abs(t_ant_eigsim - t_neg))
        err_pos = np.max(np.abs(t_ant_eigsim - t_pos))

        assert err_neg < err_pos, (
            f"beam_rot=-{angle} should match better: "
            f"err_neg={err_neg:.2e}, err_pos={err_pos:.2e}"
        )


# ── Test 5: make_beam unit tests ────────────────────────────────────────


class TestMakeBeam:
    """Unit tests for eigsim.make_beam()."""

    def test_unrotated_data_unchanged(self):
        beam_data = _dipole_beam()
        beam = make_beam(beam_data, FREQS_MHZ, sampling=SAMPLING)
        np.testing.assert_array_equal(
            np.asarray(beam.data), np.asarray(jnp.asarray(beam_data))
        )

    def test_rotated_shape(self):
        beam_data = _dipole_beam()
        beam = make_beam(
            beam_data,
            FREQS_MHZ,
            sampling=SAMPLING,
            elevation_deg=30.0,
            azimuth_deg=45.0,
        )
        assert beam.data.shape == beam_data.shape

    def test_no_beam_rot_set(self):
        beam_data = _dipole_beam()
        beam = make_beam(beam_data, FREQS_MHZ, sampling=SAMPLING)
        assert float(beam.beam_rot) == 0.0


# ── Test 6: simulate output shape/structure ──────────────────────────────


class TestSimulateOutput:
    """Shape and type checks for simulate()."""

    def test_output_shape(self):
        freqs = np.array([100.0, 150.0])
        beam_data = _dipole_beam(nfreqs=2)
        sky = _make_sky(freqs_mhz=freqs)
        times = _single_time()
        defaults = _sim_defaults()

        result = simulate(
            beam_data,
            freqs,
            sky,
            times,
            [0.0, 10.0, 20.0],
            [0.0, 0.0, 0.0],
            **defaults,
        )
        assert result.shape == (3, 1, 2)

    def test_output_is_jax(self):
        beam_data = _dipole_beam()
        sky = _make_sky()
        times = _single_time()
        defaults = _sim_defaults()

        result = simulate(beam_data, FREQS_MHZ, sky, times, [0.0], [0.0], **defaults)
        assert isinstance(result, jax.Array)
