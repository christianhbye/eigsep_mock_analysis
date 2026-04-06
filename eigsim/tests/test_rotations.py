"""Tests for eigsim.rotations."""

import jax

jax.config.update("jax_enable_x64", True)

import croissant as cro  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import s2fft  # noqa: E402
from eigsim.rotations import (  # noqa: E402
    beam_to_alm,
    drive_rotation_matrix,
    rotate_alm_to_beam,
    rotate_beam_data,
    rotation_matrix_x,
    rotation_matrix_z,
)

LMAX = 16
L = LMAX + 1
SAMPLING = "mwss"
NTHETA = L + 1  # 18
NPHI = 2 * L  # 34
FREQS_MHZ = np.array([100.0])


class TestElementaryRotations:
    """Sanity checks for Rx and Rz."""

    def test_identity_x(self):
        np.testing.assert_allclose(rotation_matrix_x(0), np.eye(3), atol=1e-15)

    def test_identity_z(self):
        np.testing.assert_allclose(rotation_matrix_z(0), np.eye(3), atol=1e-15)

    @pytest.mark.parametrize("angle", [0.3, np.pi / 4, np.pi, -1.2])
    def test_orthogonality_x(self, angle):
        R = rotation_matrix_x(angle)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)

    @pytest.mark.parametrize("angle", [0.3, np.pi / 4, np.pi, -1.2])
    def test_orthogonality_z(self, angle):
        R = rotation_matrix_z(angle)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)

    @pytest.mark.parametrize("angle", [0.3, np.pi / 4, np.pi, -1.2])
    def test_det_one_x(self, angle):
        assert np.isclose(np.linalg.det(rotation_matrix_x(angle)), 1.0)

    @pytest.mark.parametrize("angle", [0.3, np.pi / 4, np.pi, -1.2])
    def test_det_one_z(self, angle):
        assert np.isclose(np.linalg.det(rotation_matrix_z(angle)), 1.0)

    def test_rz_90_moves_x_to_y(self):
        """Rz(90deg) should map x-hat to y-hat."""
        R = rotation_matrix_z(np.pi / 2)
        result = R @ np.array([1, 0, 0])
        np.testing.assert_allclose(result, [0, 1, 0], atol=1e-15)

    def test_rx_90_moves_y_to_z(self):
        """Rx(90deg) should map y-hat to z-hat."""
        R = rotation_matrix_x(np.pi / 2)
        result = R @ np.array([0, 1, 0])
        np.testing.assert_allclose(result, [0, 0, 1], atol=1e-15)


class TestDriveRotation:
    """Tests for the combined EIGSEP drive rotation."""

    def test_identity(self):
        R = drive_rotation_matrix(0.0, 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_pure_azimuth(self):
        """Pure azimuth should equal Rz."""
        R = drive_rotation_matrix(0.0, 45.0)
        expected = rotation_matrix_z(np.radians(45.0))
        np.testing.assert_allclose(R, expected, atol=1e-14)

    def test_pure_elevation(self):
        """Pure elevation should equal Rx."""
        R = drive_rotation_matrix(30.0, 0.0)
        expected = rotation_matrix_x(np.radians(30.0))
        np.testing.assert_allclose(R, expected, atol=1e-14)

    def test_orthogonality(self):
        R = drive_rotation_matrix(25.0, 60.0)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)

    def test_det_one(self):
        R = drive_rotation_matrix(25.0, 60.0)
        assert np.isclose(np.linalg.det(R), 1.0)

    def test_composition_order(self):
        """R = Rx(elev) @ Rz(az), not the other way around."""
        elev, az = 20.0, 35.0
        R = drive_rotation_matrix(elev, az)
        expected = rotation_matrix_x(np.radians(elev)) @ rotation_matrix_z(
            np.radians(az)
        )
        np.testing.assert_allclose(R, expected, atol=1e-14)

    def test_zenith_tilts_toward_south(self):
        """Positive elevation tilts zenith toward -y (South, right-hand rule)."""
        R = drive_rotation_matrix(45.0, 0.0)
        z_hat = np.array([0, 0, 1])
        tilted = R @ z_hat
        # Rx(+θ): z tilts toward -y
        assert tilted[1] < 0
        assert tilted[2] < 1

    def test_inverse(self):
        """R(-elev, -az) should undo R(elev, az) when order is reversed."""
        R = drive_rotation_matrix(30.0, 45.0)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)


# ── helpers for SHT tests ───────────────────────────────────────────────


def _make_grids():
    """Return (theta_grid, phi_grid) for MWSS at LMAX."""
    thetas = s2fft.sampling.s2_samples.thetas(L=L, sampling=SAMPLING)
    phis = s2fft.sampling.s2_samples.phis_equiang(L=L, sampling=SAMPLING)
    return np.meshgrid(thetas, phis, indexing="ij")


def _dipole_beam(nfreqs=1):
    """Beam with azimuthal structure."""
    theta_grid, phi_grid = _make_grids()
    pattern = (
        0.5 + 0.3 * np.cos(theta_grid) + 0.2 * np.sin(theta_grid) * np.cos(phi_grid)
    )
    return np.broadcast_to(pattern[None], (nfreqs, NTHETA, NPHI)).copy()


# ── beam_to_alm ─────────────────────────────────────────────────────────


class TestBeamToAlm:
    """Tests for beam_to_alm (forward SHT)."""

    def test_output_shape(self):
        data = _dipole_beam()
        alm = beam_to_alm(data, LMAX, SAMPLING)
        assert alm.shape == (1, L, 2 * L - 1)

    def test_output_shape_multifreq(self):
        data = _dipole_beam(nfreqs=4)
        alm = beam_to_alm(data, LMAX, SAMPLING)
        assert alm.shape == (4, L, 2 * L - 1)

    def test_output_is_jax(self):
        alm = beam_to_alm(_dipole_beam(), LMAX, SAMPLING)
        assert isinstance(alm, jax.Array)

    def test_matches_s2fft_directly(self):
        """beam_to_alm should give the same result as calling s2fft.forward."""
        data = _dipole_beam()
        alm = beam_to_alm(data, LMAX, SAMPLING)
        expected = s2fft.forward(
            jnp.asarray(data[0]),
            L=L,
            spin=0,
            sampling=SAMPLING,
            method="jax",
            reality=True,
        )
        np.testing.assert_allclose(np.asarray(alm[0]), np.asarray(expected), atol=1e-14)

    def test_uniform_beam_is_monopole(self):
        """A uniform beam should have power only in the (0, 0) mode."""
        data = np.ones((1, NTHETA, NPHI))
        alm = beam_to_alm(data, LMAX, SAMPLING)
        # monopole (ell=0, m=0) should be nonzero
        assert np.abs(alm[0, 0, L - 1]) > 0.1
        # all other modes should be negligible
        other = np.asarray(alm[0].at[0, L - 1].set(0.0))
        np.testing.assert_allclose(other, 0.0, atol=1e-12)


# ── rotate_alm_to_beam ──────────────────────────────────────────────────


class TestRotateAlmToBeam:
    """Tests for rotate_alm_to_beam (Wigner-D rotation + inverse SHT)."""

    def test_output_shape(self):
        alm = beam_to_alm(_dipole_beam(), LMAX, SAMPLING)
        rotated = rotate_alm_to_beam(alm, LMAX, SAMPLING, 30.0, 45.0)
        assert rotated.shape == (1, NTHETA, NPHI)

    @pytest.mark.xfail(reason="requires croissant gimbal lock fix (croissant#120)")
    def test_identity_rotation_preserves_data(self):
        """Near-zero rotation should reconstruct the original beam."""
        data = _dipole_beam()
        alm = beam_to_alm(data, LMAX, SAMPLING)
        reconstructed = rotate_alm_to_beam(alm, LMAX, SAMPLING, 0.0, 0.0)
        np.testing.assert_allclose(np.asarray(reconstructed), data, atol=1e-10)

    def test_rotation_changes_data(self):
        """A nontrivial rotation should change the beam pattern."""
        data = _dipole_beam()
        alm = beam_to_alm(data, LMAX, SAMPLING)
        rotated = rotate_alm_to_beam(alm, LMAX, SAMPLING, 30.0, 45.0)
        assert not np.allclose(np.asarray(rotated), data, atol=1e-3)

    def test_preserves_total_power(self):
        """Rotation should preserve the integral of the beam (Parseval)."""
        data = _dipole_beam()
        alm = beam_to_alm(data, LMAX, SAMPLING)
        rotated = rotate_alm_to_beam(alm, LMAX, SAMPLING, 30.0, 45.0)

        beam_orig = cro.Beam(data, FREQS_MHZ, sampling=SAMPLING)
        beam_rot = cro.Beam(np.asarray(rotated), FREQS_MHZ, sampling=SAMPLING)
        np.testing.assert_allclose(
            np.asarray(beam_rot.compute_norm()),
            np.asarray(beam_orig.compute_norm()),
            rtol=1e-8,
        )


# ── rotate_beam_data ────────────────────────────────────────────────────


class TestRotateBeamData:
    """Tests for rotate_beam_data (convenience wrapper)."""

    def test_identity_returns_input(self):
        """Zero angles should return the input as a JAX array."""
        data = _dipole_beam()
        result = rotate_beam_data(data, LMAX, SAMPLING, 0.0, 0.0)
        np.testing.assert_array_equal(np.asarray(result), data)
        assert isinstance(result, jax.Array)

    def test_matches_two_step(self):
        """Should give the same result as beam_to_alm + rotate_alm_to_beam."""
        data = _dipole_beam()
        one_step = rotate_beam_data(data, LMAX, SAMPLING, 25.0, 60.0)
        alm = beam_to_alm(data, LMAX, SAMPLING)
        two_step = rotate_alm_to_beam(alm, LMAX, SAMPLING, 25.0, 60.0)
        np.testing.assert_allclose(
            np.asarray(one_step), np.asarray(two_step), atol=1e-12
        )

    def test_output_shape(self):
        data = _dipole_beam(nfreqs=3)
        result = rotate_beam_data(data, LMAX, SAMPLING, 15.0, 30.0)
        assert result.shape == data.shape

    def test_real_valued_output(self):
        """Rotated beam should be real (spin-0, reality=True)."""
        data = _dipole_beam()
        result = rotate_beam_data(data, LMAX, SAMPLING, 30.0, 45.0)
        assert jnp.isrealobj(result)
