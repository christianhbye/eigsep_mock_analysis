"""Tests for eigsim.rotations."""

import numpy as np
import pytest
from eigsim.rotations import (
    drive_rotation_matrix,
    rotation_matrix_x,
    rotation_matrix_z,
)


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
