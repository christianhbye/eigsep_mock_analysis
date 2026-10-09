import numpy as np
import pytest
from eigsep_pointing_sim.controls import (
    inject_motion_loss,
    inject_pot_angle_shift,
    simulate_pot_voltage,
)


def test_motion_loss_sign_and_spread_preserve_other_legs():
    y = np.array([-180, -100, 0, 100, 115, 130, 180.0])
    mask = np.ones(len(y), bool)
    mask[-1] = False
    zero = np.zeros(len(y))
    np.testing.assert_allclose(
        inject_motion_loss(y, zero, mask, 1, start_deg=100, loss_deg=3),
        [0, 0, 0, -3, -3, -3, 0],
    )
    np.testing.assert_allclose(
        inject_motion_loss(-y, zero, mask, -1, start_deg=-100, loss_deg=3),
        [0, 0, 0, 3, 3, 3, 0],
    )
    np.testing.assert_allclose(
        inject_motion_loss(y, zero, mask, 1, start_deg=100, loss_deg=5, width_deg=30),
        [0, 0, 0, 0, -2.5, -5, 0],
    )
    np.testing.assert_array_equal(zero, np.zeros(len(y)))


def test_pot_voltage_perturbation_units_and_reproducible_benchmark():
    voltage = np.array([1, 1.2, 1.4])
    np.testing.assert_allclose(
        inject_pot_angle_shift(voltage, [0, 2, -2], scale_v_per_deg=0.01),
        [1, 1.22, 1.38],
    )
    kwargs = dict(
        az_deg=[-10, 0, 10],
        walk_deg=[0, 2, 4],
        scale_v_per_deg=0.01,
        offset_v=1,
        noise_sd_deg=0.3,
    )
    a = simulate_pot_voltage(rng=np.random.default_rng(963), **kwargs)
    b = simulate_pot_voltage(rng=np.random.default_rng(963), **kwargs)
    np.testing.assert_array_equal(a["voltage"], b["voltage"])
    np.testing.assert_allclose(
        a["voltage"], 1 + 0.01 * (np.array([-10, 2, 14]) + a["noise_deg"])
    )
    with pytest.raises(ValueError, match="scale"):
        inject_pot_angle_shift(voltage, 2, scale_v_per_deg=0)
