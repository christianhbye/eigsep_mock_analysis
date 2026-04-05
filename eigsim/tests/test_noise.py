"""Tests for eigsim.noise."""

import numpy as np
from eigsim.noise import radiometer_noise


class TestRadiometerNoise:
    """Tests for the radiometer noise generator."""

    def test_output_shape(self):
        t_sys = np.full((3, 10, 5), 100.0)
        noise = radiometer_noise(t_sys, 1e6, 10.0, rng=np.random.default_rng(0))
        assert noise.shape == t_sys.shape

    def test_zero_mean(self):
        rng = np.random.default_rng(42)
        t_sys = np.full(100_000, 200.0)
        noise = radiometer_noise(t_sys, 1e6, 10.0, rng=rng)
        assert np.abs(noise.mean()) < 0.5  # generous tolerance

    def test_std_matches_radiometer_equation(self):
        rng = np.random.default_rng(42)
        t_sys_val = 300.0
        delta_freq = 1e6  # 1 MHz
        delta_time = 10.0  # 10 s
        expected_sigma = t_sys_val / np.sqrt(delta_freq * delta_time)

        t_sys = np.full(500_000, t_sys_val)
        noise = radiometer_noise(t_sys, delta_freq, delta_time, rng=rng)
        np.testing.assert_allclose(noise.std(), expected_sigma, rtol=0.01)

    def test_higher_tsys_gives_more_noise(self):
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)
        n = 100_000
        noise_low = radiometer_noise(np.full(n, 100.0), 1e6, 10.0, rng=rng1)
        noise_high = radiometer_noise(np.full(n, 1000.0), 1e6, 10.0, rng=rng2)
        assert noise_high.std() > noise_low.std()

    def test_more_bandwidth_less_noise(self):
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)
        n = 100_000
        t_sys = np.full(n, 300.0)
        noise_narrow = radiometer_noise(t_sys, 1e5, 10.0, rng=rng1)
        noise_wide = radiometer_noise(t_sys, 1e7, 10.0, rng=rng2)
        assert noise_narrow.std() > noise_wide.std()

    def test_more_time_less_noise(self):
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)
        n = 100_000
        t_sys = np.full(n, 300.0)
        noise_short = radiometer_noise(t_sys, 1e6, 1.0, rng=rng1)
        noise_long = radiometer_noise(t_sys, 1e6, 100.0, rng=rng2)
        assert noise_short.std() > noise_long.std()

    def test_scalar_tsys(self):
        noise = radiometer_noise(300.0, 1e6, 10.0, rng=np.random.default_rng(0))
        assert np.ndim(noise) == 0

    def test_reproducible_with_rng(self):
        t_sys = np.full(100, 300.0)
        n1 = radiometer_noise(t_sys, 1e6, 10.0, rng=np.random.default_rng(99))
        n2 = radiometer_noise(t_sys, 1e6, 10.0, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(n1, n2)
