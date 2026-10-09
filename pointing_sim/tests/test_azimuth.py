import numpy as np
import pytest
from eigsep_pointing_sim.azimuth import ramp_window_means, simulate_azimuth_window


def test_exact_azimuth_ramp_moments_and_shift():
    edges = np.arange(5.0)
    np.testing.assert_allclose(
        ramp_window_means(edges, 0.25, 1.5), [0.1875, 0.8125, 1, 1]
    )
    out = simulate_azimuth_window(
        edges,
        count_start=0.5,
        duration=1.5,
        tau_s=0.25,
        step=0.113,
        contamination=np.array([0.2, 0.1, -0.1, -0.2]),
    )
    np.testing.assert_allclose(
        out, 0.113 * np.array([0.1875, 0.8125, 1, 1]) + [0.2, 0.1, -0.1, -0.2]
    )
    with pytest.raises(ValueError, match="duration"):
        ramp_window_means(edges, 0, 0)


def test_azimuth_contamination_shape_and_noise_reproducibility():
    kwargs = dict(
        count_start=1,
        duration=1,
        tau_s=0.2,
        step=0.113,
        contamination=np.zeros(4),
        noise_sd=0.002,
    )
    a = simulate_azimuth_window(
        np.arange(5.0), rng=np.random.default_rng(952), **kwargs
    )
    b = simulate_azimuth_window(
        np.arange(5.0), rng=np.random.default_rng(952), **kwargs
    )
    np.testing.assert_array_equal(a, b)
    kwargs["contamination"] = np.zeros(3)
    with pytest.raises(ValueError, match="per integration"):
        simulate_azimuth_window(np.arange(5.0), **kwargs)
