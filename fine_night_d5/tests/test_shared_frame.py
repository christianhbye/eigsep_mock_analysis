"""Normalization invariance independent of heavy simulation libraries."""

import numpy as np
import pytest
from fine_night_d5.shared_frame import array_identity, rescale_beam


def test_per_frequency_gauge_preserves_sky_and_ground_response():
    rng = np.random.default_rng(2)
    beam = rng.uniform(0.1, 2, (3, 4, 7))
    w = rng.uniform(0.1, 1, (4, 7))
    horizon = rng.uniform(0, 1, (4, 7))
    sky = rng.uniform(100, 3000, (3, 4, 7))
    norm = np.sum(beam * w, axis=(1, 2))
    ref = np.array([3.0, 8.0, 2.0])
    scaled = rescale_beam(beam, norm, ref)
    for value in [horizon, horizon * sky]:
        original = np.sum(beam * w * value, axis=(1, 2)) / norm
        reused = np.sum(scaled * w * value, axis=(1, 2)) / ref
        np.testing.assert_allclose(original, reused, rtol=1e-15, atol=1e-12)
    assert array_identity(beam) != array_identity(scaled)
    with pytest.raises(ValueError):
        rescale_beam(beam, np.array([1.0, 0.0, 2.0]), ref)
