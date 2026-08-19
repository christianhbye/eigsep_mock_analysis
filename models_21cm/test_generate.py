"""Unit tests for the parts of the driver that need no Zeus21.

Main env: `uv run pytest models_21cm/test_generate.py`
"""

import json

import generate
import numpy as np
import priors
import pytest


def test_paper_grid_matches_the_spec():
    f = generate.PAPER_FREQS
    assert f.size == 201
    assert f[0] == 50.0 and f[-1] == 250.0
    assert np.allclose(np.diff(f), 1.0)


def test_interpolation_reproduces_a_smooth_function():
    """A cubic spline must recover a smooth curve to far better than the
    0.62 mK foreground floor the ensemble is compared against."""
    z = np.logspace(np.log10(4.65), np.log10(35.0), 102)

    def truth(zz):
        return -100.0 * np.exp(-((np.log(zz) - np.log(18.0)) ** 2) / 0.08)

    got = generate.interpolate_to_grid(z, truth(z)[None, :], generate.PAPER_FREQS)
    z_target = generate.NU21 / generate.PAPER_FREQS - 1.0
    assert np.max(np.abs(got[0] - truth(z_target))) < 1e-2


def test_interpolation_refuses_to_extrapolate():
    z = np.logspace(np.log10(6.0), np.log10(35.0), 50)  # too shallow for 250 MHz
    with pytest.raises(ValueError, match="extrapolate"):
        generate.interpolate_to_grid(z, np.zeros((1, 50)), generate.PAPER_FREQS)


def test_fixed_and_varied_parameters_do_not_collide():
    """A keyword in both ASTRO_FIXED and PARAMS would be passed twice."""
    assert not set(generate.ASTRO_FIXED) & set(priors.PARAM_NAMES)


def test_fixed_config_is_json_serialisable():
    """Both dicts go into the provenance header verbatim."""
    json.dumps([generate.ASTRO_FIXED, generate.COSMO_FIXED])


def test_interpolation_handles_a_batch_of_models():
    z = np.logspace(np.log10(4.65), np.log10(35.0), 102)
    native = np.vstack([np.full(102, 1.0), np.full(102, -2.0)])
    got = generate.interpolate_to_grid(z, native, generate.PAPER_FREQS)
    assert got.shape == (2, 201)
    assert np.allclose(got[0], 1.0) and np.allclose(got[1], -2.0)
