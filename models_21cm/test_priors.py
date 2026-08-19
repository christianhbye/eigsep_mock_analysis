"""Unit tests for the ensemble parameter space. Main env: `uv run pytest`."""

import numpy as np
import priors
import pytest


def test_fourteen_uniquely_named_params():
    assert len(priors.PARAMS) == 14
    assert len(set(priors.PARAM_NAMES)) == 14


def test_every_param_has_a_known_transform_and_ordered_bounds():
    for p in priors.PARAMS:
        assert p.transform in ("log10", "linear")
        assert p.lo < p.hi


def test_sample_shape_is_two_to_the_m():
    x = priors.sample(m=5, seed=1)
    assert x.shape == (32, 14)


def test_sample_respects_bounds():
    x = priors.sample(m=6, seed=1)
    lo = np.array([p.lo for p in priors.PARAMS])
    hi = np.array([p.hi for p in priors.PARAMS])
    assert np.all(x >= lo) and np.all(x <= hi)


def test_sample_is_reproducible_for_a_seed():
    assert np.array_equal(priors.sample(m=5, seed=7), priors.sample(m=5, seed=7))


def test_sample_differs_between_seeds():
    assert not np.array_equal(priors.sample(m=5, seed=7), priors.sample(m=5, seed=8))


def test_to_astro_kwargs_applies_the_transform():
    row = np.array([p.lo for p in priors.PARAMS])
    kw = priors.to_astro_kwargs(row)
    assert set(kw) == set(priors.PARAM_NAMES)
    for p in priors.PARAMS:
        expected = 10.0**p.lo if p.transform == "log10" else p.lo
        assert kw[p.name] == pytest.approx(expected)


def test_to_native_rejects_an_unknown_transform():
    bad = priors.Param("x", "sqrt", 0.0, 1.0)
    with pytest.raises(ValueError, match="sqrt"):
        bad.to_native(0.5)
