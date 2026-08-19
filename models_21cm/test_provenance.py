"""Unit tests for the self-describing header. Main env: `uv run pytest`."""

import json
from pathlib import Path

import numpy as np
import priors
import provenance
import pytest


def _header_fields():
    return dict(
        user_params={"precisionboost": 3.0},
        cosmo_params={"USE_RELATIVE_VELOCITIES": True, "zmin_CLASS": 4.5},
        astro_fixed={"USE_POPIII": True, "USE_LW_FEEDBACK": True},
        zmin=4.65,
        sampler={
            "kind": "sobol",
            "scramble": True,
            "seed": 20260819,
            "m": 5,
            "n_models": 32,
        },
        varied=[
            {"name": p.name, "transform": p.transform, "lo": p.lo, "hi": p.hi}
            for p in priors.PARAMS
        ],
        interpolation={"method": "CubicSpline", "variable": "log10(z)"},
        selection={"applied": False, "xHI_max": 0.1, "z_ref": 5.9},
        packages={"numpy": np.__version__},
        code={"zeus21": {"commit": "deadbeef"}},
        command_line=["generate.py", "--n-log2", "5"],
    )


def test_header_round_trips_through_json():
    text = provenance.build_header(**_header_fields())
    assert provenance.parse_header(text)["zmin"] == 4.65


def test_header_is_plain_json_needing_no_pickle():
    text = provenance.build_header(**_header_fields())
    assert json.loads(text)["sampler"]["seed"] == 20260819


def test_header_records_its_own_version():
    header = provenance.parse_header(provenance.build_header(**_header_fields()))
    assert header["header_version"] == provenance.HEADER_VERSION


def test_rebuild_params_from_header_alone_matches_the_sampler_bitwise():
    """The sufficiency check: the header must regenerate the draw exactly."""
    header = provenance.parse_header(provenance.build_header(**_header_fields()))
    rebuilt = provenance.rebuild_params(header)
    expected = priors.sample(m=5, seed=20260819)
    assert rebuilt.shape == expected.shape
    assert np.array_equal(rebuilt, expected)


def test_rebuild_params_rejects_an_unknown_sampler():
    header = provenance.parse_header(provenance.build_header(**_header_fields()))
    header["sampler"]["kind"] = "grid"
    with pytest.raises(ValueError, match="grid"):
        provenance.rebuild_params(header)


def test_git_info_reports_this_repository():
    info = provenance.git_info(Path(__file__).resolve().parent)
    assert info["commit"] is not None and len(info["commit"]) == 40


def test_git_info_is_none_outside_a_repository(tmp_path):
    info = provenance.git_info(tmp_path)
    assert info["commit"] is None


def test_package_versions_marks_missing_packages_as_none():
    got = provenance.package_versions(["numpy", "definitely-not-a-package"])
    assert got["numpy"] is not None
    assert got["definitely-not-a-package"] is None
