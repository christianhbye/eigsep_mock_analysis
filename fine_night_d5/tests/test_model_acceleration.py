"""Incomplete, altered or failed numerical gates cannot select a consumer."""

import copy

import numpy as np
import pytest
from fine_night_d5.model_acceleration import ten_cases


def cases():
    specs = [
        (0, r)
        for r in [
            "raster_hybrid",
            "represented_hfss_control",
            "full_native_hfss_control",
        ]
    ]
    specs += [(1, r) for r in ["raster_hybrid", "represented_hfss_control"]]
    specs += [
        (2, r)
        for r in [
            "represented_hfss_control",
            "penalty_0.01",
            "delete_stripe_9",
            "delete_stripe_10",
            "delete_stripe_12",
        ]
    ]
    return [
        dict(
            chunk=i,
            role=r,
            qc=dict(rows=37697, orientations=43),
            original_operator_closure=dict(
                original_callable=True, max_abs_S_K=0.0, max_abs_fg=0.0
            ),
            maxima={
                k: 0.0
                for k in [
                    "native_S",
                    "native_A",
                    "native_fg",
                    "bin_S",
                    "bin_A",
                    "bin_fg",
                ]
            },
        )
        for i, r in specs
    ]


def test_incomplete_or_missing_native_coverage_blocks_acceleration():
    full = cases()
    ten_cases(full)
    with pytest.raises(ValueError, match="complete ten"):
        ten_cases(full[:-1])
    missing = copy.deepcopy(full)
    missing[4]["qc"]["rows"] -= 1
    with pytest.raises(ValueError, match="coverage"):
        ten_cases(missing)
    altered = copy.deepcopy(full)
    altered[-1]["role"] = "delete_stripe_29"
    with pytest.raises(ValueError, match="complete ten"):
        ten_cases(altered)


def test_native_bin_and_original_operator_limits_all_gate_selection():
    for metric, value in [
        ("bin_A", 1.01e-9),
        ("native_fg", 1.01e-12),
        ("native_S", np.nan),
    ]:
        bad = cases()
        bad[0]["maxima"][metric] = value
        with pytest.raises(ValueError, match="numerical gate failed"):
            ten_cases(bad)
    bad = cases()
    bad[1]["original_operator_closure"]["max_abs_fg"] = np.inf
    with pytest.raises(ValueError, match="original-operator closure"):
        ten_cases(bad)
