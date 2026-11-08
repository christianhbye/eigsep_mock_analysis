"""Independent measured-mask checks; no physical model, GSM, SHT or fit."""

import copy

import numpy as np
import pytest
from fine_night_d5.beam import frequency_operator
from fine_night_d5.join import PHYSICS_SOURCES, context, join_arrays
from fine_night_d5.model_frequency import FrequencyPlan, from_config, verify_payload
from fine_night_d5.streaming import reduce_columns

CHANNELS = [
    283,
    284,
    285,
    291,
    292,
    293,
    603,
    604,
    605,
    611,
    612,
    613,
    843,
    844,
    845,
    851,
    852,
    853,
]


def declaration():
    return {
        "version": 1,
        "kind": "c53-nominal-centres",
        "data_channels": CHANNELS.copy(),
        "evaluation_frequencies_mhz": [70.3125, 148.4375, 207.03125],
        "data_to_evaluation": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
    }


def test_expand_before_each_actual_channel_mask_and_keep18_fit_coordinates():
    plan = FrequencyPlan(CHANNELS, declaration())
    # Distinct row dependence proves repeating a binned mean is incorrect.
    three = np.array([[10, 100, 1000], [20, 200, 2000], [40, 400, 4000]], float)
    evaluated = np.stack([three, three / 10000], -1)
    use = np.ones((3, 18), bool)
    use[:, 0], use[:, 1], use[:, 2] = (
        [True, False, False],
        [False, True, False],
        [True, True, True],
    )
    use[:, 3] = False
    native, means, counts = plan.expand_and_reduce(evaluated, use, np.zeros(3, int), 2)
    assert native.shape == (3, 18, 2)
    for channel in range(18):
        band = channel // 6
        np.testing.assert_array_equal(native[:, channel], evaluated[:, band])
        rows = [i for i in range(3) if use[i, channel]]
        assert counts[0, channel] == len(rows)
        if rows:
            expected = sum(evaluated[i, band] for i in rows) / len(rows)
            np.testing.assert_array_equal(means[0, channel], expected)
        else:
            assert np.isnan(means[0, channel]).all()
    np.testing.assert_array_equal(means[0, :3, 0], [10, 20, 70 / 3])
    assert np.isnan(means[1]).all() and (counts[1] == 0).all()
    np.testing.assert_array_equal(plan.channels, CHANNELS)
    np.testing.assert_array_equal(
        plan.data_frequencies, np.array(CHANNELS) * 250 / 1024
    )
    assert len(np.unique(plan.channels)) == 18
    assert len(plan.evaluation_frequencies) == 3
    # Each data-channel coefficient remains independent: no averaging of
    # the18 measured columns or tying of their eventual fit parameters.
    independent_parameters = np.arange(1.0, 19.0)
    assert len(np.unique(native[0, :6, 0] * independent_parameters[:6])) == 6


def test_absent_control_default_has_exact_values_keys_and_reduction():
    channels = np.array([283, 603, 843])
    plan = from_config({"channels": channels.tolist()})
    np.testing.assert_array_equal(plan.evaluation_frequencies, channels * 250 / 1024)
    assert plan.identity() is None and plan.payload() == {}
    values = np.arange(24.0).reshape(4, 3, 2)
    use = np.array(
        [
            [True, False, True],
            [False, True, True],
            [True, True, False],
            [True, False, False],
        ]
    )
    bins = np.array([0, 0, 1, 1])
    direct, count = reduce_columns(values, use, bins, 3)
    native, actual, actual_count = plan.expand_and_reduce(values, use, bins, 3)
    assert native is values
    np.testing.assert_array_equal(native, values)
    np.testing.assert_array_equal(actual, direct)
    np.testing.assert_array_equal(actual_count, count)


@pytest.mark.parametrize(
    "bad",
    [
        "null",
        "version",
        "kind",
        "unknown",
        "partial",
        "channel_float",
        "channel_duplicate",
        "swapped",
        "map_float",
        "map_bool",
        "map_negative",
        "map_outside",
        "map_short",
        "frequency_nan",
        "frequency_inf",
        "frequency_shift",
        "frequency_duplicate",
    ],
)
def test_ambiguous_changed_or_nonfinite_map_refused(bad):
    d = declaration()
    channels = CHANNELS.copy()
    if bad == "null":
        d = None
    elif bad == "version":
        d["version"] = True
    elif bad == "kind":
        d["kind"] = "nearest"
    elif bad == "unknown":
        d["fallback"] = True
    elif bad == "partial":
        channels = channels[:3]
    elif bad == "channel_float":
        d["data_channels"] = np.array(CHANNELS, float)
    elif bad == "channel_duplicate":
        d["data_channels"][1] = CHANNELS[0]
    elif bad == "swapped":
        d["data_to_evaluation"][0] = 1
    elif bad == "map_float":
        d["data_to_evaluation"] = np.array(d["data_to_evaluation"], float)
    elif bad == "map_bool":
        d["data_to_evaluation"] = np.array(d["data_to_evaluation"], bool)
    elif bad == "map_negative":
        d["data_to_evaluation"][0] = -1
    elif bad == "map_outside":
        d["data_to_evaluation"][-1] = 3
    elif bad == "map_short":
        d["data_to_evaluation"].pop()
    else:
        d["evaluation_frequencies_mhz"][0] = {
            "frequency_nan": np.nan,
            "frequency_inf": np.inf,
            "frequency_shift": 70.0,
            "frequency_duplicate": 148.4375,
        }[bad]
    with pytest.raises(ValueError):
        from_config({"channels": channels, "model_frequency_operator": d})


def test_declared_centres_still_obey_existing_beam_support_without_extrapolation():
    plan = FrequencyPlan(CHANNELS, declaration())
    frequencies = np.array([75.0, 100.0, 180.0, 220.0])
    with pytest.raises(ValueError, match="outside native support"):
        frequency_operator(frequencies, np.ones((4, 2, 2)), plan.evaluation_frequencies)


@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "map",
        "dtype",
        "version",
        "data_frequency",
        "record",
        "input",
        "unexpected_default",
    ],
)
def test_model_payload_cannot_reuse_native_or_changed_mapping_context(bad):
    plan = FrequencyPlan(CHANNELS, declaration())
    config = {"channels": CHANNELS, "model_frequency_operator": declaration()}
    record = {"model_frequency_operator": plan.identity()}
    arrays = {
        "frequencies_mhz": plan.data_frequencies.copy(),
        **{k: v.copy() for k, v in plan.payload().items()},
    }
    verify_payload(config, record, arrays)
    if bad == "missing":
        arrays.pop("model_evaluation_frequencies_mhz")
    elif bad == "map":
        arrays["data_to_model_frequency_index"][0] = 2
    elif bad == "dtype":
        arrays["data_to_model_frequency_index"] = arrays[
            "data_to_model_frequency_index"
        ].astype(np.int32)
    elif bad == "version":
        arrays["model_frequency_operator_version"] = np.array(2)
    elif bad == "data_frequency":
        arrays["frequencies_mhz"] = plan.evaluation_frequencies[plan.mapping]
    elif bad == "record":
        record.clear()
    elif bad == "input":
        config["model_frequency_operator"]["data_to_evaluation"][0] = 2
    else:
        config.pop("model_frequency_operator")
        record.clear()
    with pytest.raises(ValueError):
        verify_payload(config, record, arrays)


def test_context_binds_new_helper_and_cannot_join_away_control_identity():
    config = {
        "channels": CHANNELS,
        "model_frequency_operator": declaration(),
        "_verified_manifest_sha256": "input",
        "native_stage": {"sha256": "native"},
        "pointing_stage": {"sha256": "point"},
        "beam": {"role": "old_symmetric", "sha256": "beam"},
        "runtime_modules": {
            k: k for k in ["eigsim.simulate", "eigsim.rotations", "croissant.simulator"]
        },
    }
    plan = from_config(config)
    record = {
        "input_manifest_sha256": "input",
        "calibration_stage": "native",
        "pointing_stage": "point",
        "beam": {"role": "old_symmetric", "input_sha256": "beam"},
        "source_sha256": {k: k for k in PHYSICS_SOURCES | {"model_frequency.py"}},
        "source_commit": "producer",
        "runtime": {},
        "iers": {},
        "horizon": {},
        "model_frequency_operator": plan.identity(),
    }
    runtime = {
        "dependency_sources": {k: {"sha256": k} for k in config["runtime_modules"]},
        "jax_binaries": {"lib": {"sha256": "binary"}},
    }
    identity = context(config, record, runtime)
    assert identity["configuration"]["model_frequency_operator"] == declaration()
    changed = copy.deepcopy(record)
    changed["source_sha256"]["model_frequency.py"] = "different"
    assert context(config, changed, runtime) != identity
    changed.pop("model_frequency_operator")
    with pytest.raises(ValueError, match="frequency identity"):
        context(config, changed, runtime)
    # A complete18 control is consumed directly. Until a separately scoped
    # chunk protocol exists, assembly must reject rather than drop fields.
    data = {
        k: np.arange(2)
        for k in [
            "native_index",
            "fine_bin_id",
            "source_file",
            "row",
            "input",
            "time_best",
        ]
    }
    data.update(plan.payload())
    with pytest.raises(ValueError, match="cannot be chunk-joined"):
        join_arrays([(data, identity)], CHANNELS)
