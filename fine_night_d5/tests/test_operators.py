"""Small analytic fixtures; no actual SHT, maps, beam generation or fits."""

import numpy as np
import pytest
from fine_night_d5.time_operator import average_native, join_pointing

from fine_night_d5 import beam


def test_cartesian_symmetry_parity_idempotence_and_phase():
    rng = np.random.default_rng(9)
    e = rng.normal(size=(3, 5, 8)) + 1j * rng.normal(size=(3, 5, 8))
    mirror = np.array([4, 3, 2, 1, 0, 7, 6, 5])
    expected = np.stack(
        [
            (e[0] + e[0][:, mirror]) / 2,
            (e[1] - e[1][:, mirror]) / 2,
            (e[2] - e[2][:, mirror]) / 2,
        ]
    )
    got = beam.body_x_symmetry(e)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_allclose(beam.body_x_symmetry(got), got, rtol=0, atol=0)
    np.testing.assert_allclose(beam.body_x_symmetry(1j * e), 1j * got)
    # Squaring before the projection changes the operator materially.
    assert not np.allclose(np.sum(abs(e) ** 2, axis=0), np.sum(abs(got) ** 2, axis=0))


def test_directivity_integral_and_raw_explicit_control():
    e = np.ones((3, 5, 8), complex)
    weights = np.full(5, 4 * np.pi / (5 * 8))
    primary, _ = beam.normalize_native(e, weights, "body-x")
    control, _ = beam.normalize_native(e, weights, "raw")
    assert np.einsum("tp,t->", primary, weights) == pytest.approx(4 * np.pi)
    np.testing.assert_allclose(primary, control)
    with pytest.raises(ValueError):
        beam.normalize_native(e, weights * 2, "body-x")
    with pytest.raises(ValueError):
        beam.normalize_native(e, weights, "unknown")


@pytest.mark.parametrize(
    "bad",
    [
        np.ones((3, 5, 8)),
        np.ones((2, 5, 8), complex),
        np.ones((3, 5, 7), complex),
    ],
)
def test_bad_complex_axes_rejected(bad):
    with pytest.raises(ValueError):
        beam.body_x_symmetry(bad)


def test_exact_frequency_knots_cubic_and_no_extrapolation():
    f = np.arange(7.0) + 50
    # Not-a-knot reproduces this specified cubic independently.
    maps = (1 + (f - 50) ** 3 / 100)[:, None, None] * np.ones((1, 3, 4))
    requested = np.array([50.0, 51.25, 54.5, 56.0])
    expected = 1 + (requested - 50) ** 3 / 100
    got = beam.frequency_operator(f, maps, requested)
    np.testing.assert_allclose(got[:, 0, 0], expected, rtol=1e-14)
    np.testing.assert_array_equal(beam.frequency_operator(f, maps, f), maps)
    for target in (np.array([49.99]), np.array([56.01])):
        with pytest.raises(ValueError):
            beam.frequency_operator(f, maps, target)
    _, qc = beam.interpolation_qc(f, maps, requested, np.ones(3))
    assert np.max(qc["withheld_max_abs_power_error"]) < 1e-13


def test_negative_spline_values_retained_and_reported():
    f = np.arange(7.0)
    maps = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])[:, None, None] * np.ones(
        (1, 3, 4)
    )
    got, qc = beam.interpolation_qc(f, maps, np.array([1.5]), np.ones(3))
    assert got.min() < 0
    assert qc["requested_negative_pixel_count"][0] == 12


def test_actual_channel_weights_and_model_time_averaging():
    times = np.array([0.0, 1.0, 3.0, 9.0])
    freqs = np.array([70.0, 148.0])
    rotations = np.repeat(np.eye(3)[None], 4, axis=0)
    use = np.array([[1, 1], [1, 0], [0, 1], [1, 0]], bool)
    mapping = np.array([0, 0, 0, 1])

    def evaluate(t, f, r):
        value = t[:, None] ** 2 + f[None, :]
        return np.stack([value, np.ones_like(value)], axis=-1)

    expected = np.array([[(70 + 71) / 2, (148 + 157) / 2], [151, np.nan]])
    for chunk in (1, 2, 4):
        result, count = average_native(
            times,
            freqs,
            rotations,
            use,
            mapping,
            2,
            evaluate,
            chunk_rows=chunk,
        )
        np.testing.assert_allclose(result[:, :, 0], expected)
        np.testing.assert_array_equal(count, [[2, 2], [1, 0]])
    assert result[0, 0, 0] != 70 + ((0 + 1) / 2) ** 2


def test_missing_model_cannot_remove_measured_support():
    def bad(t, f, r):
        return np.full((len(t), len(f), 1), np.nan)

    with pytest.raises(ValueError, match="measured native cell"):
        average_native(
            np.array([0.0]),
            np.array([70.0]),
            np.eye(3)[None],
            np.ones((1, 1), bool),
            np.array([0]),
            1,
            bad,
        )


def test_pointing_exact_native_identity_no_nearest_parent():
    native = dict(
        source_file=np.array(["a", "b"]),
        row=np.array([1, 0]),
        input=np.array(["4", "4"]),
        time_best=np.array([10.0, 11.0]),
    )
    table = {k: v[::-1].copy() for k, v in native.items()}
    table["rotation"] = np.repeat(np.eye(3)[None], 2, axis=0)
    indices, _ = join_pointing(native, table)
    np.testing.assert_array_equal(indices, [1, 0])
    table["time_best"][0] += 1e-9
    with pytest.raises(ValueError, match="time_best"):
        join_pointing(native, table)


def test_native_drive_factorization_uses_exact_outer_mount():
    from fine_night_d5.eigsim_adapter import drive_angles

    a, e = np.radians([17.0, -21.0]), np.radians([38.0, 5.0])
    ca, sa, ce, se, z = np.cos(a), np.sin(a), np.cos(e), np.sin(e), np.zeros(2)
    rotation = np.stack(
        [
            np.stack([ca, -sa, z], -1),
            np.stack([ce * sa, ce * ca, -se], -1),
            np.stack([se * sa, se * ca, ce], -1),
        ],
        -2,
    )
    angle = np.radians(4.0)
    mis = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    el, az = drive_angles(mis @ rotation, mis)
    np.testing.assert_allclose(el, np.degrees(e))
    np.testing.assert_allclose(az, np.degrees(a))
    bad = np.repeat(np.eye(3)[None], 2, axis=0)
    with pytest.raises(ValueError, match="pinned outer-mount"):
        drive_angles(bad, mis)
