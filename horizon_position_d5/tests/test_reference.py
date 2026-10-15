"""Independent synthetic controls for the bilinear horizon reference."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from horizon_position_d5.reference import BilinearDEM, trace_horizon


def make_dem(fn, nx=32, ny=32, res=1.0):
    east, north = np.meshgrid(np.arange(nx) * res, np.arange(ny) * res)
    return BilinearDEM(fn(east, north), res)


def test_interpolation_gradient_and_file_coordinates(tmp_path):
    east, north = np.meshgrid((np.arange(8) - 2) * 0.5, (np.arange(6) - 1) * 0.5)
    data = (5 + 2 * east - 3 * north + 0.7 * east * north).astype(np.float32)
    path = tmp_path / "dem.npz"
    np.savez(path, dem=data, res=0.5, e0_px=2, n0_px=1)
    dem = BilinearDEM.from_npz(path)
    assert dem.data.dtype == np.float64
    e, n = np.array([-0.8, 0.1, 1.8]), np.array([-0.3, 0.3, 1.4])
    assert_allclose(dem.interp(e, n), 5 + 2 * e - 3 * n + 0.7 * e * n, atol=5e-7)
    assert_allclose(
        dem.gradient(e, n), np.stack((2 + 0.7 * n, -3 + 0.7 * e), -1), atol=1e-6
    )
    assert_allclose(dem.edge_range(0.0, 0.0, 1.0, 0.0), 2.0)
    assert_allclose(dem.edge_range(0.0, 0.0, 0.0, 1.0), 1.5)
    assert_allclose(dem.edge_range(0.0, 0.0, -1.0, 0.0), 1.0)
    # The helper can evaluate the full last sample, but tracing stops one cell earlier.
    assert dem.interp(2.5, 2.0) == data[-1, -1]


def test_sampled_mode_matches_independent_team_march():
    rng = np.random.default_rng(19)
    values = rng.normal(size=(13, 15)).astype(np.float32).astype(float)
    dem = BilinearDEM(values, 0.5)
    p = np.array([3.1, 2.2, 0.4])
    offsets = np.array([-1.0, 0.0, 1.0])
    bearings = np.array([0.0, 90.0, 180.0, 270.0, 33.3, 89.999999999])
    got = trace_horizon(dem, p, bearings, height_offsets_m=offsets, method="sampled")
    for j, beta in enumerate(np.radians(bearings)):
        ux, uy = np.cos(beta), np.sin(beta)
        limits = []
        for v, c, hi in (
            (ux, p[0], (values.shape[1] - 2) * 0.5),
            (uy, p[1], (values.shape[0] - 2) * 0.5),
        ):
            if v > 1e-12:
                limits.append((hi - c) / v)
            elif v < -1e-12:
                limits.append(-c / v)
        edge = min(limits)
        r = np.arange(0.25, edge, 0.25)
        x, y = (p[0] + r * ux) / 0.5, (p[1] + r * uy) / 0.5
        ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = x - ix, y - iy
        z = (
            (1 - fx) * (1 - fy) * values[iy, ix]
            + fx * (1 - fy) * values[iy, ix + 1]
            + (1 - fx) * fy * values[iy + 1, ix]
            + fx * fy * values[iy + 1, ix + 1]
        )
        zcurv = z - r**2 / (2 * 6371e3)
        angles = np.arctan2(zcurv[None] - (p[2] + offsets)[:, None], r[None])
        winners = np.argmax(angles, axis=1)
        assert_array_equal(got.alpha_rad[:, j], angles[np.arange(3), winners])
        assert_array_equal(got.distance_m[:, j], r[winners])
        assert_array_equal(got.surface_height_m[:, j], z[winners])
        assert_array_equal(got.curvature_height_m[:, j], zcurv[winners])
        assert_array_equal(got.edge_limited[:, j], winners == len(r) - 1)
        assert got.edge_range_m[j] == edge
    assert got.gradient_en.shape == (3, len(bearings), 2)


def test_curvature_produces_exact_interior_maximum_on_flat_terrain():
    dem = make_dem(lambda e, n: np.zeros_like(e))
    p = np.array([12.0, 12.0, 1.0])
    got = trace_horizon(dem, p, [0, 90, 180, 270], earth_radius_m=20)
    assert_allclose(got.distance_m, np.sqrt(40), rtol=0, atol=1e-13)
    assert_allclose(got.alpha_rad, np.arctan(-2 / np.sqrt(40)), rtol=0, atol=1e-14)
    assert not got.edge_limited.any()
    assert_allclose(got.gradient_en, 0)


def test_oblique_bilinear_quadratic_has_interior_stationary_maximum():
    dem = make_dem(lambda e, n: -0.1 * (e - 0.5) * (n - 0.5))
    got = trace_horizon(dem, [0.5, 0.5, 1.0], [45.0], earth_radius_m=np.inf)
    assert_allclose(got.distance_m, np.sqrt(20), rtol=0, atol=2e-14)
    assert_allclose(got.alpha_rad, np.arctan(-2 * np.sqrt(0.05)), rtol=0, atol=2e-15)


def test_continuous_contains_dense_independent_samples():
    rng = np.random.default_rng(792)
    dem = BilinearDEM(rng.uniform(-2, 2, (12, 13)), 0.5)
    p = [2.17, 1.93, 0.3]
    bearings = [0, 23.8, 45, 89.999999999999, 90, 180, 225, 270]
    exact = trace_horizon(dem, p, bearings)
    coarse = trace_horizon(dem, p, bearings, method="sampled", march_step_m=0.001)
    fine = trace_horizon(dem, p, bearings, method="sampled", march_step_m=0.0005)
    assert np.all(exact.alpha_rad >= fine.alpha_rad - 2e-14)
    assert np.all(fine.alpha_rad >= coarse.alpha_rad - 2e-14)
    assert np.max(exact.alpha_rad - fine.alpha_rad) < 0.002


def test_crease_identities_and_e7prime_failure():
    # Analytic crest fixture independent of the ray-maximizing algorithm.
    dem = make_dem(lambda e, n: 3 + 0.3 * n - np.abs(e - 10), nx=62, ny=40)
    p = np.array([0.5, 0.1, 0.0])
    beta = 0.45
    ux, uy = np.cos(beta), np.sin(beta)
    c = np.array([uy, -ux])

    def value(position, angle=beta):
        return trace_horizon(dem, position, [np.degrees(angle)]).alpha_rad[0, 0]

    trace = trace_horizon(dem, p, [np.degrees(beta)])
    radius = (10 - p[0]) / ux
    alpha = np.arctan(
        (3 + 0.3 * (p[1] + radius * uy) - p[2]) / radius - radius / (2 * 6371e3)
    )
    assert_allclose(trace.distance_m, radius, rtol=0, atol=3e-14)
    assert_allclose(trace.alpha_rad, alpha, rtol=0, atol=2e-15)
    step = 1e-5
    fd = np.array(
        [(value(p + step * v) - value(p - step * v)) / (2 * step) for v in np.eye(3)]
    )
    slope = -(value(p, beta + step) - value(p, beta - step)) / (2 * step)
    along = np.sin(alpha) * np.cos(alpha) / radius + np.cos(alpha) ** 2 / 6371e3
    across = slope / radius
    predicted = np.array(
        [
            ux * along + uy * across,
            uy * along - ux * across,
            -(np.cos(alpha) ** 2) / radius,
        ]
    )
    assert_allclose(fd, predicted, rtol=0, atol=2e-10)
    assert_allclose(
        fd,
        [0.025740480671943, 0.024211273400021, -0.080704244681540],
        rtol=0,
        atol=2e-10,
    )
    for side_gradient in ([1.0, 0.3], [-1.0, 0.3]):
        prime = np.cos(alpha) ** 2 * np.dot(side_gradient, c) / radius
        assert abs(prime - across) > 0.02


def test_near_tie_is_not_an_exact_tie_and_nearest_exact_tie_wins():
    for extra, expected_radius in [(0.0, 1.5), (1e-12, 3.5)]:
        data = np.zeros((8, 9))
        data[:, 2] = 1.5
        data[:, 4] = 3.5 * (1 + extra)
        result = trace_horizon(
            BilinearDEM(data, 1.0), [0.5, 2, 0], [0], earth_radius_m=np.inf
        )
        assert result.distance_m[0, 0] == expected_radius


def test_constant_q_and_a_minus_k_zero():
    dem = make_dem(lambda e, n: e - 0.5)
    got = trace_horizon(dem, [0.5, 2, 0], [0], earth_radius_m=np.inf)
    assert got.distance_m[0, 0] == 0.25
    assert got.alpha_rad[0, 0] == np.pi / 4


def test_outer_endpoint_and_sampled_edge_flag_are_explicit():
    dem = make_dem(lambda e, n: np.zeros_like(e), nx=8, ny=8)
    p = [1, 2, 1]
    exact = trace_horizon(dem, p, [0], earth_radius_m=np.inf)
    sampled = trace_horizon(dem, p, [0], method="sampled", earth_radius_m=np.inf)
    assert exact.edge_range_m[0] == 5
    assert exact.distance_m[0, 0] == 5
    assert sampled.distance_m[0, 0] == 4.75
    assert exact.edge_limited[0, 0] and sampled.edge_limited[0, 0]
    assert exact.alpha_rad[0, 0] > sampled.alpha_rad[0, 0]


def test_inner_cutoff_does_not_move_when_march_is_refined():
    dem = make_dem(lambda e, n: np.zeros_like(e))
    for method, step in [("continuous", 0.25), ("sampled", 0.25), ("sampled", 0.1)]:
        trace = trace_horizon(dem, [3, 3, -1], [0], method=method, march_step_m=step)
        assert trace.distance_m[0, 0] == 0.25
    # E4 is deliberately inapplicable at a fixed radial cutoff.
    step = 1e-5
    left = trace_horizon(dem, [3 - step, 3, -1], [0]).alpha_rad[0, 0]
    right = trace_horizon(dem, [3 + step, 3, -1], [0]).alpha_rad[0, 0]
    assert right == left


def test_duplicate_corner_crossings_and_large_coordinate_offsets():
    dem = make_dem(lambda e, n: np.zeros_like(e), nx=12, ny=12)
    for bearing in [45, np.nextafter(45.0, 46.0), 135, 225, 315]:
        got = trace_horizon(dem, [5, 5, 1], [bearing], earth_radius_m=np.inf)
        assert np.isfinite(got.alpha_rad).all()
        assert got.distance_m[0, 0] == got.edge_range_m[0]
    shifted = BilinearDEM(dem.data, 1.0, e0_px=-1_000_000, n0_px=2_000_000)
    a = trace_horizon(dem, [5, 5, 1], [23, 45, 135, 270])
    b = trace_horizon(shifted, [1_000_005, -1_999_995, 1], [23, 45, 135, 270])
    assert_allclose(a.alpha_rad, b.alpha_rad, rtol=0, atol=1e-10)
    assert_allclose(a.distance_m, b.distance_m, rtol=0, atol=1e-9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "bad"},
        {"march_step_m": 0},
        {"inner_cutoff_m": 0},
        {"earth_radius_m": -1},
        {"height_offsets_m": []},
    ],
)
def test_invalid_tracing_inputs(kwargs):
    dem = make_dem(lambda e, n: np.zeros_like(e))
    with pytest.raises(ValueError):
        trace_horizon(dem, [3, 3, 1], [0], **kwargs)


def test_invalid_dem_and_empty_ray_fail_clearly():
    with pytest.raises(ValueError):
        BilinearDEM(np.full((4, 4), np.nan), 1)
    with pytest.raises(ValueError):
        BilinearDEM(np.zeros((4, 4)), 0)
    dem = make_dem(lambda e, n: np.zeros_like(e), nx=8, ny=8)
    with pytest.raises(ValueError):
        trace_horizon(dem, [6, 3, 1], [0])
    with pytest.raises(ValueError):
        trace_horizon(dem, [-1, 3, 1], [0])
    with pytest.raises(ValueError):
        dem.interp(100, 3)


def test_additive_height_offsets_have_independent_continuous_maxima():
    dem = make_dem(lambda e, n: np.zeros_like(e))
    offsets = np.array([-0.75, 0.0, 1.25])
    got = trace_horizon(
        dem, [12, 12, 1], [0, 90, 180, 270], height_offsets_m=offsets, earth_radius_m=20
    )
    heights = 1 + offsets
    expected_r = np.broadcast_to(np.sqrt(40 * heights)[:, None], (3, 4))
    expected_alpha = np.broadcast_to(
        np.arctan(-2 * np.sqrt(heights / 40))[:, None], (3, 4)
    )
    assert_allclose(got.distance_m, expected_r, rtol=0, atol=1e-13)
    assert_allclose(got.alpha_rad, expected_alpha, rtol=0, atol=2e-15)
