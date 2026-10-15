"""Independent polynomial and boundary checks for position interpolation."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from eigsim.position import PositionHorizonGrid  # noqa: E402

AXES = (np.linspace(-2, 2, 5), np.linspace(1, 3, 5), np.linspace(10, 18, 5))
AZ = np.array([0.0, 0.7, 2.4, 5.9])
COEFFICIENTS = np.array(
    [
        [0.04, -0.02, 0.003],
        [-0.01, 0.06, 0.002],
        [0.03, 0.01, -0.004],
        [0.01, -0.03, 0.005],
    ]
)


def affine(p):
    return 0.2 + np.asarray(p) @ COEFFICIENTS.T


def samples(fn, axes=AXES):
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return fn(points)


def make_grid(method="cubic", fn=affine, axes=AXES):
    return PositionHorizonGrid(axes, AZ, samples(fn, axes), method=method)


@pytest.mark.parametrize("method", ["linear", "cubic"])
def test_all_nodes_reproduce_general_values(method):
    rng = np.random.default_rng(790)
    alpha = rng.uniform(-0.8, 0.8, (5, 5, 5, len(AZ)))
    grid = PositionHorizonGrid(AXES, AZ, alpha, method=method)
    points = np.stack(np.meshgrid(*AXES, indexing="ij"), axis=-1).reshape(-1, 3)
    result = jax.jit(jax.vmap(grid))(points)
    np.testing.assert_allclose(result, alpha.reshape(-1, len(AZ)), atol=2e-14)


@pytest.mark.parametrize("method", ["linear", "cubic"])
def test_affine_values_and_exact_gradients_on_unequal_axes(method):
    grid = make_grid(method)
    # Include all corners, internal nodes and off-grid positions. The distinct
    # slopes and spacings detect axis permutations and missing metre factors.
    corners = np.array(np.meshgrid(*[(a[0], a[-1]) for a in AXES])).T.reshape(-1, 3)
    points = np.concatenate([corners, [[-0.37, 1.62, 14.3], [0.0, 2.0, 14.0]]])
    values = jax.jit(jax.vmap(grid))(points)
    gradients = jax.jit(jax.vmap(jax.jacfwd(grid)))(points)
    np.testing.assert_allclose(values, affine(points), atol=2e-14)
    np.testing.assert_allclose(
        gradients, np.broadcast_to(COEFFICIENTS, gradients.shape), atol=2e-14
    )
    direction = jnp.array([0.3, -0.7, 1.2])
    _, tangent = jax.jvp(grid, (jnp.asarray(points[-2]),), (direction,))
    np.testing.assert_allclose(tangent, COEFFICIENTS @ direction, atol=2e-14)


@pytest.mark.parametrize("method", ["linear", "cubic"])
def test_constant_values_and_zero_derivatives(method):
    grid = make_grid(method, lambda p: np.full(p.shape[:-1] + (len(AZ),), 0.3))
    p = jnp.array([-1.3, 1.2, 10.1])
    np.testing.assert_allclose(grid(p), 0.3, atol=2e-15)
    np.testing.assert_allclose(jax.jacfwd(grid)(p), 0.0, atol=2e-15)


def quadratic(p):
    e, n, u = np.moveaxis(p, -1, 0)
    value = 0.001 * e**2 + 0.002 * n**2 + 0.0001 * u**2 + 0.0003 * e * n
    return value[..., None] * np.array([1.0, 2.0, -1.0, 0.5])


def test_cubic_interior_quadratic_values_and_analytic_derivative():
    grid = make_grid(fn=quadratic)
    p = jnp.array([-0.3, 1.8, 13.4])  # Each stencil lies wholly inside its axis.
    value, tangent = jax.jvp(grid, (p,), (jnp.array([0.2, -0.5, 1.1]),))
    grad = np.array(
        [0.002 * p[0] + 0.0003 * p[1], 0.004 * p[1] + 0.0003 * p[0], 0.0002 * p[2]]
    )
    expected = (grad @ np.array([0.2, -0.5, 1.1])) * np.array([1, 2, -1, 0.5])
    np.testing.assert_allclose(value, quadratic(np.asarray(p)), atol=1e-14)
    np.testing.assert_allclose(tangent, expected, atol=1e-14)


@pytest.mark.parametrize("method", ["linear", "cubic"])
def test_off_node_jvp_matches_finite_difference(method):
    grid = make_grid(method, quadratic)
    p = jnp.array([-1.3, 1.8, 15.4])
    direction = jnp.array([0.4, -0.3, 0.7])
    _, tangent = jax.jvp(grid, (p,), (direction,))
    h = 1e-5
    numerical = (grid(p + h * direction) - grid(p - h * direction)) / (2 * h)
    np.testing.assert_allclose(tangent, numerical, atol=1e-11, rtol=1e-8)


def test_cubic_joins_are_c1_but_linear_slopes_jump():
    cubic = make_grid("cubic", quadratic)
    linear = make_grid("linear", quadratic)
    node = np.array([0.0, 2.0, 14.0])
    for axis in range(3):
        delta = np.eye(3)[axis] * 1e-7
        left = jax.jacfwd(cubic)(node - delta)
        right = jax.jacfwd(cubic)(node + delta)
        exact = jax.jacfwd(cubic)(node)
        np.testing.assert_allclose(left, right, atol=2e-9)
        np.testing.assert_allclose(exact, (left + right) / 2, atol=2e-9)
        jump = jax.jacfwd(linear)(node + delta) - jax.jacfwd(linear)(node - delta)
        assert np.max(np.abs(jump[:, axis])) > 1e-4


def test_boundary_ghosts_give_endpoint_secant_slopes():
    grid = make_grid(fn=quadratic)
    centre = np.array([0.0, 2.0, 14.0])
    for axis, coords in enumerate(AXES):
        for edge, neighbour in [(0, 1), (-1, -2)]:
            p = centre.copy()
            p[axis] = coords[edge]
            q = centre.copy()
            q[axis] = coords[neighbour]
            expected = (quadratic(p) - quadratic(q)) / (
                coords[edge] - coords[neighbour]
            )
            np.testing.assert_allclose(
                jax.jacfwd(grid)(p)[:, axis], expected, atol=1e-14
            )


def test_two_nodes_per_axis_cubic_reduces_to_trilinear():
    axes = (np.array([-1.0, 2.0]), np.array([2.0, 2.5]), np.array([4.0, 9.0]))
    data = np.random.default_rng(421).uniform(-1, 1, (2, 2, 2, len(AZ)))
    cubic = PositionHorizonGrid(axes, AZ, data)
    linear = PositionHorizonGrid(axes, AZ, data, method="linear")
    p = jnp.array([0.1, 2.2, 7.1])
    np.testing.assert_allclose(cubic(p), linear(p), atol=2e-14)
    np.testing.assert_allclose(jax.jacfwd(cubic)(p), jax.jacfwd(linear)(p), atol=2e-14)


def test_cubic_overshoot_is_not_clipped():
    axes = (np.arange(4.0), np.arange(2.0), np.arange(2.0))
    data = np.broadcast_to(
        np.array([0, 1.55, 1.55, 0])[:, None, None, None], (4, 2, 2, len(AZ))
    )
    grid = PositionHorizonGrid(axes, AZ, data)
    assert np.all(np.asarray(grid([1.5, 0.5, 0.5])) > np.pi / 2)


@pytest.mark.parametrize(
    "position",
    [
        [-2.0001, 2, 14],
        [0, 3.0001, 14],
        [0, 2, 18.0001],
        [np.nan, 2, 14],
        [0, np.inf, 14],
    ],
)
def test_invalid_concrete_queries_raise_but_traced_queries_are_nan(position):
    grid = make_grid()
    assert not bool(grid.contains(position))
    with pytest.raises(ValueError, match="position"):
        grid(position)
    assert np.isnan(np.asarray(jax.jit(grid)(jnp.array(position)))).all()


def test_vmap_domain_mask_and_contains_include_boundaries():
    grid = make_grid()
    positions = jnp.array([[-2, 1, 10], [2, 3, 18], [0, 2, 14], [0, 3.1, 14]])
    np.testing.assert_array_equal(
        jax.jit(jax.vmap(grid.contains))(positions), [True, True, True, False]
    )
    result = jax.jit(jax.vmap(grid))(positions)
    np.testing.assert_allclose(
        result[:3], affine(np.asarray(positions[:3])), atol=2e-14
    )
    assert np.isnan(np.asarray(result[3])).all()


@pytest.mark.parametrize("position", [[0, 1], [[0, 2, 14]], [0, 2, 14, 1]])
def test_wrong_position_shape_raises(position):
    grid = make_grid()
    for call in [grid, grid.contains, jax.jit(grid)]:
        with pytest.raises(ValueError, match="shape"):
            call(jnp.array(position))


@pytest.mark.parametrize(
    "axis",
    [
        np.array([0.0]),
        np.array([[0.0, 1.0]]),
        np.array([0.0, 1.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0, 2.1]),
        np.array([0.0, np.nan]),
        np.array([0.0, np.inf]),
        np.array([0.0, 1j]),
    ],
)
def test_invalid_axis_rejected(axis):
    with pytest.raises(ValueError, match="axes_m"):
        PositionHorizonGrid((axis, AXES[1], AXES[2]), AZ, samples(affine))


@pytest.mark.parametrize("axes", [AXES[:2], AXES + (AXES[0],)])
def test_exactly_three_axes_required(axes):
    with pytest.raises(ValueError, match="axes_m"):
        PositionHorizonGrid(axes, AZ, samples(affine))


@pytest.mark.parametrize(
    "az",
    [
        np.array([]),
        np.array([0.0]),
        np.array([[0.0, 1.0]]),
        np.array([0.1, 0.1, 1.0, 2.0]),
        AZ[::-1],
        np.array([-0.1, 0.7, 2.4, 5.9]),
        np.array([0.0, 0.7, 2.4, 2 * np.pi]),
        np.array([0.0, 0.7, 2.4, np.nan]),
    ],
)
def test_invalid_azimuth_rejected(az):
    with pytest.raises(ValueError, match="azimuth_rad"):
        PositionHorizonGrid(AXES, az, samples(affine))


@pytest.mark.parametrize("kind", ["shape", "nan", "inf", "high", "low", "complex"])
def test_invalid_horizon_grid_rejected(kind):
    data = samples(affine)
    if kind == "shape":
        data = data[:, :, :-1]
    elif kind == "complex":
        data = data.astype(complex) + 0.1j
    else:
        data[0, 0, 0, 0] = {
            "nan": np.nan,
            "inf": np.inf,
            "high": np.pi / 2 + 0.01,
            "low": -np.pi / 2 - 0.01,
        }[kind]
    with pytest.raises(ValueError, match="alpha_rad"):
        PositionHorizonGrid(AXES, AZ, data)


def test_invalid_method_rejected():
    with pytest.raises(ValueError, match="method"):
        make_grid("nearest")


def test_input_arrays_are_copied_and_fixed():
    axes = tuple(a.copy() for a in AXES)
    az = AZ.copy()
    data = samples(affine)
    grid = PositionHorizonGrid(axes, az, data)
    axes[0][0] = -99
    az[0] = 0.1
    data[:] = 0
    np.testing.assert_array_equal(grid.axes_m[0], AXES[0])
    np.testing.assert_array_equal(grid.azimuth_rad, AZ)
    np.testing.assert_allclose(grid([0, 2, 14]), affine([0, 2, 14]), atol=2e-14)
    with pytest.raises(ValueError):
        grid.azimuth_rad[0] = 0.2


def test_large_coordinate_offset_retains_regular_axis_validation():
    axes = (500000 + np.arange(5) * 0.1, AXES[1], AXES[2])
    grid = PositionHorizonGrid(axes, AZ, np.zeros((5, 5, 5, len(AZ))))
    np.testing.assert_array_equal(grid([500000.15, 2, 14]), 0.0)


def test_public_export():
    from eigsim import PositionHorizonGrid as PublicGrid

    assert PublicGrid is PositionHorizonGrid


def test_composes_with_horizon_weights_and_reverse_mode():
    from eigsim import open_sky_weight

    grid = make_grid()

    def open_weight(position):
        return open_sky_weight(grid(position), grid.azimuth_rad, 4, sub=3).sum()

    p = jnp.array([-0.37, 1.62, 14.3])
    gradient = jax.jit(jax.grad(open_weight))(p)
    assert np.all(np.isfinite(gradient))
    assert np.max(np.abs(gradient)) > 1e-3
    for axis in range(3):
        delta = jnp.eye(3)[axis] * 1e-5
        expected = (open_weight(p + delta) - open_weight(p - delta)) / 2e-5
        np.testing.assert_allclose(gradient[axis], expected, rtol=1e-7, atol=1e-9)


def test_complex_position_rejected():
    grid = make_grid()
    with pytest.raises(ValueError, match="real"):
        grid([1j, 2, 14])
