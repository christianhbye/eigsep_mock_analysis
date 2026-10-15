"""Differentiable interpolation of fixed horizon curves over antenna position."""

import jax
import jax.numpy as jnp
import numpy as np


def _real_array(value, name):
    """Copy a concrete, finite real array without discarding complex parts."""
    try:
        array = np.asarray(value)
        if np.iscomplexobj(array):
            raise ValueError("complex values are not supported")
        array = np.array(array, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numbers") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.flags.writeable = False
    return array


def _cubic_stencil(cell, t, size):
    """Catmull-Rom stencil with linear boundary ghosts folded into weights."""
    w0 = -0.5 * t + t**2 - 0.5 * t**3
    w1 = 1.0 - 2.5 * t**2 + 1.5 * t**3
    w2 = 0.5 * t + 2.0 * t**2 - 1.5 * t**3
    w3 = -0.5 * t**2 + 0.5 * t**3
    lower = cell == 0
    upper = cell == size - 2
    # A ghost value is 2*f(edge) - f(neighbour). Folding it into these
    # weights avoids storing a padded copy of the whole 4-D horizon grid.
    # Both corrections apply when size == 2, giving linear interpolation.
    weights = jnp.stack(
        [
            jnp.where(lower, 0.0, w0),
            w1 + jnp.where(lower, 2.0 * w0, 0.0) - jnp.where(upper, w3, 0.0),
            w2 - jnp.where(lower, w0, 0.0) + jnp.where(upper, 2.0 * w3, 0.0),
            jnp.where(upper, 0.0, w3),
        ]
    )
    indices = jnp.clip(cell + jnp.arange(-1, 3), 0, size - 1)
    return indices, weights


class PositionHorizonGrid:
    """A fixed grid of horizon curves with position derivatives through JAX.

    Construct outside ``jit`` from concrete arrays. Query with ``grid(p)``
    for one position, or use ``jax.vmap(grid)`` for several positions. Axes,
    azimuths and sampled horizons are copied at construction; only the query
    position is differentiable through this API.

    Parameters
    ----------
    axes_m : sequence of three 1-D arrays
        East, North and Up coordinates in metres, in the caller's stated
        frame. Each axis has at least two finite, strictly ascending,
        uniformly spaced nodes. Spacing may differ between axes. Regularity
        allows float64 rounding at the scale of the coordinate values.
    azimuth_rad : (n_az,) array
        Fixed true compass azimuths, North towards East, strictly ascending
        in ``[0, 2*pi)`` with at least two samples. Spatial interpolation
        leaves this axis unchanged, ready for :func:`eigsim.open_sky_weight`.
    alpha_rad : (n_e, n_n, n_u, n_az) array
        Finite horizon elevations in radians, within ``[-pi/2, pi/2]`` at
        every node. The first three axes correspond to ``axes_m`` in order.
    method : {"cubic", "linear"}, default "cubic"
        Tensor-product Catmull-Rom cubic or trilinear interpolation. Cubic
        uses linearly extrapolated ghost samples at each boundary, so its
        boundary derivative is the adjacent-node secant. Interior joins
        have continuous first derivatives. Linear derivatives jump at
        spatial cell boundaries.

    Notes
    -----
    The closed training box includes all faces and corners. A concrete
    out-of-box or nonfinite query raises ``ValueError``. During a JAX trace
    it returns a curve of NaNs; use :meth:`contains` to test validity. No
    query is extrapolated outside the box.

    Cubic interpolation can overshoot the sampled elevation range. Outputs
    are not clipped, since clipping changes derivatives. Such an overshoot
    must be checked before passing the curve to a horizon-weight function.
    A C1 interpolated horizon does not make the complete horizon-weight or
    antenna-temperature response C1: their cell boundaries can have kinks.
    Smoothness also does not establish accuracy against retraced terrain.
    """

    def __init__(self, axes_m, azimuth_rad, alpha_rad, method="cubic"):
        if method not in ("linear", "cubic"):
            raise ValueError("method must be 'linear' or 'cubic'")
        try:
            axes = tuple(axes_m)
        except TypeError as exc:
            raise ValueError(
                "axes_m must contain exactly three coordinate axes"
            ) from exc
        if len(axes) != 3:
            raise ValueError("axes_m must contain exactly three coordinate axes")

        checked_axes = []
        steps = []
        for i, axis in enumerate(axes):
            name = f"axes_m[{i}]"
            axis = _real_array(axis, name)
            if axis.ndim != 1 or axis.size < 2:
                raise ValueError(f"{name} must be 1-D with at least two nodes")
            diffs = np.diff(axis)
            if not np.all(np.isfinite(diffs)) or np.any(diffs <= 0):
                raise ValueError(f"{name} must be strictly ascending")
            step = (axis[-1] - axis[0]) / (axis.size - 1)
            atol = 8 * np.finfo(np.float64).eps * max(1.0, np.max(np.abs(axis)))
            if not np.allclose(diffs, step, rtol=1e-10, atol=atol):
                raise ValueError(f"{name} must be uniformly spaced")
            checked_axes.append(axis)
            steps.append(step)

        azimuth = _real_array(azimuth_rad, "azimuth_rad")
        if azimuth.ndim != 1 or azimuth.size < 2:
            raise ValueError("azimuth_rad must be 1-D with at least two samples")
        if np.any(np.diff(azimuth) <= 0) or azimuth[0] < 0 or azimuth[-1] >= 2 * np.pi:
            raise ValueError("azimuth_rad must be strictly ascending within [0, 2*pi)")

        alpha = _real_array(alpha_rad, "alpha_rad")
        shape = tuple(len(a) for a in checked_axes) + (azimuth.size,)
        if alpha.shape != shape:
            raise ValueError(f"alpha_rad must have shape {shape}, got {alpha.shape}")
        if np.any(np.abs(alpha) > np.pi / 2):
            raise ValueError("alpha_rad must lie within [-pi/2, pi/2]")

        self._axes_m = tuple(checked_axes)
        self._azimuth_rad = azimuth
        self._alpha_rad = jnp.asarray(alpha, dtype=jnp.float64)
        self._method = method
        self._lower = np.array([a[0] for a in checked_axes])
        self._upper = np.array([a[-1] for a in checked_axes])
        self._spacing = np.array(steps)

    @property
    def axes_m(self):
        """Read-only East, North and Up coordinate arrays, in metres."""
        return self._axes_m

    @property
    def azimuth_rad(self):
        """Read-only true compass azimuths, in radians."""
        return self._azimuth_rad

    @property
    def alpha_rad(self):
        """Immutable JAX array of sampled horizon elevations, in radians."""
        return self._alpha_rad

    @property
    def method(self):
        """Spatial interpolation method: ``'linear'`` or ``'cubic'``."""
        return self._method

    @staticmethod
    def _position(position):
        if jnp.iscomplexobj(position):
            raise ValueError("position must contain real coordinates")
        position = jnp.asarray(position, dtype=jnp.float64)
        if position.shape != (3,):
            raise ValueError(f"position must have shape (3,), got {position.shape}")
        return position

    def contains(self, position):
        """Return a JAX boolean for a finite position inside the closed box.

        ``position`` has shape ``(3,)`` in East, North, Up order, in metres.
        This predicate supports ``jit`` and ``vmap``. Malformed shapes raise
        ``ValueError``, while NaNs and infinities return ``False``.
        """
        position = self._position(position)
        return jnp.all(
            jnp.isfinite(position)
            & (position >= self._lower)
            & (position <= self._upper)
        )

    def __call__(self, position):
        """Return ``(n_az,)`` horizon elevations at one position in metres.

        This call supports ``jit``, ``vmap`` and differentiation with
        respect to the three position coordinates. Gradients have units
        radians per metre. Gradients of invalid queries are undefined;
        check :meth:`contains` before interpreting a result.
        """
        position = self._position(position)
        if not isinstance(position, jax.core.Tracer):
            concrete = np.asarray(position)
            if not (
                np.all(np.isfinite(concrete))
                and np.all(concrete >= self._lower)
                and np.all(concrete <= self._upper)
            ):
                raise ValueError("position must be finite and inside the grid box")
        valid = self.contains(position)
        # Invalid traced queries use harmless gather indices, then return
        # NaNs below. This is not clipping or extrapolating a query.
        safe_position = jnp.where(valid, position, self._lower)
        coordinate = (safe_position - self._lower) / self._spacing
        indices, weights = [], []
        for axis, nodes in enumerate(self._axes_m):
            cell = jnp.minimum(
                jnp.floor(coordinate[axis]).astype(jnp.int32), len(nodes) - 2
            )
            t = coordinate[axis] - cell
            if self._method == "cubic":
                idx, weight = _cubic_stencil(cell, t, len(nodes))
            else:
                idx = cell + jnp.arange(2)
                weight = jnp.stack([1.0 - t, t])
            indices.append(idx)
            weights.append(weight)
        values = self._alpha_rad[
            indices[0][:, None, None],
            indices[1][None, :, None],
            indices[2][None, None, :],
        ]
        result = jnp.einsum("i,j,k,ijkn->n", *weights, values)
        return jnp.where(valid, result, jnp.full_like(result, jnp.nan))
