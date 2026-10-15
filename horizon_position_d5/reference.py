"""Independent horizons of a continuous bilinear DEM or a sampled ray.

The continuous reference maximizes elevation along every raster cell traversed
by the ray. It evaluates cell endpoints and the stationary points of the
quadratic surface restricted to that ray. It does not differentiate a selected
march sample. Units are metres and radians; bearings are degrees CCW from the
DEM's grid East. All arithmetic is NumPy float64.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class HorizonTrace:
    """Horizon values and selected terrain points.

    All arrays except ``edge_range_m`` have shape ``(n_heights, n_bearings)``;
    ``gradient_en`` appends an axis of length two, ordered E, N. Heights are
    the raw and curvature-corrected terrain elevations in the DEM datum.

    ``edge_limited`` means the far endpoint won in continuous mode, or the
    last evaluated sample won in sampled mode. The latter flags a potentially
    truncated skyline; it does not assert the maximum lies at the true edge.
    An inner-cutoff maximum is identifiable by ``distance_m == inner_cutoff_m``.
    A selected cell gradient is one-sided at a raster crease; it is not the
    derivative of the horizon envelope.
    """

    alpha_rad: np.ndarray
    distance_m: np.ndarray
    surface_height_m: np.ndarray
    curvature_height_m: np.ndarray
    gradient_en: np.ndarray
    edge_range_m: np.ndarray
    edge_limited: np.ndarray


class BilinearDEM:
    """Float64 bilinear surface over an original DEM sample array.

    Array indexing is ``data[north_pixel, east_pixel]``, with
    ``east_pixel = E / res + e0_px`` and similarly N. ``interp`` and ``gradient``
    support the entire sample rectangle, including its boundary. Ray tracing
    uses the team's more conservative upper bound, one grid interval below the
    final sample. At a crease, ``gradient`` selects the cell given by floor;
    at the final sample it selects the last available cell.
    """

    def __init__(self, data, res, e0_px=0, n0_px=0):
        values = np.asarray(data, dtype=np.float64)
        if values.ndim != 2 or min(values.shape) < 3:
            raise ValueError(
                "DEM data must be a 2-D array with at least 3 samples per axis"
            )
        if not np.isfinite(values).all():
            raise ValueError("DEM samples must all be finite")
        if not np.isfinite(res) or res <= 0:
            raise ValueError("DEM resolution must be finite and positive")
        for name, origin in (("e0_px", e0_px), ("n0_px", n0_px)):
            if not np.isfinite(origin) or float(origin) != int(origin):
                raise ValueError(f"{name} must be a finite integer pixel offset")
        self.data = values
        self.res = float(res)
        self.e0_px = int(e0_px)
        self.n0_px = int(n0_px)
        ny, nx = values.shape
        self.extent_e = (-self.e0_px * self.res, (nx - 1 - self.e0_px) * self.res)
        self.extent_n = (-self.n0_px * self.res, (ny - 1 - self.n0_px) * self.res)

    @classmethod
    def from_npz(cls, path: str | Path):
        """Read the team's ``dem``, ``res``, ``e0_px``, ``n0_px`` NPZ fields."""
        with np.load(path, allow_pickle=False) as arrays:
            return cls(
                arrays["dem"],
                float(arrays["res"]),
                float(arrays["e0_px"]),
                float(arrays["n0_px"]),
            )

    def _cell_coordinates(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        ny, nx = self.data.shape
        # Endpoint arithmetic can stray by a few ulps beyond a raster edge.
        # This tolerance only repairs arithmetic at the full sample rectangle;
        # it never extends the team's conservative tracing domain.
        tolerance = 16 * np.finfo(float).eps * max(nx, ny)
        invalid = (
            ~np.isfinite(x)
            | ~np.isfinite(y)
            | (x < -tolerance)
            | (y < -tolerance)
            | (x > nx - 1 + tolerance)
            | (y > ny - 1 + tolerance)
        )
        if np.any(invalid):
            raise ValueError(
                "Requested coordinate lies outside the DEM sample rectangle"
            )
        x = np.clip(x, 0, nx - 1)
        y = np.clip(y, 0, ny - 1)
        ix = np.minimum(np.floor(x).astype(np.intp), nx - 2)
        iy = np.minimum(np.floor(y).astype(np.intp), ny - 2)
        return ix, iy, x - ix, y - iy

    def _interp_pixels(self, x, y):
        ix, iy, fx, fy = self._cell_coordinates(x, y)
        d = self.data
        # Preserve the team's interpolation arithmetic in sampled mode.
        return (
            (1 - fx) * (1 - fy) * d[iy, ix]
            + fx * (1 - fy) * d[iy, ix + 1]
            + (1 - fx) * fy * d[iy + 1, ix]
            + fx * fy * d[iy + 1, ix + 1]
        )

    def _gradient_pixels(self, x, y):
        ix, iy, fx, fy = self._cell_coordinates(x, y)
        d = self.data
        de = (
            (1 - fy) * (d[iy, ix + 1] - d[iy, ix])
            + fy * (d[iy + 1, ix + 1] - d[iy + 1, ix])
        ) / self.res
        dn = (
            (1 - fx) * (d[iy + 1, ix] - d[iy, ix])
            + fx * (d[iy + 1, ix + 1] - d[iy, ix + 1])
        ) / self.res
        return np.stack((de, dn), axis=-1)

    def interp(self, east, north):
        """Bilinearly interpolate at broadcastable E, N coordinates in metres."""
        return self._interp_pixels(
            np.asarray(east) / self.res + self.e0_px,
            np.asarray(north) / self.res + self.n0_px,
        )

    def gradient(self, east, north):
        """Selected cell's E/N height gradient, shape ``broadcast_shape + (2,)``."""
        return self._gradient_pixels(
            np.asarray(east) / self.res + self.e0_px,
            np.asarray(north) / self.res + self.n0_px,
        )

    def edge_range(self, east, north, ux, uy):
        """Team-domain edge along a unit direction; cardinal threshold is 1e-12.

        The allowed rectangle is ``[extent_lo, extent_hi - res]`` on each axis.
        A point on its boundary is permitted, but an outward ray has zero range.
        """
        coordinates = np.asarray([east, north, ux, uy], dtype=float)
        if not np.isfinite(coordinates).all():
            raise ValueError("Ray coordinates and direction must be finite")
        if not np.isclose(np.hypot(ux, uy), 1.0, rtol=0, atol=1e-12):
            raise ValueError("Ray direction must have unit length")
        limits = []
        for direction, point, (lo, hi) in (
            (ux, east, self.extent_e),
            (uy, north, self.extent_n),
        ):
            hi -= self.res
            if not lo <= point <= hi:
                raise ValueError("Antenna lies outside the team's tracing rectangle")
            if direction > 1e-12:
                limits.append((hi - point) / direction)
            elif direction < -1e-12:
                limits.append((lo - point) / direction)
        return min(limits)


def _ray_intervals(dem, pixel_origin, direction, inner, edge):
    """Ordered distinct intersections; retain every representable short cell."""
    radii = [np.array([inner, edge])]
    for origin, unit in zip(pixel_origin, direction, strict=True):
        if abs(unit) <= 1e-12:
            continue
        start = origin + inner * unit / dem.res
        stop = origin + edge * unit / dem.res
        lower, upper = sorted((start, stop))
        lines = np.arange(np.ceil(lower), np.floor(upper) + 1)
        crossed = (lines - origin) * dem.res / unit
        radii.append(crossed[(crossed > inner) & (crossed < edge)])
    return np.unique(np.concatenate(radii))


def _ray_polynomials(dem, pixel_origin, direction, radii):
    """Local ``a*t**2 + b*t + c`` for intervals with a representable interior."""
    left, right = radii[:-1], radii[1:]
    middle = left + (right - left) / 2
    usable = (middle > left) & (middle < right)
    left, right, middle = left[usable], right[usable], middle[usable]
    ux, uy = direction
    x = pixel_origin[0] + middle * ux / dem.res
    y = pixel_origin[1] + middle * uy / dem.res
    ix, iy, _, _ = dem._cell_coordinates(x, y)
    d = dem.data
    z00 = d[iy, ix]
    de = d[iy, ix + 1] - z00
    dn = d[iy + 1, ix] - z00
    cross = d[iy + 1, ix + 1] - d[iy, ix + 1] - d[iy + 1, ix] + z00
    fx = pixel_origin[0] + left * ux / dem.res - ix
    fy = pixel_origin[1] + left * uy / dem.res - iy
    a = cross * ux * uy / dem.res**2
    b = ((de + cross * fy) * ux + (dn + cross * fx) * uy) / dem.res
    c = z00 + de * fx + dn * fy + cross * fx * fy
    return left, right, a, b, c


def _stationary_radii(left, right, a, b, c, height, curvature):
    """Stable interior roots; a derivative-sign bracket prevents huge quotients."""
    coefficient = a - curvature
    d0 = b * left - (c - height) - curvature * left**2
    width = right - left
    d1 = d0 + coefficient * width * (2 * left + width)
    bracket = ((d0 > 0) & (d1 < 0)) | ((d0 < 0) & (d1 > 0))
    bracket &= coefficient != 0
    r0, r1 = left[bracket], right[bracket]
    quotient = d0[bracket] / coefficient[bracket]
    delta = -quotient / (np.sqrt(r0**2 - quotient) + r0)
    roots = r0 + delta
    return roots[(roots > r0) & (roots < r1)]


def trace_horizon(
    dem,
    position_enu_m,
    bearings_deg,
    *,
    height_offsets_m=(0.0,),
    method="continuous",
    march_step_m=0.25,
    inner_cutoff_m=0.25,
    earth_radius_m=6371e3,
):
    """Trace the horizon for one position and several additive U offsets.

    ``continuous`` tests every raster intersection, the fixed near cutoff, the
    far endpoint and all interior stationary points. Exact equal maxima choose
    the nearest radius; no tolerance changes the winner. The output includes
    far-edge maxima even though the team's finite march excludes that endpoint.

    ``sampled`` evaluates ``arange(inner_cutoff_m, edge, march_step_m)`` using
    the team's float64 bilinear interpolation and first-angle-maximum rule.
    Refining the step never moves the inner cutoff. ``earth_radius_m=inf`` is
    supported for no-curvature synthetic controls. Empty rays raise ValueError.
    No terrain beyond the finite domain is assumed or extrapolated.
    """
    if method not in {"continuous", "sampled"}:
        raise ValueError("method must be 'continuous' or 'sampled'")
    p = np.asarray(position_enu_m, dtype=float)
    bearings = np.asarray(bearings_deg, dtype=float)
    offsets = np.asarray(height_offsets_m, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError("position_enu_m must contain three finite coordinates")
    if bearings.ndim != 1 or not bearings.size or not np.isfinite(bearings).all():
        raise ValueError("bearings_deg must be a nonempty finite 1-D array")
    if offsets.ndim != 1 or not offsets.size or not np.isfinite(offsets).all():
        raise ValueError("height_offsets_m must be a nonempty finite 1-D array")
    for name, value in (
        ("march_step_m", march_step_m),
        ("inner_cutoff_m", inner_cutoff_m),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if np.isnan(earth_radius_m) or earth_radius_m <= 0:
        raise ValueError(
            "earth_radius_m must be positive (infinity disables curvature)"
        )
    curvature = 1 / (2 * earth_radius_m)
    heights = p[2] + offsets
    shape = (len(heights), len(bearings))
    alpha, distance, surface, curved = [np.empty(shape) for _ in range(4)]
    gradients = np.empty((*shape, 2))
    edges = np.empty(len(bearings))
    limited = np.empty(shape, dtype=bool)
    pixel_origin = (p[0] / dem.res + dem.e0_px, p[1] / dem.res + dem.n0_px)

    for j, beta in enumerate(np.radians(bearings)):
        ux, uy = np.cos(beta), np.sin(beta)
        edge = dem.edge_range(p[0], p[1], ux, uy)
        edges[j] = edge
        if edge <= inner_cutoff_m:
            raise ValueError(
                f"Bearing {bearings[j]} has no ray beyond the inner cutoff"
            )
        if method == "sampled":
            radii = np.arange(inner_cutoff_m, edge, march_step_m)
            raw = dem.interp(p[0] + radii * ux, p[1] + radii * uy)
            corrected = raw - radii**2 / (2 * earth_radius_m)
            angles = np.arctan2(corrected[None] - heights[:, None], radii[None])
            winners = np.argmax(angles, axis=1)
            rows = np.arange(len(heights))
            selected = radii[winners]
            alpha[:, j] = angles[rows, winners]
            distance[:, j] = selected
            surface[:, j] = raw[winners]
            curved[:, j] = corrected[winners]
            gradients[:, j] = dem.gradient(p[0] + selected * ux, p[1] + selected * uy)
            limited[:, j] = winners == len(radii) - 1
            continue

        radii = _ray_intervals(dem, pixel_origin, (ux, uy), inner_cutoff_m, edge)
        polynomial = _ray_polynomials(dem, pixel_origin, (ux, uy), radii)
        for i, height in enumerate(heights):
            stationary = _stationary_radii(*polynomial, height, curvature)
            candidates = np.unique(np.concatenate((radii, stationary)))
            x = pixel_origin[0] + candidates * ux / dem.res
            y = pixel_origin[1] + candidates * uy / dem.res
            raw = dem._interp_pixels(x, y)
            corrected = raw - curvature * candidates**2
            slope = (corrected - height) / candidates
            winner = np.argmax(slope)
            radius = candidates[winner]
            alpha[i, j] = np.arctan2(corrected[winner] - height, radius)
            distance[i, j] = radius
            surface[i, j] = raw[winner]
            curved[i, j] = corrected[winner]
            gradients[i, j] = dem._gradient_pixels(x[winner], y[winner])
            limited[i, j] = radius == edge

    return HorizonTrace(alpha, distance, surface, curved, gradients, edges, limited)
