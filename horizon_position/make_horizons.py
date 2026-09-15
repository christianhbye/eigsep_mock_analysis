"""Compute horizon elevation profiles alpha_h(az) for the 19 positions.

Runs in the eigsep_terrain environment (it imports eigsep_terrain, which
is NOT available in the mock_analysis env):

    PYTHONPATH=/home/christian/Documents/research/eigsep/eigsep_terrain \
    uv run --frozen --project /home/christian/Documents/research/eigsep/eigsep_terrain \
        python horizon_position/make_horizons.py

Note: PYTHONPATH is required because eigsep_terrain uses a flat (non-src)
layout and is not installed as an editable package in its venv — uv only
adds the project dir to sys.path when running with -c, not a script path.

`calc_horizon` assigns each DEM pixel to a whole number of azimuth bins and
takes the max per bin, so a coarse grid does not resolve how the horizon
responds to a horizontal antenna move: moving the antenna re-sorts which pixel
wins each bin (108 of 720 bins flip for a 0.1 m East shift, against 2 for
0.1 m Up) and each flip is a discontinuous jump. N_AZ is therefore set fine
enough to converge, and consumers reduce as they need.

Converged at N_AZ = 46080: refining to 92160 moves the +1 m curves by 6 per
cent, corr(+d,-d) for a +/-0.1 m pair reaches -0.989 (geometry: -1) and the
RMS ratio 1 m / 0.1 m reaches 9.94 (geometry: 10). At 720 those are -0.058 and
4.69. 46080 is 64 x masks.N_AZ_MASK so it reduces onto the mask grid exactly.

WHO REDUCES, AND WHY: `run_sims.py` now uses `eigsim.open_sky_weight`, which
integrates over the phi cell, so it has no reduction step of its own.
`run_beam_sims.py` still uses `masks.open_sky_weight` with
`masks.reduce_azimuth`, which is unchanged because the paper is pinned. The
figures still plot alpha_h as it is stored here, because averaging hides
55-60 per cent of the peak at cliff edges. Nothing derived is stored: the
reduced curve (where one is still taken) is a function call, not an array in
this file.

Output: output/horizons_position.npz with
  names            (19,)          position names
  enu              (19, 3)        antenna ENU positions [m]
  az_grid          (n_az,)        azimuths [rad], = atan2(E, N), North->East
  alpha_h          (19, n_az)     horizon elevation [rad] per position, float64
  crds             (19, 2, n_az)  centre of the DEM pixel that sets alpha_h
                                  [m], float64, DEM frame, [:, 0] North,
                                  [:, 1] East; NaN where no pixel rises above
                                  0, none matches, or the match is ambiguous
  dalpha_dE        (n_az,)        total d alpha_h / d East at nominal, pixel
                                  term plus azimuthal parallax [rad/m]
  dalpha_dN        (n_az,)        total d alpha_h / d North, likewise [rad/m]
  dalpha_dU        (n_az,)        d alpha_h / d Up at nominal [rad/m]; no
                                  parallax term, so it is pointwise and total
  dalpha_dE_pixel  (n_az,)        the winning pixel's own d alpha / d East
  dalpha_dN_pixel  (n_az,)        the winning pixel's own d alpha / d North
  daz_dE           (n_az,)        d az_p / d East of the winning pixel [rad/m]
  daz_dN           (n_az,)        d az_p / d North of the winning pixel [rad/m]
  jac_valid        (n_az,)        bool; every derivative above is 0 where False
  n_az             scalar
  pos_sha          hash of enu  (content identifier for the 19-position
                                configuration)

`calc_horizon` stores res * pixel index in an int array, which truncates crds
to whole metres (two pixels per value at 0.5 m/px). The pixel is recovered by
recomputing each candidate's horizon angle as calc_horizon does and keeping
the one that equals alpha_h exactly.

The pixel partials differentiate that pixel's arctan2(U_p - u0, r_min), with
r_min to the pixel's nearest point p*. They are the pointwise derivative of
alpha_h wherever the winning pixel does not change; a move that hands the
azimuth to another pixel is a jump they do not see. A horizontal move also
turns the pixel's azimuth az_p = atan2(p*_E - e0, p*_N - n0), which at fixed
azimuth adds -alpha_h'(az) * d az_p / dx, with alpha_h' the central difference
of the nominal curve. dalpha_dE and dalpha_dN include that term. They are a
first-order translation of a piecewise-constant curve, so they hold for
cell-integrated quantities (eigsim.open_sky_weight's W, t_ant, and jax.jvp
through them), not pointwise on alpha_h: use them for the simulation tangent
and the *_pixel keys for pointwise checks against alpha_h.
"""

import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
from eigsep_terrain.marjum_dem import MarjumDEM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from positions import build_positions  # noqa: E402

N_AZ = 46080  # = 64 * masks.N_AZ_MASK; see the module docstring
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEM_CACHE = OUTPUT_DIR / "marjum_dem.npz"


def pixel_candidates(stored, res, size):
    """Full-resolution pixel indices that ``calc_horizon`` stores as ``stored``.

    ``calc_horizon`` writes ``res * index`` into an int array, which truncates
    it toward zero, so every index ``j`` with ``int(res * j) == stored`` is a
    candidate (two per value at res = 0.5 m).

    Returns ``(idx, ok)``, both ``(n_az, k)``: candidate indices and whether
    each is in ``[0, size)`` and truncates to ``stored``.
    """
    stored = np.asarray(stored, dtype=np.int64)
    span = int(np.ceil(1.0 / float(res))) + 2
    base = np.floor(stored / float(res)).astype(np.int64) - 1
    idx = base[:, None] + np.arange(span)
    ok = (idx >= 0) & (idx < size)
    # the same float64 product and truncating cast as calc_horizon's base case
    ok &= (res * idx).astype(np.int64) == stored[:, None]
    return idx, ok


def pixel_offset(dem, ni, ei, e0, n0):
    """Vector from the antenna to the nearest point of pixels ``(ni, ei)``.

    Returns ``(de, dn, r_min)`` [m]. The nearest point clamps ``(e0, n0)``
    into the pixel rectangle whose edges come from ``get_en(edges=True)``;
    ``r_min`` is ``calc_rmin``'s arithmetic for that pixel, bit for bit.
    """
    e_edges, n_edges = dem.get_en(edges=True)
    de = np.clip(e0, e_edges[ei], e_edges[ei + 1]) - e0
    dn = np.clip(n0, n_edges[ni], n_edges[ni + 1]) - n0
    r_min = np.sqrt(dn**2 + de**2)
    return de, dn, r_min


def match_horizon_pixels(dem, crds, alpha_h, e0, n0, u0):
    """Recover the full-resolution DEM pixel behind each azimuth's horizon.

    ``crds`` is ``calc_horizon``'s truncated second return. Each candidate
    pixel's angle is recomputed as calc_horizon's base case does it, with a
    float32 altitude minus the Python float ``u0``, and the candidate whose
    angle equals ``alpha_h`` exactly is kept.

    Returns ``(ni, ei, n_match)``: pixel indices, -1 unless exactly one
    candidate matches, and the number of matching candidates per azimuth.
    Azimuths with ``alpha_h == 0`` are unset (no pixel rose above 0) and
    have ``n_match == 0``.
    """
    n_rows, n_cols = dem.data.shape
    n_idx, n_ok = pixel_candidates(crds[0], dem.res, n_rows)
    e_idx, e_ok = pixel_candidates(crds[1], dem.res, n_cols)
    ni = np.clip(n_idx[:, :, None], 0, n_rows - 1)
    ei = np.clip(e_idx[:, None, :], 0, n_cols - 1)
    ni, ei = np.broadcast_arrays(ni, ei)

    _, _, r_min = pixel_offset(dem, ni, ei, e0, n0)
    angle = np.arctan2(dem.data[ni, ei] - u0, r_min)
    match = n_ok[:, :, None] & e_ok[:, None, :] & (angle == alpha_h[:, None, None])
    match &= (alpha_h > 0.0)[:, None, None]

    n_match = match.reshape(len(alpha_h), -1).sum(axis=1)
    first = np.argmax(match.reshape(len(alpha_h), -1), axis=1)
    unique = n_match == 1
    ni_out = np.where(
        unique, ni.reshape(len(alpha_h), -1)[np.arange(len(first)), first], -1
    )
    ei_out = np.where(
        unique, ei.reshape(len(alpha_h), -1)[np.arange(len(first)), first], -1
    )
    return ni_out, ei_out, n_match


def horizon_jacobian(dem, ni, ei, e0, n0, u0):
    """Analytic d alpha_h / d(e, n, u) at one antenna position [rad/m].

    ``(ni, ei)`` is the pixel from ``match_horizon_pixels`` (-1 where none).
    With ``dz = U_p - u0``, ``r_min`` to the pixel's nearest point ``p*`` and
    ``rho2 = r_min**2 + dz**2``, ``alpha_h = arctan2(dz, r_min)`` gives

        d alpha/dE = dz * (p*_E - e0) / (r_min * rho2)
        d alpha/dN = dz * (p*_N - n0) / (r_min * rho2)
        d alpha/dU = -r_min / rho2

    ``dz`` is taken in float64 here, not calc_horizon's float32. The result
    is exact while the same pixel keeps winning. Returns
    ``(d_alpha_de, d_alpha_dn, d_alpha_du, valid)``; ``valid`` is a matched
    pixel with ``r_min > 0`` and the derivatives are 0 elsewhere.
    """
    valid = (ni >= 0) & (ei >= 0)
    ni_s, ei_s = np.where(valid, ni, 0), np.where(valid, ei, 0)
    de, dn, r_min = pixel_offset(dem, ni_s, ei_s, e0, n0)
    valid &= r_min > 0.0

    dz = dem.data[ni_s, ei_s].astype(np.float64) - u0
    r_safe = np.where(valid, r_min, 1.0)
    rho2 = r_safe**2 + dz**2
    d_alpha_de = np.where(valid, dz * de / (r_safe * rho2), 0.0)
    d_alpha_dn = np.where(valid, dz * dn / (r_safe * rho2), 0.0)
    d_alpha_du = np.where(valid, -r_safe / rho2, 0.0)
    return d_alpha_de, d_alpha_dn, d_alpha_du, valid


def azimuth_parallax(dem, ni, ei, valid, e0, n0):
    """d az_p / d(e, n) of each azimuth's winning pixel [rad/m].

    ``az_p = atan2(p*_E - e0, p*_N - n0)`` to the same nearest point ``p*`` and
    ``r_min`` as ``horizon_jacobian``, so

        d az_p/dE = -(p*_N - n0) / r_min**2
        d az_p/dN = +(p*_E - e0) / r_min**2

    Both are 0 where ``valid`` (``horizon_jacobian``'s) is False. There is no
    Up term: raising the antenna does not turn a pixel's azimuth.
    """
    ni_s, ei_s = np.where(valid, ni, 0), np.where(valid, ei, 0)
    de, dn, r_min = pixel_offset(dem, ni_s, ei_s, e0, n0)
    r2_safe = np.where(valid, r_min, 1.0) ** 2
    daz_de = np.where(valid, -dn / r2_safe, 0.0)
    daz_dn = np.where(valid, de / r2_safe, 0.0)
    return daz_de, daz_dn


def azimuth_slope(alpha_h):
    """Central difference d alpha_h / d az on the periodic native grid [rad/rad]."""
    d_az = 2 * np.pi / alpha_h.size
    return (np.roll(alpha_h, -1) - np.roll(alpha_h, 1)) / (2 * d_az)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("Building / loading Marjum DEM...")
    dem = MarjumDEM(cache_file=str(DEM_CACHE))
    print(f"  DEM {dem.data.shape} {dem.data.dtype}, {float(dem.res)} m/px")

    positions = build_positions()
    names = [n for n, _ in positions]
    enu = np.array([e for _, e in positions], dtype=np.float64)
    az_grid = np.linspace(0.0, 2 * np.pi, N_AZ, endpoint=False)

    alpha_h = np.empty((len(positions), N_AZ), dtype=np.float64)
    crds_all = np.full((len(positions), 2, N_AZ), np.nan, dtype=np.float64)
    pixels = []
    for i, (name, e) in enumerate(positions):
        e0, n0, u0 = float(e[0]), float(e[1]), float(e[2])
        hangles, crds = dem.calc_horizon(e0, n0, u0, n_az=N_AZ)
        alpha_h[i] = np.asarray(hangles, dtype=np.float64)
        deg = np.degrees([alpha_h[i].min(), np.median(alpha_h[i]), alpha_h[i].max()])
        print(
            f"  [{i:2d}] {name:10s} alpha_h(min,med,max) deg = "
            f"{deg[0]:6.2f} {deg[1]:6.2f} {deg[2]:6.2f}"
        )

        ni, ei, n_match = match_horizon_pixels(dem, crds, alpha_h[i], e0, n0, u0)
        pixels.append((ni, ei))
        found = ni >= 0
        crds_all[i, 0] = np.where(found, dem.res * ni, np.nan)
        crds_all[i, 1] = np.where(found, dem.res * ei, np.nan)
        n_won = int(np.count_nonzero(alpha_h[i] > 0.0))
        print(
            f"       pixel match: {found.sum()}/{n_won} exact "
            f"({100.0 * found.sum() / max(n_won, 1):.3f}%), "
            f"ambiguous {np.count_nonzero(n_match > 1)}, "
            f"unmatched {n_won - np.count_nonzero(n_match > 0)}, "
            f"alpha_h == 0: {N_AZ - n_won}"
        )

    i_nom = names.index("nominal")
    e0, n0, u0 = (float(x) for x in enu[i_nom])
    d_alpha_de_px, d_alpha_dn_px, d_alpha_du, jac_valid = horizon_jacobian(
        dem, *pixels[i_nom], e0, n0, u0
    )
    daz_de, daz_dn = azimuth_parallax(dem, *pixels[i_nom], jac_valid, e0, n0)
    slope = azimuth_slope(alpha_h[i_nom])
    d_alpha_de = d_alpha_de_px - slope * daz_de
    d_alpha_dn = d_alpha_dn_px - slope * daz_dn
    print(
        f"  Jacobian at nominal: {jac_valid.sum()}/{N_AZ} azimuths valid, "
        f"|d alpha/dU| median {np.median(np.abs(d_alpha_du[jac_valid])):.3e} rad/m"
    )

    pos_sha = hashlib.sha256(np.ascontiguousarray(enu).tobytes()).hexdigest()
    out = OUTPUT_DIR / "horizons_position.npz"
    np.savez(
        out,
        names=np.array(names),
        enu=enu,
        az_grid=az_grid,
        alpha_h=alpha_h,
        crds=crds_all,
        dalpha_dE=d_alpha_de,
        dalpha_dN=d_alpha_dn,
        dalpha_dU=d_alpha_du,
        dalpha_dE_pixel=d_alpha_de_px,
        dalpha_dN_pixel=d_alpha_dn_px,
        daz_dE=daz_de,
        daz_dN=daz_dn,
        jac_valid=jac_valid,
        n_az=N_AZ,
        pos_sha=pos_sha,
    )
    print(f"Saved {out}  alpha_h shape {alpha_h.shape}")


if __name__ == "__main__":
    main()
