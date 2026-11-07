"""Map-unit, sparse HEALPix profile and conditional-background accounting.

No sky map is loaded at import. Catalogue uncertainty is external and correlated;
least-squares map residuals are not converted to independent-pixel flux errors.
"""

import numpy as np

KB = 1.38065e-23
C = 2.99792e8
H = 6.62607e-34
T_CMB = 2.725
METHODS = ("aperture3_annulus4_8", "plane8", "quadratic10")


def kcmb_sr_per_jy(frequency_hz):
    """Thermodynamic conversion, using the exact pygdsm1.7.0 constants."""
    nu = np.asarray(frequency_hz, dtype=float)
    if np.any(~np.isfinite(nu)) or np.any(nu <= 0):
        raise ValueError("frequency_hz must be finite and positive")
    x = H * nu / (KB * T_CMB)
    g = x**2 * np.exp(-x) / (-np.expm1(-x)) ** 2
    return 1e-26 * C**2 / (2 * KB * nu**2 * g)


def catalogue_flux(record, frequency_mhz):
    """Evaluate a pinned catalogue polynomial only within its published range."""
    lo, hi = record["valid_frequency_mhz"]
    nu = float(frequency_mhz)
    if not np.isfinite(nu) or not lo <= nu <= hi:
        raise ValueError(f"{nu} MHz outside catalogue domain [{lo}, {hi}]")
    unit = record["convention"]["frequency_unit"]
    if unit not in ("GHz", "MHz") or record["convention"]["flux_unit"] != "Jy":
        raise ValueError("unsupported catalogue units")
    return float(
        10
        ** np.polynomial.polynomial.polyval(
            np.log10(nu / 1000 if unit == "GHz" else nu), record["coefficients"]
        )
    )


def area_sr(nside):
    """Equal HEALPix pixel solid angle."""
    if not isinstance(nside, (int, np.integer)) or nside < 1 or nside & (nside - 1):
        raise ValueError("nside must be a positive integer power of two")
    return 4 * np.pi / (12 * nside**2)


def patch(nside, centre_vector, radius_deg=10.0):
    """RING/Galactic pixel identities, separation, and local angular coordinates."""
    import healpy as hp

    centre = np.asarray(centre_vector, dtype=float)
    if centre.shape != (3,) or not np.isclose(np.linalg.norm(centre), 1):
        raise ValueError("centre_vector must be a unit three-vector")
    if not 0 < radius_deg <= 10:
        raise ValueError("reviewed patch radius is at most10 degrees")
    pix = hp.query_disc(nside, centre, np.deg2rad(radius_deg), nest=False)
    vec = np.array(hp.pix2vec(nside, pix, nest=False)).T
    lon = np.arctan2(centre[1], centre[0])
    east = np.array([-np.sin(lon), np.cos(lon), 0])
    north = np.cross(centre, east)
    dot = vec @ centre
    return dict(
        pixel=pix,
        vector=vec,
        radius_deg=np.rad2deg(np.arccos(np.clip(dot, -1, 1))),
        x_deg=np.rad2deg(np.arctan2(vec @ east, dot)),
        y_deg=np.rad2deg(np.arctan2(vec @ north, dot)),
    )


def nominal_profile(nside, centre_vector, fwhm_arcmin=56.0, support_deg=5.0):
    """Conditional nominal Gaussian, not a measured or bounded GSM source PSF.

    Values are sr^-1 at native pixel centres, with exact discrete unit integral.
    Position, width and support are external configuration, never residual-fit.
    """
    if not np.isfinite(fwhm_arcmin) or fwhm_arcmin <= 0:
        raise ValueError("positive finite FWHM required")
    if not 0 < support_deg <= 5:
        raise ValueError("initial reviewed compact support is at most5 degrees")
    p = patch(nside, centre_vector, support_deg)
    sigma_deg = fwhm_arcmin / 60 / np.sqrt(8 * np.log(2))
    q = np.exp(-0.5 * (p["radius_deg"] / sigma_deg) ** 2)
    norm = q.sum() * area_sr(nside)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("profile has no numerically supported pixels")
    return p["pixel"], q / norm


def degrade_sparse(pixels, values, nside_in, nside_out):
    """Exact RING pixel mean under power=None, retaining integrated flux.

    NESTED ancestry determines parent identity without angular-roundoff joins.
    Unlisted input pixels have zero profile, not unmeasured map data.
    """
    import healpy as hp

    area_sr(nside_in)
    area_sr(nside_out)
    if nside_out > nside_in:
        raise ValueError("degradation cannot increase nside")
    pixels = np.asarray(pixels)
    values = np.asarray(values, dtype=float)
    if (
        pixels.ndim != 1
        or pixels.dtype.kind not in "iu"
        or values.shape != pixels.shape
        or len(np.unique(pixels)) != len(pixels)
        or np.any(pixels < 0)
        or np.any(pixels >= 12 * nside_in**2)
        or np.any(~np.isfinite(values))
    ):
        raise ValueError("invalid sparse RING profile")
    factor = (nside_in // nside_out) ** 2
    parent = hp.nest2ring(nside_out, hp.ring2nest(nside_in, pixels) // factor)
    return np.bincount(parent, weights=values / factor, minlength=12 * nside_out**2)


def sample_sparse(pixels, values, requested):
    """Coordinate join with explicit zero support outside the compact profile."""
    order = np.argsort(pixels)
    pixels, values = np.asarray(pixels)[order], np.asarray(values)[order]
    requested = np.asarray(requested)
    idx = np.searchsorted(pixels, requested)
    out = np.zeros(requested.shape, dtype=float)
    ok = idx < len(pixels)
    ok[ok] &= pixels[idx[ok]] == requested[ok]
    out[ok] = values[idx[ok]]
    return out


def neighbour_mask(vectors, centres, radius_deg=5.0):
    """Fixed external-neighbour support mask; True means excluded."""
    bad = np.zeros(len(vectors), dtype=bool)
    for centre in centres:
        bad |= np.asarray(vectors) @ np.asarray(centre) >= np.cos(
            np.deg2rad(radius_deg)
        )
    return bad


def account_patch(data_k, q_sr, geometry, frequency_mhz, nside, method, excluded=None):
    """Fit one predeclared background; return conditional flux, model and masks.

    The annulus check is unmasked, as prescribed for the original diagnostic.
    Plane/quadratic exclude only the saved external-neighbour mask. Source and
    background are solved simultaneously. No covariance or confidence interval is
    inferred from map scatter. Negative formal source coefficients are preserved.
    """
    if method not in METHODS:
        raise ValueError("unknown background method")
    y, q = np.asarray(data_k, float), np.asarray(q_sr, float)
    r = np.asarray(geometry["radius_deg"])
    if y.ndim != 1 or y.shape != q.shape or y.shape != r.shape:
        raise ValueError("patch axes differ")
    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(q)) or np.any(q < 0):
        raise ValueError("finite map and nonnegative finite profile required")
    excluded = np.zeros(y.shape, bool) if excluded is None else np.asarray(excluded)
    if excluded.shape != y.shape or excluded.dtype.kind != "b":
        raise ValueError("excluded mask must be boolean on the patch pixel axis")
    cnu = float(kcmb_sr_per_jy(frequency_mhz * 1e6))
    template = cnu * q  # K_CMB/Jy, full normalized profile including pixel response
    result = dict(
        method=method,
        status="unsupported",
        formal_flux_jy=None,
        covariance_jy2=None,
        compatible_interval_jy=None,
        uncertainty_scope="external profile/background systematics unresolved",
    )
    arrays = dict(
        data_k=y,
        template_k_per_jy=template,
        excluded=excluded,
        fit_mask=np.zeros(y.shape, bool),
    )
    if method == METHODS[0]:
        ap, ann = r <= 3, (r >= 4) & (r <= 8)
        arrays.update(aperture_mask=ap, annulus_mask=ann, fit_mask=ap | ann)
        if not np.any(ap) or not np.any(ann):
            return result | {"reason": "empty aperture or annulus"}, arrays
        response = template[ap].sum() - ap.sum() * template[ann].mean()
        if not np.isfinite(response) or response <= 0:
            return result | {"reason": "nonpositive aperture profile response"}, arrays
        flux = (y[ap].sum() - ap.sum() * y[ann].mean()) / response
        bg = np.full(y.shape, np.mean(y[ann] - flux * template[ann]))
        result.update(
            aperture_area_sr=float(ap.sum() * area_sr(nside)),
            annulus_area_sr=float(ann.sum() * area_sr(nside)),
            profile_response_k_per_jy=float(response),
            neighbour_mask_applied=False,
        )
    else:
        use = (r <= (8 if method == METHODS[1] else 10)) & ~excluded
        x, z = np.asarray(geometry["x_deg"]) / 10, np.asarray(geometry["y_deg"]) / 10
        terms = [np.ones(y.shape), x, z]
        if method == METHODS[2]:
            terms += [x * x, x * z, z * z]
        basis = np.array(terms).T
        peak = np.max(template)
        if peak <= 0:
            return result | {"reason": "no compact-profile support"}, arrays
        design = np.column_stack([template / peak, basis])
        arrays["fit_mask"] = use
        if use.sum() < design.shape[1]:
            return result | {"reason": "fewer pixels than fitted coefficients"}, arrays
        coef, _, rank, singular = np.linalg.lstsq(design[use], y[use], rcond=None)
        result.update(
            rank=int(rank),
            singular_values=singular.tolist(),
            condition=float(singular[0] / singular[-1]) if singular[-1] > 0 else None,
            neighbour_mask_applied=True,
            background_coefficients_k=coef[1:].tolist(),
            background_coordinates="x_deg/10,y_deg/10; [1,x,y,x²,xy,y²]",
        )
        if rank != design.shape[1]:
            return result | {
                "reason": "rank-deficient compact/background design"
            }, arrays
        flux, bg = coef[0] / peak, basis @ coef[1:]
    model = flux * template + bg
    residual = y - model
    arrays.update(
        background_k=bg, compact_k=flux * template, model_k=model, residual_k=residual
    )
    result.update(
        status="conditional",
        formal_flux_jy=float(flux),
        negative_formal_warning=bool(flux < 0),
        residual_rms_k=float(np.sqrt(np.mean(residual[arrays["fit_mask"]] ** 2))),
        fitted_pixel_count=int(arrays["fit_mask"].sum()),
    )
    return result, arrays
