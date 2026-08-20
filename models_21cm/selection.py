"""Post-generation selection for the Zeus21 ensemble.

Named ``selection``, NOT ``select``: pytest and the figure script both put
this directory at ``sys.path[0]``, so a module named ``select`` would shadow
the Python standard library's ``select`` for the whole process -- which
``subprocess``, ``asyncio`` and some matplotlib backends import.

The reionization cut is necessarily *posterior* -- xHI(z) only exists
after a model runs -- so it cannot be a prior. It is applied once to the
whole ensemble, before both the quoted statistics and the figure
subsample, so the two cannot disagree.

Pure module; imports numpy only.
"""

import numpy as np

# McGreer+2015 dark-pixel limit is xHI < 0.06 at z = 5.9. The cut here is
# deliberately looser, so it removes only unambiguously excluded models
# rather than models sitting near the limit. Per-model xHI is stored in
# the npz, so tightening this later costs nothing.
Z_REION_REF = 5.9
XHI_MAX = 0.1


def reionized(xHI, z_xHI, z_ref=Z_REION_REF, x_max=XHI_MAX):
    """Boolean mask of models whose xHI at ``z_ref`` is below ``x_max``.

    ``xHI`` is ``(n_model, n_z)`` on the strictly ascending grid ``z_xHI``.
    """
    xHI = np.atleast_2d(np.asarray(xHI, dtype=float))
    z_xHI = np.asarray(z_xHI, dtype=float)
    if not np.all(np.diff(z_xHI) > 0):
        raise ValueError("z_xHI must be strictly ascending")
    if not z_xHI[0] <= z_ref <= z_xHI[-1]:
        raise ValueError(f"z_ref={z_ref} outside z_xHI [{z_xHI[0]}, {z_xHI[-1]}]")
    at_ref = np.array([np.interp(z_ref, z_xHI, row) for row in xHI])
    return at_ref < x_max


# The band's top edge (250 MHz) is an *observed* frequency, not merely a
# reference point: a model kept for the analysis must be reionized there
# too. z = 4.6816 is NU21 / 250 - 1 (see generate.NU21, generate.PAPER_FREQS),
# the redshift corresponding to the top of the paper's 50-250 MHz band.
Z_BAND_TOP = 4.6816
XHI_MAX_BAND_TOP = 0.05


def reionized_across_band(
    xHI,
    z_xHI,
    z_ref=Z_REION_REF,
    x_max=XHI_MAX,
    z_top=Z_BAND_TOP,
    x_max_top=XHI_MAX_BAND_TOP,
):
    """Boolean mask of models reionized at both ``z_ref`` and the band top.

    ``reionized`` alone only tests xHI at the McGreer+2015 reference
    redshift (z = 5.9) -- the observational anchor for the cut. That is
    necessary but not sufficient: Zeus21's reionization model integrates
    an ionized-fraction ODE (dQ/dt = ndot_ion - Q/t_rec) with a fixed
    clumping factor, and for very low escape fractions recombination can
    outrun the ionizing supply, so Q -- and hence xHI -- can *rise* again
    below z ~ 6. A model can pass the z = 5.9 check and still carry
    several mK of unphysical residual signal at the top of the observed
    band (z = 4.6816, i.e. 250 MHz) because it re-neutralises in between.
    This composes the two checks so a kept model stays reionized across
    the whole band, not just at the single reference redshift.
    """
    return reionized(xHI, z_xHI, z_ref=z_ref, x_max=x_max) & reionized(
        xHI, z_xHI, z_ref=z_top, x_max=x_max_top
    )


def figure_subsample(n_survivors, n_draw, seed):
    """Sorted indices of ``n_draw`` models drawn without replacement.

    The figure draws a subsample for legibility while all statistics use
    the full surviving ensemble, so the seed and count must be recorded
    for the figure to be reproducible.
    """
    if n_draw > n_survivors:
        raise ValueError(f"n_draw={n_draw} exceeds n_survivors={n_survivors}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_survivors, size=n_draw, replace=False))
