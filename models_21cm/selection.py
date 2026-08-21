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

# z = 5.9 is where the dark-pixel method is quoted, so it is the anchor.
# McGreer+2015 gave xHI <= 0.06 + 0.05 (1 sigma) there, from 6 quasar
# sightlines. That measurement has since been superseded: Davies et al.
# 2025 (MNRAS 545, arXiv:2510.25829) redo it with 34 E-XQR-30 spectra and
# find a *weaker* limit, xHI <= 0.191 + 0.056 at z = 5.831 from the
# Lyb + Lyg forests, attributing the difference to cosmic variance across
# McGreer's small sample. Reionization is now understood to finish near
# z ~ 5.3-5.4 (Bosman+2022), not by z = 6.
#
# 0.1 is kept, and is NOT defended as tracking the current constraint.
# Two things make the naive "the limit loosened, so this cut is
# conservative" reading wrong:
#
#   - A single threshold at one redshift is not comparable to Davies'
#     four-redshift ladder. It is looser than 0.1 at the anchor but
#     *tighter* at z = 5.481 (0.030 + 0.048), where this cut tests
#     nothing; applied as a conjunction over all four redshifts the
#     current data keep 1750 models, *fewer* than this cut's 1769.
#   - Being stricter than the data is not automatically conservative for
#     what the paper reports. It does guarantee no kept model is
#     observationally excluded -- a statement about the kept set's
#     validity -- but the models it drops that Davies+2025 would allow
#     retain *less* signal, so the cut raises the reported above-floor
#     fraction by 0.6 points.
#
# What justifies keeping 0.1 is that it changes nothing. xHI(z=5.9) is
# strongly bimodal -- 1765 of 4096 models fall below 0.06, 1812 below
# 0.10, 1925 below 0.25 -- so 0.1 sits in a nearly empty valley, and only
# 71 models lie in [0.1, 0.25) while also passing the band-top limb. The
# cut is almost entirely separating "reionizes" from "never reionizes":
# 52.9% of the 2327 dropped models never reach xHI < 0.5 anywhere on the
# grid, down to z = 4.65. Sweeping the threshold across the whole allowed
# range, or replacing it with the Davies+2025 ladder, leaves the
# operating point at N = 9 and holds the paper's above-floor fraction
# within 0.8 points (74.8-75.6%, against the adopted 75.4%).
# See README.md, "Provenance of the z = 5.9 anchor" and "Sensitivity to
# the threshold", and `horizon_position/reionization_sensitivity.py`,
# which regenerates the numbers.
#
# Per-model xHI is stored in the npz, so retuning this costs nothing.
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
# 0.01, not the 0.05 this project originally shipped with. 0.05 was picked
# from a small candidate table (see README.md) but landed just under the
# threshold of verify_ensemble.py's own `< 1.0 mK` gate -- written by the
# same author, so the closeness was not independent evidence the choice was
# sound. A whole-branch review argued for 0.01 instead: it needs no change
# to that gate (unlike the monotonicity candidate, which would have),
# widens the band-edge margin by roughly an order of magnitude, and costs
# only 13 of 1782 previously-kept models (0.7%). The user agreed. Like the
# 0.05 value before it, 0.01 is still a choice, not a derivation -- see
# README.md for the candidate table it was picked from.
XHI_MAX_BAND_TOP = 0.01


def reionized_across_band(
    xHI,
    z_xHI,
    z_ref=Z_REION_REF,
    x_max=XHI_MAX,
    z_top=Z_BAND_TOP,
    x_max_top=XHI_MAX_BAND_TOP,
):
    """Boolean mask of models reionized at both ``z_ref`` and the band top.

    ``reionized`` alone only tests xHI at the dark-pixel reference
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
