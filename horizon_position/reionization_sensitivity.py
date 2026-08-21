"""How much does the reionization cut's threshold move the reported numbers?

    uv run python horizon_position/reionization_sensitivity.py

`selection.XHI_MAX` (0.1 at z = 5.9) is a choice, and the observational
limit it was anchored to has moved: McGreer et al. 2015 gave
xHI <= 0.06 + 0.05 (1 sigma) at z = 5.9 from 6 quasar sightlines, and
Davies et al. 2025 (MNRAS 545, arXiv:2510.25829) supersede it using 34
E-XQR-30 spectra with a *weaker* limit, xHI <= 0.191 + 0.056 at
z = 5.831 from the Lyb+Lyg forests, attributing the difference to cosmic
variance across McGreer's small sample.

0.1 is kept, but not because the newer limit makes it conservative --
that reading does not survive contact with the numbers. The Davies ladder
is looser than 0.1 at the anchor yet tighter at z = 5.481, and applied as
a conjunction over all four redshifts it keeps *fewer* models (1750) than
the adopted cut (1769). Nor is being stricter than the data conservative
for the statistic the paper reports: the models the cut drops retain
less signal, so it raises the above-floor fraction slightly. What
justifies 0.1 is that it changes nothing -- which is what this script
measures.

This script quantifies what that choice costs, by re-cutting the stored
per-model xHI(z) -- no regeneration -- and recomputing, for each variant,
the operating point under the same stays-below rule
`notebooks/horizon_shift.ipynb` uses, plus the retained-RMS percentiles
and the above-floor fraction the paper quotes.

The band-top limb of the cut (xHI(z=4.6816) < 0.01) is held fixed in
every variant except the last. It is unaffected by the literature move:
reionization ending near z ~ 5.3-5.4 is comfortably above the band's top
edge at z = 4.6816, so requiring a kept model to be reionized there is if
anything better supported now than it was.
"""

import make_paper_signal_loss_figure as mp
import numpy as np
import selection  # via the sys.path entry mp adds on import

PAPER = mp.PAPER
N_SHOW = mp.N_SHOW_TEXT

# Davies et al. 2025, abstract: fiducial 1 sigma upper limits on the
# volume-averaged neutral fraction from the optimally sensitive
# Lyb + Lyg combination, as (z, limit, +1 sigma). Applied as a ladder --
# every redshift must pass -- rather than at a single anchor redshift.
DAVIES_2025 = (
    (5.481, 0.030, 0.048),
    (5.654, 0.095, 0.037),
    (5.831, 0.191, 0.056),
    (6.043, 0.199, 0.087),
)

# Window used to count the models the current cut drops but Davies+2025
# still allow: at or above the adopted 0.1, below the z = 5.831 limit
# taken with its 1 sigma.
RESTORE_LO, RESTORE_HI = 0.1, 0.25


def main():
    fg = np.load(PAPER / "foreground_svd.npz")
    shift = np.load(PAPER / "horizon_shift.npz")
    freqs, Vh = fg["freqs_MHz"], shift["Vh"]
    assert np.array_equal(shift["freqs_MHz"], freqs), "frequency grid mismatch"
    n_f = freqs.size

    # Uncut ensemble: mp.load_t21 applies the cut, which is the thing under
    # test here, so the npz is read directly instead.
    m = np.load(mp.MODELS_NPZ, allow_pickle=False)
    assert np.array_equal(m["freqs_MHz"], freqs), "frequency grid mismatch"
    xHI, z_xHI = m["xHI"], m["z_xHI"]
    T21 = m["T21_mK"] * 1e-3  # K

    # Foreground floor, in the basis Vh was built for. Dropping the constant
    # receiver temperature first is load-bearing: Vh was derived by SVD of
    # t_sys - t_receiver, and leaving the ~50 K offset in inflates the
    # apparent residual by orders of magnitude (see notebooks/horizon_shift.ipynb).
    t_ant = fg["t_sys"] - fg["t_receiver"]
    n_time = t_ant.shape[0]
    s_fg = np.linalg.svd(t_ant, compute_uv=False)
    tail = np.concatenate([np.cumsum(s_fg[::-1] ** 2)[::-1], [0.0]])
    fg_resid = np.sqrt(tail / (n_time * n_f))[: N_SHOW + 1] * 1e3  # mK

    n_modes = np.arange(N_SHOW + 1)
    c = T21 @ Vh.T
    ret_all = np.array(
        [np.sqrt(np.sum(c[:, N:] ** 2, axis=1) / n_f) * 1e3 for N in n_modes]
    )  # (N_SHOW+1, n_model) mK

    def at(z):
        return np.array([np.interp(z, z_xHI, row) for row in xHI])

    def anchor(median_retained):
        """Smallest N with fg_resid below the median, and below thereafter."""
        below = fg_resid < median_retained
        for N in n_modes:
            if below[N:].all():
                return int(N)
        return None

    def report(label, keep):
        ret = ret_all[:, keep]
        n_star = anchor(np.median(ret, axis=1))
        p5, p50, p95 = np.percentile(ret[mp.N_ANCHOR], [5, 50, 95])
        above = (ret[mp.N_ANCHOR] > fg_resid[mp.N_ANCHOR]).mean() * 100
        print(
            f"| {label:<38s} | {keep.sum():4d} | {n_star:2d} | "
            f"{p5:5.3f} | {p50:5.3f} | {p95:6.3f} | {above:5.1f}% |"
        )

    band_top = at(selection.Z_BAND_TOP) < selection.XHI_MAX_BAND_TOP
    x_ref = at(selection.Z_REION_REF)

    print(
        f"{T21.shape[0]} models, {n_f} channels, foreground floor at "
        f"N = {mp.N_ANCHOR}: {fg_resid[mp.N_ANCHOR]:.3f} mK\n"
    )

    print(
        f"xHI(z={selection.Z_REION_REF}) is strongly bimodal -- the threshold "
        "sits in a nearly empty valley:"
    )
    for lim in (0.06, 0.10, 0.20, 0.25):
        print(
            f"    below {lim:.2f}: {(x_ref < lim).sum():4d} "
            f"({(x_ref < lim).mean() * 100:.1f}%)"
        )

    ladders = {}
    for tag, use_sigma in (("fiducial", False), ("+1 sigma", True)):
        keep = band_top.copy()
        for z, lim, err in DAVIES_2025:
            keep &= at(z) < (lim + err if use_sigma else lim)
        ladders[tag] = keep

    print(
        f"\nRetained RMS [mK] and above-floor fraction quoted at "
        f"N = {mp.N_ANCHOR}; N* is the operating point each variant would "
        "select on its own.\n"
    )
    head = ("cut", "keep", "N*", "p5", "p50", "p95", "above")
    print(
        f"| {head[0]:<38s} | {head[1]:>4s} | {head[2]:>2s} | {head[3]:>5s} "
        f"| {head[4]:>5s} | {head[5]:>6s} | {head[6]:>6s} |"
    )
    print(f"|{'-' * 40}|{'-' * 6}|{'-' * 4}|{'-' * 7}|{'-' * 7}|{'-' * 8}|{'-' * 7}|")
    report("McGreer+2015 strict, x(5.9) < 0.06", (x_ref < 0.06) & band_top)
    report("adopted, x(5.9) < 0.1", (x_ref < selection.XHI_MAX) & band_top)
    report("Davies+2025 ladder, fiducial", ladders["fiducial"])
    report("Davies+2025 ladder, +1 sigma", ladders["+1 sigma"])
    report("relaxed, x(5.9) < 0.2", (x_ref < 0.2) & band_top)
    report("relaxed, x(5.9) < 0.25", (x_ref < 0.25) & band_top)
    report("band-top limb only, no z ~ 6 anchor", band_top)

    # Direction of the residual bias. The models the adopted cut drops but
    # the current limit still allows are not a high-signal tail being
    # suppressed -- they retain *less* than the kept ensemble, so the
    # adopted cut is marginally favourable to the above-floor fraction the
    # paper reports. Quantified rather than asserted.
    kept = (x_ref < selection.XHI_MAX) & band_top
    restored = band_top & (x_ref >= RESTORE_LO) & (x_ref < RESTORE_HI)
    r, k = ret_all[mp.N_ANCHOR][restored], ret_all[mp.N_ANCHOR][kept]
    print(
        f"\n{restored.sum()} models sit in xHI(5.9) = "
        f"[{RESTORE_LO}, {RESTORE_HI}) and pass the band-top limb -- "
        "dropped by the\nadopted cut, allowed by Davies+2025. They retain "
        "less, not more:"
    )
    print(
        f"    restored: p5 = {np.percentile(r, 5):.2f}, "
        f"p50 = {np.median(r):.2f}, p95 = {np.percentile(r, 95):.2f} mK, "
        f"{(r > fg_resid[mp.N_ANCHOR]).mean() * 100:.1f}% above floor"
    )
    print(
        f"    kept:     p5 = {np.percentile(k, 5):.2f}, "
        f"p50 = {np.median(k):.2f}, p95 = {np.percentile(k, 95):.2f} mK, "
        f"{(k > fg_resid[mp.N_ANCHOR]).mean() * 100:.1f}% above floor"
    )
    worst = np.abs(T21[restored][:, -1]).max() * 1e3
    print(
        f"    worst |T21(250 MHz)| among them: {worst:.3f} mK, against "
        "verify_ensemble.py's < 1.0 mK gate"
    )


if __name__ == "__main__":
    main()
