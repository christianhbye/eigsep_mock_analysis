"""Recompute PR #5's operating point against the Zeus21 ensemble.

    uv run python horizon_position/recompute_operating_point.py

N_ANCHOR is defined as the smallest N at which both floors -- the
foreground residual and the position systematic -- fall below the median
retained 21 cm signal. That definition is unchanged; the ensemble it is
evaluated against is not, so the answer may move.
"""

import make_paper_signal_loss_figure as mp
import numpy as np

PAPER = mp.PAPER

# Scan range for N_ANCHOR. Starting at 4 rather than 0 is an assumption --
# a reviewer's independent N = 0..19 sweep on the Zeus21 ensemble confirmed
# no crossing exists below N = 4, but that isn't guaranteed to hold for a
# future ensemble. Revisit (widen downward) if `chosen` below ever lands
# suspiciously close to N_SCAN's lower bound.
N_SCAN = range(4, 17)


def main():
    fg = np.load(PAPER / "foreground_svd.npz")
    shift = np.load(PAPER / "horizon_shift.npz")
    freqs, Vh = fg["freqs_MHz"], shift["Vh"]
    assert np.array_equal(shift["freqs_MHz"], freqs), "frequency grid mismatch"

    T21 = mp.load_t21(freqs)
    n_f = freqs.size

    # Antenna temperature, receiver dropped -- Vh (loaded above, from
    # horizon_shift.npz) was itself derived by SVD of this same quantity
    # (see make_paper_horizon_figure.build_data / make_paper_signal_loss_
    # figure.build_data), so the foreground floor must be measured in the
    # basis it was built for. Using the raw t_sys leaves the ~50 K receiver
    # constant unremoved, which the Vh basis was never optimized to
    # compress, inflating the apparent residual by orders of magnitude and
    # preventing N_ANCHOR from ever resolving.
    t_ant = fg["t_sys"] - fg["t_receiver"]
    n_time = t_ant.shape[0]
    fg_c = t_ant.reshape(-1, n_f) @ Vh.T
    dT = shift["dT_spectra"].reshape(-1, n_f) @ Vh.T
    t21_c = T21 @ Vh.T

    def rms(c, N):
        """Worst single row (LST/axis) after filtering the leading N modes."""
        return np.sqrt(np.sum(c[:, N:] ** 2, axis=1) / n_f) * 1e3

    def rms_pooled(c, N, n_rows):
        """RMS pooled over every row and channel after filtering N modes.

        Equivalent (Parseval) to make_paper_signal_loss_figure.py's CALC_SRC
        `fg_resid = sqrt(tail / (n_time * n_f))` built from the singular
        values of the same t_ant -- i.e. the curve actually plotted and
        quoted as the foreground floor, not a worst-single-LST bound. The
        position systematic keeps the worst-case (per-row) form below,
        matching CALC_SRC's `sys_resid`, which is deliberately worst-axis.
        """
        return np.sqrt(np.sum(c[:, N:] ** 2) / (n_rows * n_f)) * 1e3

    print(f"{len(T21)} models on {n_f} channels, {freqs[0]:.0f}-{freqs[-1]:.0f} MHz\n")
    print(
        f"{'N':>3}  {'fg resid':>9}  {'pos sys':>9}  {'21cm med':>9}  {'above fg':>9}"
    )
    chosen = None
    for N in N_SCAN:
        f_r, s_r = rms_pooled(fg_c, N, n_time), rms(dT, N).max()
        med = np.median(rms(t21_c, N))
        above = (rms(t21_c, N) > f_r).mean()
        flag = ""
        if chosen is None and f_r < med and s_r < med:
            chosen, flag = N, "  <- smallest N with both floors below the median"
        print(
            f"{N:>3}  {f_r:>9.3f}  {s_r:>9.3f}  {med:>9.3f}  {100 * above:>8.1f}%{flag}"
        )

    # Without this, `chosen` stays None and `rms(t21_c, None)` silently
    # slices every column (no filtering at all) below, printing plausible
    # but meaningless percentiles instead of failing loudly.
    assert chosen is not None, (
        f"no N in {N_SCAN} satisfies both floors < median retained signal "
        "-- widen N_SCAN"
    )

    print(f"\nN_ANCHOR = {chosen}")
    ret = rms(t21_c, chosen)
    print(
        f"retained RMS at N = {chosen}: "
        f"p5={np.percentile(ret, 5):.3f} p50={np.percentile(ret, 50):.3f} "
        f"p95={np.percentile(ret, 95):.3f} mK, max={ret.max():.3f}"
    )
    for edge in (1.0, 5.0, 10.0, 25.0):
        print(
            f"  fraction retaining > {edge:>5.1f} mK: {100 * (ret > edge).mean():5.2f}%"
        )
    return chosen


if __name__ == "__main__":
    main()
