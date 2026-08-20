"""Recompute PR #5's operating point against the Zeus21 ensemble.

    uv run python horizon_position/recompute_operating_point.py

N_ANCHOR is the smallest N at which the foreground residual falls below
the median retained 21 cm signal *and stays below* for every larger N in
the scan. Only the foreground floor enters the criterion: Fig. 1 is
scoped to foreground dimensionality alone, and the position systematic
belongs to the forward-modelling section, where horizon_shift.pdf shows
it against the same benchmark.

The "stays below" clause is load-bearing. Both curves fall with N, and
they cross more than once, so a first-crossing rule can select an N that
the floor later climbs back above -- which is exactly what happens to
the position systematic (below the median at N = 7 and 8, above it again
at N = 9). The position systematic is still reported here, under the
same rule, so the two figures cannot disagree about which N clears it.
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
    # horizon_shift.npz now carries every simulated magnitude; the operating
    # point is quoted against the one the figure's spectra row draws.
    mags = shift["mags_m"]
    i_top = int(np.argmin(np.abs(mags - float(shift["top_mag_m"]))))
    dT = shift["dT_disp"][:, i_top].reshape(-1, n_f) @ Vh.T
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

    scan = list(N_SCAN)
    fg_r = np.array([rms_pooled(fg_c, N, n_time) for N in scan])
    sys_r = np.array([rms(dT, N).max() for N in scan])
    med = np.array([np.median(rms(t21_c, N)) for N in scan])

    def first_stable(floor):
        """Smallest scanned N with floor < median there and at every larger N."""
        below = floor < med
        for i in range(len(scan)):
            if below[i:].all():
                return scan[i]
        return None

    chosen, sys_n = first_stable(fg_r), first_stable(sys_r)

    print(f"{len(T21)} models on {n_f} channels, {freqs[0]:.0f}-{freqs[-1]:.0f} MHz\n")
    print(
        f"{'N':>3}  {'fg resid':>9}  {'pos sys':>9}  {'21cm med':>9}  {'above fg':>9}"
    )
    for i, N in enumerate(scan):
        above = (rms(t21_c, N) > fg_r[i]).mean()
        flag = ""
        if N == chosen:
            flag = "  <- N_ANCHOR: foreground floor below the median from here on"
        elif N == sys_n:
            flag = "  <- position systematic below the median from here on"
        print(
            f"{N:>3}  {fg_r[i]:>9.3f}  {sys_r[i]:>9.3f}  {med[i]:>9.3f}  "
            f"{100 * above:>8.1f}%{flag}"
        )

    # Without this, `chosen` stays None and `rms(t21_c, None)` silently
    # slices every column (no filtering at all) below, printing plausible
    # but meaningless percentiles instead of failing loudly.
    assert chosen is not None, (
        f"no N in {N_SCAN} puts the foreground residual below the median "
        "retained signal and keeps it there -- widen N_SCAN"
    )
    assert sys_n is not None, (
        f"no N in {N_SCAN} puts the position systematic below the median "
        "retained signal and keeps it there -- widen N_SCAN"
    )
    print(
        f"\nfolding in the +1 m position systematic costs {sys_n - chosen} "
        f"extra mode(s): N {chosen} -> {sys_n}"
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
