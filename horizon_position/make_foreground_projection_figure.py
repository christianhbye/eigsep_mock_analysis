"""How foreground-like is the horizon-shift signal? -- two paper figures.

The horizon-shift spectra (``horizon_shift.pdf``: raw uncorrected
Delta T_ant(nu) at 24 LSTs for +1 m East/North/Up displacements) are
projected onto the *foreground* spectral eigenbasis. The basis is the
right singular vectors ``V`` of the nominal, uncorrected antenna-
temperature waterfall ``t_sys[0]`` (uncentered SVD, as in
``svd_baseline.py`` -- the mean spectrum is the dominant foreground and
counts as mode 1). Both the basis and the Delta T are uncorrected
antenna temperatures, so the comparison is self-consistent.

Result: the Delta T is almost entirely foreground. ~85-90% of its power
sits in the single dominant foreground mode, >99.9% in the first five,
and removing the leading ``N`` foreground modes (chosen as the smallest
N that drives all three +1 m axes below a 1 mK residual -- N=10, the
foreground singular-value knee) collapses the residual to <~0.3 mK,
below any realistic noise floor. Antenna-position error therefore
projects onto the same low-order subspace as the foreground and is
removed by the same cleaning.

Two figures (each 2 rows x 3 axes, sharing the top row with
``horizon_shift.pdf``):

* ``horizon_shift_residual.pdf`` -- bottom row: residual Delta T(nu)
  after projecting out the leading N foreground modes (frequency basis,
  both rows).
* ``horizon_shift_eigenbasis.pdf`` -- bottom row: the same Delta T in
  the foreground eigenmode basis (|coefficient| vs mode index), showing
  the power piled into the low-order foreground modes.
* ``horizon_shift_cumulative.pdf`` -- bottom row: the cumulative
  residual RMS vs number of foreground modes removed (the
  ``fig_svd_residual.pdf`` style), with the baseline foreground's own
  residual curve overlaid -- the Delta T residual falls at the same rate
  as the foreground and, starting lower, crosses 1 mK at or before the
  foreground's own crossing (N~11): East 7, North 6, Up 10 modes.

Standalone; default env (numpy + matplotlib + astropy):

    uv run python horizon_position/make_foreground_projection_figure.py
"""

import sys
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from astropy.utils import iers
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis  # noqa: E402

iers.conf.auto_download = False
iers.conf.auto_max_age = None  # sub-second UT1-UTC on the sim dates

SIMS_FILE = HERE / "output" / "position_sims.npz"

ONE_MK = 1e-3  # 1 mK residual reference (K)
N_SHOW = 25  # eigenmodes shown on the mode-index axis
CMAP = "twilight"
DIRS = [  # (position tag, panel title)
    ("x_p_1", "East +1 m"),
    ("y_p_1", "North +1 m"),
    ("z_p_1", "Up +1 m"),
]


def load():
    d = np.load(SIMS_FILE, allow_pickle=True)
    t_sys, fgnd = d["t_sys"], d["fgnd"]
    names = [str(n) for n in d["names"]]
    freqs = d["freqs_mhz"]
    t_gnd, t_rcvr = float(d["t_ground"]), float(d["t_receiver"])
    lst = (
        Time(d["times_jd"], format="jd", scale="utc")
        .sidereal_time("apparent", longitude=float(d["lon"]) * u.deg)
        .hour
    )
    dT = analysis.delta_waterfall(t_sys, fgnd, "uncorrected", t_gnd, t_rcvr)
    return t_sys, dT, names, freqs, lst


def foreground_basis(t_sys):
    """Uncentered SVD of the nominal uncorrected waterfall.

    Returns the frequency-space right singular vectors ``Vh`` (the
    foreground spectral modes) and the singular values ``s``.
    """
    _, s, Vh = np.linalg.svd(t_sys[0], full_matrices=False)  # Vh: (n_f, n_f)
    return Vh, s


def choose_n_modes(dT, Vh, names, thresh=ONE_MK):
    """Smallest N driving all DIRS' residual RMS below ``thresh`` (K)."""
    n_f = Vh.shape[0]
    for N in range(n_f + 1):
        worst = 0.0
        for tag, _ in DIRS:
            coeff = dT[names.index(tag)] @ Vh.T
            worst = max(worst, np.sqrt(np.mean(coeff[:, N:] ** 2)))
        if worst < thresh:
            return N
    return n_f


def lst_indices(lst, n=24):
    """One waterfall row per integer LST hour (0..23)."""
    return [int(np.argmin(np.abs((lst - h + 12) % 24 - 12))) for h in range(n)]


def _top_spectra(ax, freqs, dT_i, idx, lst, norm):
    """Top-row panel: Delta T(nu) [K] coloured by LST (matches horizon_shift)."""
    cmap = plt.get_cmap(CMAP)
    for ti in idx:
        ax.plot(freqs, dT_i[ti], color=cmap(norm(lst[ti])), lw=0.9, alpha=0.9)
    ax.axhline(0, color="0.5", lw=0.8, ls="--", zorder=0)
    ax.grid(alpha=0.2)


def _resid_spectra(ax, freqs, resid_i, idx, lst, norm):
    """Bottom-row panel (Fig. A): residual Delta T(nu) in mK."""
    cmap = plt.get_cmap(CMAP)
    for ti in idx:
        ax.plot(freqs, resid_i[ti] * 1e3, color=cmap(norm(lst[ti])), lw=0.9, alpha=0.9)
    ax.axhline(0, color="0.5", lw=0.8, ls="--", zorder=0)
    ax.grid(alpha=0.2)
    rms = np.sqrt(np.mean(resid_i**2)) * 1e3
    ax.text(
        0.96,
        0.06,
        f"residual RMS\n{rms:.2f} mK",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9),
    )


def _eigen_spectra(ax, coeff_i, idx, lst, norm, n_cut):
    """Bottom-row panel (Fig. B): |coefficient| vs foreground-mode index."""
    cmap = plt.get_cmap(CMAP)
    modes = np.arange(1, N_SHOW + 1)
    ax.axvspan(
        0.5, n_cut + 0.5, color="0.85", alpha=0.5, zorder=0
    )  # removed by cleaning
    for ti in idx:
        ax.semilogy(
            modes,
            np.abs(coeff_i[ti, :N_SHOW]),
            color=cmap(norm(lst[ti])),
            lw=0.9,
            alpha=0.9,
        )
    ax.axvline(n_cut + 0.5, color="0.35", ls=":", lw=1.1, zorder=5)
    ax.text(
        0.5 * (n_cut + 1),
        0.04,
        f"removed by\nFG cleaning\n($N={n_cut}$)",
        transform=ax.get_xaxis_transform(),
        va="bottom",
        ha="center",
        fontsize=7.5,
        color="0.3",
        linespacing=1.3,
    )
    ax.grid(alpha=0.2, which="both")
    ax.set_xlim(0.5, N_SHOW + 0.5)


def build_residual_figure(freqs, dT, resid, names, lst, idx, norm, n_cut):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True, layout="constrained")
    for col, (tag, title) in enumerate(DIRS):
        i = names.index(tag)
        _top_spectra(axes[0, col], freqs, dT[i], idx, lst, norm)
        axes[0, col].set_title(title)
        _resid_spectra(axes[1, col], freqs, resid[i], idx, lst, norm)
        axes[1, col].set_xlabel("Frequency [MHz]")
    axes[0, 0].set_ylabel(r"$\Delta T_\mathrm{ant}$ [K]")
    axes[1, 0].set_ylabel(
        rf"residual $\Delta T_\mathrm{{ant}}$ [mK]"
        "\n"
        rf"($N={n_cut}$ FG modes removed)"
    )
    for ax, lab in zip(axes.ravel(), "abcdef"):
        ax.text(0.012, 0.94, f"({lab})", transform=ax.transAxes, va="top", fontsize=10)
    _add_cbar(fig, axes, norm)
    return fig


N_MAX_RESID = 25  # x-axis range of the cumulative-residual panels


def dT_residual_curve(coeff_i, n_max=N_MAX_RESID):
    """Residual RMS over (LST, freq) after removing leading N FG modes, N=0..n_max."""
    return np.array([np.sqrt(np.mean(coeff_i[:, N:] ** 2)) for N in range(n_max + 1)])


def fg_residual_curve(s, n_t, n_f, n_max=N_MAX_RESID):
    """Baseline foreground self-residual RMS after removing N modes (svd_baseline)."""
    tail = np.concatenate([np.cumsum((s**2)[::-1])[::-1], [0.0]])  # tail[N]=sum_{i>=N}
    return np.sqrt(tail[: n_max + 1] / (n_t * n_f))


def _resid_curve_panel(ax, resid_dT, resid_fg, label_fg=False):
    """Bottom-row panel (Fig. C): cumulative residual RMS vs FG modes removed."""
    n = np.arange(N_MAX_RESID + 1)
    ax.semilogy(
        n,
        resid_fg,
        "-",
        color="0.7",
        lw=1.0,
        zorder=1,
        label="baseline foreground" if label_fg else None,
    )
    ax.semilogy(n, resid_dT, "o-", color="k", ms=3.5, lw=1.2, zorder=3)
    ax.axhline(ONE_MK, color="0.5", ls="--", lw=1, zorder=0)
    ax.text(
        N_MAX_RESID, ONE_MK, " 1 mK", color="0.4", va="bottom", ha="right", fontsize=8.5
    )

    below = np.nonzero(resid_dT < ONE_MK)[0]
    if below.size:
        n_below = int(below[0])
        ax.axvline(n_below, color="0.7", ls=":", lw=1, zorder=0)
        ax.annotate(
            f"{n_below} modes\n$\\to$ < 1 mK",
            xy=(n_below, ONE_MK),
            xytext=(n_below + 1.5, 80 * ONE_MK),
            color="0.3",
            fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8),
        )
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.set_xlim(0, N_MAX_RESID)
    ax.set_ylim(3e-5, 3e3)
    if label_fg:
        ax.legend(loc="upper right", fontsize=8, frameon=False)


def build_cumulative_figure(freqs, dT, coeff, s, names, lst, idx, norm):
    n_t, n_f = dT.shape[1], dT.shape[2]
    resid_fg = fg_residual_curve(s, n_t, n_f)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), layout="constrained")
    for col, (tag, title) in enumerate(DIRS):
        i = names.index(tag)
        _top_spectra(axes[0, col], freqs, dT[i], idx, lst, norm)
        axes[0, col].set_title(title)
        axes[0, col].set_xlabel("Frequency [MHz]")
        resid_dT = dT_residual_curve(coeff[i])
        _resid_curve_panel(axes[1, col], resid_dT, resid_fg, label_fg=(col == 0))
        axes[1, col].set_xlabel("Foreground modes removed, $N$")
    axes[0, 0].set_ylabel(r"$\Delta T_\mathrm{ant}$ [K]")
    axes[1, 0].set_ylabel("Residual RMS [K]")
    for ax, lab in zip(axes.ravel(), "abcdef"):
        ax.text(0.012, 0.94, f"({lab})", transform=ax.transAxes, va="top", fontsize=10)
    _add_cbar(fig, axes, norm)
    return fig


def build_eigenbasis_figure(freqs, dT, coeff, names, lst, idx, norm, n_cut):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), layout="constrained")
    for col, (tag, title) in enumerate(DIRS):
        i = names.index(tag)
        _top_spectra(axes[0, col], freqs, dT[i], idx, lst, norm)
        axes[0, col].set_title(title)
        axes[0, col].set_xlabel("Frequency [MHz]")
        _eigen_spectra(axes[1, col], coeff[i], idx, lst, norm, n_cut)
        axes[1, col].set_xlabel("Foreground mode index")
    axes[0, 0].set_ylabel(r"$\Delta T_\mathrm{ant}$ [K]")
    axes[1, 0].set_ylabel(r"$|\,\Delta T_\mathrm{ant}$ coeff$\,|$ [K]")
    for ax, lab in zip(axes.ravel(), "abcdef"):
        ax.text(0.012, 0.94, f"({lab})", transform=ax.transAxes, va="top", fontsize=10)
    _add_cbar(fig, axes, norm)
    return fig


def _add_cbar(fig, axes, norm):
    sm = ScalarMappable(norm=norm, cmap=CMAP)
    cb = fig.colorbar(sm, ax=axes, pad=0.015, fraction=0.035)
    cb.set_label("LST [h]")
    cb.set_ticks(np.arange(0, 25, 4))


def main():
    if not SIMS_FILE.exists():
        raise SystemExit(f"{SIMS_FILE} not found - run run_sims.py first")

    t_sys, dT, names, freqs, lst = load()
    Vh, s = foreground_basis(t_sys)
    idx = lst_indices(lst)
    norm = Normalize(0, 24)

    n_cut = choose_n_modes(dT, Vh, names)
    print(f"N foreground modes removed (all axes < 1 mK residual): {n_cut}")

    # projection coefficients and residual waterfalls for every position
    coeff = dT @ Vh.T  # (P, n_t, n_f) in mode basis
    resid = coeff[..., n_cut:] @ Vh[n_cut:]  # (P, n_t, n_f) frequency basis
    for tag, _ in DIRS:
        i = names.index(tag)
        full = np.sqrt(np.mean(dT[i] ** 2)) * 1e3
        r = np.sqrt(np.mean(resid[i] ** 2)) * 1e3
        print(f"  {tag:8s}  full RMS {full:8.2f} mK  ->  residual {r:7.3f} mK")

    fig = build_residual_figure(freqs, dT, resid, names, lst, idx, norm, n_cut)
    fig.savefig(HERE / "horizon_shift_residual.pdf")
    fig.savefig(HERE / "horizon_shift_residual.png", dpi=160)
    plt.close(fig)

    fig = build_eigenbasis_figure(freqs, dT, coeff, names, lst, idx, norm, n_cut)
    fig.savefig(HERE / "horizon_shift_eigenbasis.pdf")
    fig.savefig(HERE / "horizon_shift_eigenbasis.png", dpi=160)
    plt.close(fig)

    fig = build_cumulative_figure(freqs, dT, coeff, s, names, lst, idx, norm)
    fig.savefig(HERE / "horizon_shift_cumulative.pdf")
    fig.savefig(HERE / "horizon_shift_cumulative.png", dpi=160)
    plt.close(fig)

    print("wrote horizon_shift_{residual,eigenbasis,cumulative}.{pdf,png}")


if __name__ == "__main__":
    main()
