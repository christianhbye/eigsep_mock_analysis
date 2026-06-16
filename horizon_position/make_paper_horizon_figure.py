"""Build the paper horizon-shift / foreground-filtering figure.

Writes three artifacts into the eigsep_instrument paper notebooks dir
(the repo convention: a committed npz + a notebook that regenerates the
PDF, data archived to Zenodo):

* ``horizon_shift.npz``  -- the +1 m East/North/Up antenna-temperature
  differences Delta T_ant(nu) at 24 LSTs (one per hour, the curves of
  ``horizon_shift.pdf``), plus the foreground spectral modes ``Vh`` (the
  right singular vectors of the nominal antenna-temperature waterfall --
  the ``foreground_svd.npz`` system temperature minus the constant
  receiver temperature, the same modes as Fig. 1).
* ``horizon_shift.ipynb`` -- loads the npz and makes the figure.
* ``horizon_shift.pdf``  -- the rendered figure.

The figure (2 x 3, East/North/Up) pairs, per axis, the Delta T(nu)
spectra (top, coloured by LST) with their residual after *filtering* the
leading N foreground modes (bottom, same per-LST colours). The residual
panel matches ``foreground_svd_residual.pdf``: log-y Residual RMS [K],
10 mK red dashed reference, "Foreground modes filtered" x-axis. It shows
the position-error signal is removed by the same low-order foreground
filtering as the sky: every LST/axis drops below 10 mK within ~6 modes.

Run in the mock_analysis env (numpy + matplotlib + astropy + nbformat):
    uv run python horizon_position/make_paper_horizon_figure.py
"""

import os
import sys
from pathlib import Path

import astropy.units as u
import nbformat as nbf
import numpy as np
from astropy.time import Time
from astropy.utils import iers

iers.conf.auto_download = False
iers.conf.auto_max_age = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis  # noqa: E402

SIMS = HERE / "output" / "position_sims.npz"
PAPER = Path("/home/christian/Documents/research/papers/eigsep_instrument/notebooks")
FG_NPZ = PAPER / "foreground_svd.npz"

TAGS = [("x_p_1", "East +1 m"), ("y_p_1", "North +1 m"), ("z_p_1", "Up +1 m")]


def build_data():
    d = np.load(SIMS, allow_pickle=True)
    t_sys, fgnd = d["t_sys"], d["fgnd"]
    names = [str(n) for n in d["names"]]
    freqs = d["freqs_mhz"]
    t_gnd, t_rcvr = float(d["t_ground"]), float(d["t_receiver"])

    # foreground spectral modes: uncentered SVD of the nominal antenna-
    # temperature waterfall (system temperature minus the constant receiver
    # temperature) -- the *same* modes as foreground_svd.npz / Fig. 1.
    fg = np.load(FG_NPZ)
    assert np.array_equal(t_sys[0], fg["t_sys"]), (
        "baseline mismatch vs foreground_svd.npz"
    )
    _, _, Vh = np.linalg.svd(t_sys[0] - t_rcvr, full_matrices=False)

    # uncorrected dT at 24 LSTs (one per hour) -- the horizon_shift.pdf curves
    dT = analysis.delta_waterfall(t_sys, fgnd, "uncorrected", t_gnd, t_rcvr)
    lst = (
        Time(d["times_jd"], format="jd", scale="utc")
        .sidereal_time("apparent", longitude=float(d["lon"]) * u.deg)
        .hour
    )
    idx = [int(np.argmin(np.abs((lst - h + 12) % 24 - 12))) for h in range(24)]
    dT_spectra = np.stack([dT[names.index(t)][idx] for t, _ in TAGS])  # (3, 24, n_f)
    labels = np.array([lab for _, lab in TAGS])

    PAPER.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PAPER / "horizon_shift.npz",
        dT_spectra=dT_spectra,
        Vh=Vh,
        freqs_MHz=freqs,
        lst_hr=lst[idx],
        labels=labels,
        description=(
            "Uncorrected antenna-temperature differences dT_ant(nu) for +1 m "
            "East/North/Up antenna displacements at 24 LSTs (one per hour), and "
            "the foreground spectral modes Vh (right singular vectors of the "
            "nominal antenna-temperature waterfall = foreground_svd.npz system "
            "temperature minus the constant receiver). dT_spectra (3, 24, "
            "n_freq) in K; axis order East/North/Up."
        ),
    )
    print(f"wrote {PAPER / 'horizon_shift.npz'}")


# --- source shared by the notebook and the direct render (kept in sync) ---

IMPORTS_SRC = """import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize"""

LOAD_SRC = """d = np.load("horizon_shift.npz", allow_pickle=True)
freqs = d["freqs_MHz"]
lst = d["lst_hr"]                 # LST [h] of each plotted spectrum
dT = d["dT_spectra"]             # (3, n_lst, n_freq) uncorrected dT_ant [K]
Vh = d["Vh"]                     # (n_freq, n_freq) foreground spectral modes
labels = [str(s) for s in d["labels"]]
n_f = freqs.size
N_SHOW = 18                       # foreground modes filtered (x-axis)
print(dT.shape, "spectra at LSTs", np.round(lst, 1))"""

PLOT_SRC = '''CMAP, norm = "twilight", Normalize(0, 24)
cmap = plt.get_cmap(CMAP)
n_modes = np.arange(N_SHOW + 1)


def resid_curves(dT_axis):
    """Per-LST residual RMS over freq [K] after filtering the leading N modes."""
    coeff = dT_axis @ Vh.T                                  # (n_lst, n_freq)
    return np.array([np.sqrt(np.sum(coeff[:, N:] ** 2, axis=1) / n_f)
                     for N in n_modes])                     # (N_SHOW+1, n_lst)


fig, axes = plt.subplots(
    2, 3, figsize=(7.3, 3.5),
    gridspec_kw=dict(height_ratios=[1.7, 1]),
    layout="constrained",
)
for col, lab in enumerate(labels):
    at, ab = axes[0, col], axes[1, col]
    for j in range(lst.size):                               # top: dT(nu) spectra
        at.plot(freqs, dT[col, j], color=cmap(norm(lst[j])), lw=0.7, alpha=0.9)
    at.axhline(0, color="0.5", lw=0.6, ls="--", zorder=0)
    at.set_title(lab, fontsize=8.5)
    at.set_xlabel("Frequency [MHz]", fontsize=8)
    at.grid(alpha=0.2); at.tick_params(labelsize=7)

    rc = resid_curves(dT[col])                              # bottom: filtered residual
    for j in range(lst.size):
        ab.plot(n_modes, rc[:, j], color=cmap(norm(lst[j])), lw=0.7, alpha=0.9)
    ab.axhline(1e-2, color="red", lw=1.2, ls="--", alpha=0.8)
    ab.set_yscale("log")
    ab.set_xlabel("Foreground modes filtered", fontsize=8)
    ab.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ab.set_xlim(0, N_SHOW); ab.set_ylim(1e-4, 5); ab.tick_params(labelsize=7)

axes[0, 0].set_ylabel(r"$\\Delta T_\\mathrm{ant}$ [K]", fontsize=8)
axes[1, 0].set_ylabel("Residual RMS [K]", fontsize=8)
axes[1, 2].text(N_SHOW, 1.2e-2, "10 mK", color="red", alpha=0.8,
                fontsize=7, va="bottom", ha="right")
for col in (1, 2):
    axes[1, col].tick_params(labelleft=False)

sm = ScalarMappable(norm=norm, cmap=CMAP)
cb = fig.colorbar(sm, ax=axes, pad=0.012, fraction=0.022)
cb.set_label("LST [h]", fontsize=8); cb.set_ticks(np.arange(0, 25, 4))
cb.ax.tick_params(labelsize=7)
fig.savefig("horizon_shift.pdf", bbox_inches="tight", dpi=600)'''

SUMMARY_SRC = """for col, lab in enumerate(labels):
    rc = resid_curves(dT[col])               # (N_SHOW+1, n_lst)
    worst = rc.max(axis=1)                    # worst LST at each N
    n10 = int(np.nonzero(worst < 1e-2)[0][0])
    print(f"{lab:11s}  full RMS {worst[0]*1e3:7.1f} mK  ->  "
          f"all LSTs < 10 mK after {n10} modes; "
          f"residual after 10 modes {worst[10]*1e3:.2f} mK")"""


def build_notebook():
    md = (
        "# Horizon-shift signal vs foreground filtering\n\n"
        "Antenna-position error changes the antenna temperature by "
        "$\\Delta T_\\mathrm{ant}(\\nu)$. Here we ask how *foreground-like* "
        "that change is: we project the $\\Delta T_\\mathrm{ant}$ spectra (top "
        "row, +1 m East/North/Up at 24 LSTs) onto the foreground spectral "
        "modes -- the same SVD modes as `foreground_svd.npz` -- and plot the "
        "residual RMS after filtering the leading $N$ modes (bottom row). The "
        "signal sits in the same low-order foreground subspace: every LST and "
        "axis falls below 10 mK within a handful of modes, so the same "
        "foreground filtering that cleans the sky also removes the "
        "position-error systematic."
    )
    cells = [
        nbf.v4.new_markdown_cell(md),
        nbf.v4.new_code_cell(IMPORTS_SRC),
        nbf.v4.new_code_cell(LOAD_SRC),
        nbf.v4.new_code_cell(PLOT_SRC),
        nbf.v4.new_code_cell(SUMMARY_SRC),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nbf.write(nb, PAPER / "horizon_shift.ipynb")
    print(f"wrote {PAPER / 'horizon_shift.ipynb'}")


def render_pdf():
    """Run the notebook's source here so the PDF and notebook stay identical."""
    ns = {}
    cwd = os.getcwd()
    os.chdir(PAPER)
    try:
        exec(IMPORTS_SRC, ns)
        exec(LOAD_SRC, ns)
        exec(PLOT_SRC, ns)
        exec(SUMMARY_SRC, ns)
    finally:
        os.chdir(cwd)
    print(f"wrote {PAPER / 'horizon_shift.pdf'}")


def main():
    if not SIMS.exists():
        raise SystemExit(f"{SIMS} not found - run run_sims.py first")
    if not FG_NPZ.exists():
        raise SystemExit(f"{FG_NPZ} not found - run foreground_svd export first")
    build_data()
    build_notebook()
    render_pdf()


if __name__ == "__main__":
    main()
