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
panel is styled like ``foreground_svd_residual.pdf`` (log-y Residual RMS
[K], "Foreground modes filtered" x-axis), but its reference is now the
retained 21 cm signal (5-95% band, median dashed) under the identical
projection -- the same benchmark as ``signal_loss.pdf`` -- in place of
an arbitrary 10 mK line.

The bottom row exists to make two points at once, and they pull opposite
ways.

*Reassuring.* A position error is not a new kind of spectral structure.
The Delta T are large -- up to 8.9 K at 50 MHz, ~1 K RMS for the upward
shift -- but 99.9% of that power lies in the two leading eigenmodes of
the unperturbed antenna temperature. It is, to that accuracy, more
foreground, and the same low-order filtering that removes the sky
removes almost all of it; East and North clear the median retained
21 cm signal after 3 and 2 modes.

*Cautionary.* What survives is narrow, not smooth. After N_ANCHOR
modes, 99% of the Up displacement's remaining power sits in mode 10
alone, at 3.18 mK -- more than that mode holds from the nominal
foregrounds (1.71 mK) or from the median 21 cm model (1.66 mK), and
similar enough in shape to be confused with the retained signal (cosine
similarity up to 0.99). 73% of the ensemble carries less signal in that
mode than a 1 m shift would put there. The *vertical* response is linear
in displacement (~2% per decade over +/-0.1 to +/-10 m) and symmetric in
sign, because a vertical shift lowers the horizon by a near-uniform
offset; East and North are neither, since a horizontal shift moves the
horizon by an amount set by where the cliff edges fall in azimuth. Up is
the binding axis at every N, so the extrapolation is sound and the
requirement is a vertical one: about 0.1 m to hold the injection to a
tenth of the median retained signal.

Hence the framing this figure must keep, and which the paper text
(signal_loss_text.tex, block 5) states explicitly: neither this figure
nor signal_loss.pdf is a proposed analysis. Both project onto
eigenmodes of a *simulated* nominal instrument. An unmodelled
displacement of the size we have to anticipate deposits signal-like
power in the first mode past the filter, so a residual left by a filter
of fixed depth is not evidence of a cosmological signal. These are
characterisations of spectral structure; the analysis marginalises over
antenna position inside a forward model instead.

This figure is where the position systematic is folded into the mode
budget, and it is deliberately the only place. ``signal_loss.pdf``
(Fig. 1) sets its operating point ``N_ANCHOR`` on the foreground
residual alone, because that is all Fig. 1 claims; the dotted line in
the residual panels marks it here so the two figures read against a
common reference. The cost of adding this systematic is one mode: at
N_ANCHOR the worst-LST Up displacement still sits just above the median
retained signal (3.20 vs 3.01 mK), and one further mode drops it an
order of magnitude below (0.33 vs 2.18 mK).

Crossings are counted with a *stays below* rule -- the smallest N at
which the worst-LST residual is under the median retained signal and
remains so for every larger N. Both curves fall with N, and they cross
more than once: Up is below the median at N = 7 and N = 8 and back above
it at N = 9, so a first-crossing rule would report 7 here while
``recompute_operating_point.py`` reports 10, on identical arrays.

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
from make_paper_signal_loss_figure import N_ANCHOR, load_t21, retained_pct  # noqa: E402

SIMS = HERE / "output" / "position_sims.npz"
PAPER = Path("/home/christian/Documents/research/papers/eigsep_instrument/notebooks")
FG_NPZ = PAPER / "foreground_svd.npz"

N_SHOW = 18  # modes on the residual-panel x-axis; matches LOAD_SRC

AXES_ENU = [("x", "East"), ("y", "North"), ("z", "Up")]
MAGS_M = (0.1, 1.0, 10.0)  # the positive displacements run_sims.py simulated
TOP_MAG = 1.0  # the one the spectra row shows; must be in MAGS_M


def _tag(axis, mag):
    """run_sims.py's position name for a positive displacement, e.g. z_p_0p1."""
    return f"{axis}_p_" + ("%g" % mag).replace(".", "p")


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

    # All three positive magnitudes, not just +1 m. The residual row encodes
    # displacement rather than LST: after filtering, the 24 LSTs collapse into
    # one bundle, so an LST colour scale there discriminates nothing, whereas
    # the three magnitudes separate by decades and let the reader scale the
    # result to whatever position knowledge they actually have. run_sims.py has
    # always produced these; they were simply unused. (3, 3, 24, n_f)
    dT_disp = np.stack(
        [
            np.stack([dT[names.index(_tag(ax, m))][idx] for m in MAGS_M])
            for ax, _ in AXES_ENU
        ]
    )
    labels = np.array([lab for _, lab in AXES_ENU])
    assert TOP_MAG in MAGS_M, "TOP_MAG must be one of the simulated magnitudes"

    # retained 21 cm signal under the same projection -- the physical
    # benchmark the residual panel is read against (see signal_loss.pdf)
    t21_pct = retained_pct(load_t21(freqs), Vh, np.arange(N_SHOW + 1))

    PAPER.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PAPER / "horizon_shift.npz",
        dT_disp=dT_disp,
        mags_m=np.array(MAGS_M),
        top_mag_m=TOP_MAG,
        Vh=Vh,
        freqs_MHz=freqs,
        lst_hr=lst[idx],
        labels=labels,
        t21_pct=t21_pct,
        n_anchor=N_ANCHOR,
        description=(
            "Uncorrected antenna-temperature differences dT_ant(nu) for "
            "East/North/Up antenna displacements at 24 LSTs (one per hour), and "
            "the foreground spectral modes Vh (right singular vectors of the "
            "nominal antenna-temperature waterfall = foreground_svd.npz system "
            "temperature minus the constant receiver). dT_disp (3 axis, 3 "
            "magnitude, 24 LST, n_freq) in K; axis order East/North/Up, "
            "magnitudes mags_m in metres, all positive displacements. "
            "top_mag_m is the magnitude the spectra row of the figure draws. "
            "n_anchor is the operating point signal_loss.pdf sets on the "
            "foreground residual alone, marked here so both figures read "
            "against one reference."
        ),
    )
    print(f"wrote {PAPER / 'horizon_shift.npz'}")


# --- source shared by the notebook and the direct render (kept in sync) ---

IMPORTS_SRC = """import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D"""

LOAD_SRC = """d = np.load("horizon_shift.npz", allow_pickle=True)
freqs = d["freqs_MHz"]
lst = d["lst_hr"]                 # LST [h] of each plotted spectrum
t21 = d["t21_pct"]                # (3, N_SHOW+1) retained 21 cm RMS [K], 5/50/95
dT_disp = d["dT_disp"]            # (3 axis, 3 mag, n_lst, n_freq) dT_ant [K]
mags = d["mags_m"]                # displacement magnitudes [m]
top_mag = float(d["top_mag_m"])   # the magnitude the spectra row draws
Vh = d["Vh"]                      # (n_freq, n_freq) foreground spectral modes
labels = [str(s) for s in d["labels"]]
n_f = freqs.size
N_SHOW = 18                       # foreground modes filtered (x-axis)
N_ANCHOR = int(d["n_anchor"])     # signal_loss.pdf's operating point
i_top = int(np.argmin(np.abs(mags - top_mag)))
dT = dT_disp[:, i_top]            # (3, n_lst, n_freq), the spectra row
print(dT_disp.shape, "at magnitudes", mags, "m; spectra row =", top_mag, "m")
print("LSTs", np.round(lst, 1))"""

PLOT_SRC = '''CMAP, norm = "twilight", Normalize(0, 24)
C_21 = "0.40"                     # 21 cm band: grey and dashed, so it reads as
                                  # a benchmark in both rows without competing
                                  # with either row's colour encoding.

# The two rows encode different variables on purpose. Above, LST: the spectra
# fan out by time of day and the cyclic map is the honest one for a quantity
# that wraps. Below, displacement: after filtering, the 24 LSTs collapse into a
# single bundle, so an LST scale there would spend 24 colours discriminating
# nothing, whereas the magnitudes separate by decades. Blues is single-hue and
# sequential because displacement is ordered -- light to dark is more, and it
# needs no legend lookup to read the ordering.
D_COL = ["#6baed6", "#2171b5", "#08306b"]
cmap = plt.get_cmap(CMAP)
n_modes = np.arange(N_SHOW + 1)


def resid_curves(dT_axis):
    """Per-LST residual RMS over freq [K] after filtering the leading N modes."""
    coeff = dT_axis @ Vh.T                                  # (n_lst, n_freq)
    return np.array([np.sqrt(np.sum(coeff[:, N:] ** 2, axis=1) / n_f)
                     for N in n_modes])                     # (N_SHOW+1, n_lst)


fig, axes = plt.subplots(
    2, 3, figsize=(7.3, 3.6),
    gridspec_kw=dict(height_ratios=[1.6, 1.15]),
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
    # Headroom for the magnitude tag. The three panels span very different
    # ranges and the curves reach the top of a different one in each, so the
    # tag needs space made for it rather than a corner that happens to be free.
    lo, hi = dT[col].min(), dT[col].max()
    at.set_ylim(lo - 0.05 * (hi - lo), hi + 0.30 * (hi - lo))
    at.text(0.97, 0.94, f"$+{top_mag:g}$ m", transform=at.transAxes,
            fontsize=6.5, color="0.35", ha="right", va="top")

    for k in range(mags.size):                              # bottom: all magnitudes
        rc = resid_curves(dT_disp[col, k])                  # (N_SHOW+1, n_lst)
        for j in range(lst.size):
            ab.plot(n_modes, rc[:, j], color=D_COL[k], lw=0.55, alpha=0.6)
    ab.fill_between(n_modes, t21[0], t21[2], color=C_21, alpha=0.25, lw=0, zorder=0)
    ab.plot(n_modes, t21[1], color=C_21, lw=1.4, ls="--", zorder=1)
    ab.set_yscale("log")
    ab.set_xlabel("Foreground modes filtered", fontsize=8)
    ab.grid(True, which="both", ls=":", lw=0.5, alpha=0.55)
    ab.set_xlim(0, N_SHOW); ab.set_ylim(3e-5, 30); ab.tick_params(labelsize=7)

axes[0, 0].set_ylabel(r"$\\Delta T_\\mathrm{ant}$ [K]", fontsize=8)
axes[1, 0].set_ylabel("Residual RMS [K]", fontsize=8)
# No marker for Fig. 1's operating point. A dotted line with an "N = 9" tag was
# tried and dropped: the number is meaningless without the other figure, and
# unlabelled it was just an unexplained rule. N_ANCHOR still governs the
# summary below and the paper text, which is where the comparison belongs.
handles = [Line2D([], [], color=D_COL[k], lw=2, label=f"{m:g} m")
           for k, m in enumerate(mags)]
handles.append(Line2D([], [], color=C_21, lw=1.4, ls="--", label="21 cm models"))
# Upper right, not lower left: every curve descends with N, so the top-right
# corner is the one reliably empty region in all three residual panels, while
# the lower left still carries the 0.1 m tails.
axes[1, 0].legend(handles=handles, fontsize=5.8, loc="upper right", ncol=2,
                  framealpha=0.9, handlelength=1.4, columnspacing=0.9,
                  borderpad=0.3, labelspacing=0.25)
for col in (1, 2):
    axes[1, col].tick_params(labelleft=False)

sm = ScalarMappable(norm=norm, cmap=CMAP)
cb = fig.colorbar(sm, ax=axes[0, :], pad=0.012, fraction=0.03)
cb.set_label("LST [h]", fontsize=8); cb.set_ticks(np.arange(0, 25, 6))
cb.ax.tick_params(labelsize=7)
fig.savefig("horizon_shift.pdf", bbox_inches="tight", dpi=600)'''

SUMMARY_SRC = """def stays_below(curve, ref):
    \"\"\"Smallest N with curve < ref there and at every larger N on the axis.

    Not first-crossing: both fall with N and cross more than once, so a
    first-crossing rule reports an N the curve later climbs back above.
    \"\"\"
    below = curve < ref
    return next(N for N in n_modes if below[N:].all())


worst = np.array([[resid_curves(dT_disp[c, k]).max(axis=1)
                   for k in range(mags.size)]
                  for c in range(len(labels))])    # (axis, mag, N_SHOW+1)

med21 = t21[1, N_ANCHOR] * 1e3
print(f"worst-LST residual, median retained 21 cm at N={N_ANCHOR} is {med21:.2f} mK\\n")
print(f"{'axis':7s}{'shift':>8s}{'unfiltered':>12s}{'at N=%d' % N_ANCHOR:>10s}"
      f"{'stays below from':>18s}")
for c, lab in enumerate(labels):
    for k, m in enumerate(mags):
        w = worst[c, k]
        print(f"{lab if k == 0 else '':7s}{m:7g}m{w[0]*1e3:11.1f} mK"
              f"{w[N_ANCHOR]*1e3:9.2f} mK{stays_below(w, t21[1]):15d} modes")

i_top = int(np.argmin(np.abs(mags - top_mag)))
worst_all = worst[:, i_top].max(axis=0)
n_sys = stays_below(worst_all, t21[1])
print(f"\\nFig. 1 sets N = {N_ANCHOR} on the foreground residual alone. Folding in "
      f"the +{top_mag:g} m position systematic costs {n_sys - N_ANCHOR} further "
      f"mode(s): worst axis/LST {worst_all[N_ANCHOR]*1e3:.2f} mK at N = {N_ANCHOR} "
      f"(median retained {med21:.2f} mK), {worst_all[n_sys]*1e3:.2f} mK at "
      f"N = {n_sys} (median retained {t21[1, n_sys]*1e3:.2f} mK).")

# Does the residual scale with displacement? Only on the vertical axis, where
# the horizon drops by a near-uniform offset. The horizontal shifts move the
# horizon by an amount set by where the cliff edges fall in azimuth, so they do
# not scale -- which is why the requirement below is a vertical one.
print()
for c, lab in enumerate(labels):
    r = worst[c, :, N_ANCHOR]
    dev = np.abs((r[1:] / r[:-1]) / (mags[1:] / mags[:-1]) - 1) * 100
    print(f"{lab:7s} deviation from proportionality per decade: "
          + ", ".join(f"{x:.1f}%" for x in dev))
i_up = labels.index("Up")
spec = 0.1 * med21 / (worst[i_up, i_top, N_ANCHOR] * 1e3) * top_mag
print(f"\\nUp is linear, and is the binding axis at every N. Holding its injection "
      f"to a tenth of the median retained signal needs the vertical position "
      f"known to {spec:.2f} m.")

# The two halves of the message, as numbers. Reassuring: nearly all of the
# displacement is in the leading modes, so it is more foreground rather than a
# new kind of structure. Cautionary: what escapes the filter is not a smooth
# tail but one mode, at the signal's amplitude -- which is why the residual of
# a fixed-depth filter cannot be read as cosmology.
cu = dT[2] @ Vh.T                                   # Up, every LST
j = int(np.argmax(np.sqrt(np.sum(cu[:, N_ANCHOR:]**2, axis=1))))
mode_mK = np.abs(cu[j]) / np.sqrt(n_f) * 1e3
lead = np.sum(cu[j, :2]**2) / np.sum(cu[j]**2)
spike = int(np.argmax(mode_mK[N_ANCHOR:])) + N_ANCHOR
tail = np.sum(mode_mK[N_ANCHOR:]**2)
print(f"\\nUp +{top_mag:g} m at LST {lst[j]:.0f} h: {lead*100:.1f}% of its power "
      f"sits in the two leading foreground modes -- mostly just more foreground. But "
      f"after filtering {N_ANCHOR} modes, {mode_mK[spike]**2/tail*100:.0f}% of what "
      f"remains is mode {spike+1} alone, at {mode_mK[spike]:.2f} mK against a "
      f"{t21[1, N_ANCHOR]*1e3:.2f} mK median retained signal. A fixed-depth "
      f"filter would leave that in the residual, looking like signal.")"""


def build_notebook():
    md = (
        "# Horizon-shift signal vs foreground filtering\n\n"
        "Antenna-position error changes the antenna temperature by "
        "$\\Delta T_\\mathrm{ant}(\\nu)$. Here we ask how *foreground-like* "
        "that change is: we project the $\\Delta T_\\mathrm{ant}$ spectra (top "
        "row, +1 m East/North/Up at 24 LSTs) onto the foreground spectral "
        "modes -- the same SVD modes as `foreground_svd.npz` -- and plot the "
        "residual RMS after filtering the leading $N$ modes (bottom row). The "
        "systematic sits in the same low-order foreground subspace: every LST "
        "and axis is driven down by the same low-order filtering that cleans "
        "the sky.\n\n"
        "The grey dashed band is the retained 21 cm signal (5-95% of the model "
        "ensemble, median dashed) under the *identical* projection -- the "
        "physical benchmark this residual has to beat, in place of an "
        "arbitrary 10 mK line.\n\n"
        "**The bottom row makes two points that pull opposite ways.**\n\n"
        "*Reassuring.* A position error is not a new kind of spectral "
        "structure. The $\\Delta T$ are large -- up to 8.9 K at 50 MHz, "
        "~1 K RMS for the upward shift -- but 99.9% of that power lies in the "
        "two leading eigenmodes of the unperturbed antenna temperature. It is, "
        "to that accuracy, more foreground, and the same low-order filtering "
        "that removes the sky removes almost all of it: East and North stay "
        "below the median retained 21 cm signal from 3 and 2 modes on.\n\n"
        "*Cautionary.* What survives is narrow, not smooth. `signal_loss.ipynb` "
        "(Fig. 1) sets its operating point $N$ on the foreground residual "
        "alone, and the dotted vertical line marks it here. Filter that many "
        "modes and 99% of the Up displacement's remaining power sits in mode "
        "10 alone, at 3.18 mK -- more than that mode holds from the nominal "
        "foregrounds (1.71 mK) or from the median 21 cm model (1.66 mK), and "
        "close enough in shape to be confused with the retained signal (cosine "
        "similarity up to 0.99). 73% of the ensemble carries less signal in "
        "that mode than a 1 m shift would put there. The *vertical* response is "
        "linear in displacement (~2% per decade over $\\pm$0.1 to $\\pm$10 m) "
        "and symmetric in sign, because a vertical shift lowers the horizon by "
        "a near-uniform offset; East and North are neither, since a horizontal "
        "shift moves the horizon by an amount set by where the cliff edges "
        "fall in azimuth. Up is the binding axis at every $N$, so the "
        "extrapolation is sound and the requirement is a vertical one: about "
        "0.1 m, to hold the injection to a tenth of the median retained "
        "signal.\n\n"
        "**Neither this figure nor `signal_loss.ipynb` is a proposed "
        "analysis.** Both project onto eigenmodes of a *simulated* nominal "
        "instrument. An unmodelled displacement of the size we must anticipate "
        "deposits signal-like power in the first mode past the filter, so a "
        "residual left by a filter of fixed depth is not evidence of a "
        "cosmological signal -- excess with respect to a foreground model is "
        "only as trustworthy as the instrument model behind it. These figures "
        "characterise spectral structure; the analysis marginalises over "
        "antenna position inside a differentiable forward model instead, and "
        "the sensitivities here are what set its priors.\n\n"
        "Crossings below use a *stays below* rule: the smallest $N$ at which "
        "the worst-LST residual is under the median retained signal and "
        "remains so for every larger $N$. Both curves fall with $N$ and cross "
        "more than once -- Up is below the median at $N = 7$ and $N = 8$ and "
        "back above it at $N = 9$ -- so a first-crossing rule would report a "
        "different, over-optimistic answer from the same arrays.\n\n"
        "See `signal_loss.ipynb` for the full signal-loss calculation and its "
        "limitations. Both figures describe a blind eigenmode projection, "
        "which is the most conservative filter available and not the analysis "
        "EIGSEP plans to run; they bound spectral subspace overlap, not "
        "sensitivity."
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
