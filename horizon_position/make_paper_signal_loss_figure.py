"""Build the paper 21 cm signal-loss figure.

Referee response: the eigenmode analysis shows only that the *simulated
foregrounds* occupy a low-dimensional spectral subspace; it says nothing
about whether the cosmological signal survives the same filter. This
figure answers that by pushing an ensemble of global-signal models
through the *identical* projection used for the foregrounds and the
position-error systematic, so the residual and the retained signal are
always read off the same axes.

Writes three artifacts into the eigsep_instrument paper notebooks dir
(repo convention: a gitignored npz + a committed notebook that
regenerates the PDF, data archived to Zenodo separately):

* ``signal_loss.npz``   -- foreground spectral modes ``Vh`` and singular
  values, the +1 m position-error spectra, and the 21 cm model ensemble
  interpolated onto the paper frequency grid.
* ``signal_loss.ipynb`` -- loads the npz and makes the figure.
* ``signal_loss.pdf``   -- the rendered figure.

Panels (a1)/(a2): every model before and after filtering ``N_ANCHOR``
modes, coloured by the RMS it retains -- the same quantity, scale and
colormap as panel (b), so a curve's colour and its height there agree.
What survives is band-edge ringing from projecting onto a truncated
smooth basis, not a residual trough; retained RMS is not retained signal
*shape*, which is why the panel is here at all. Panel (b): the same
retained-signal curves against the number of modes filtered, with the
foreground residual and the worst-case position-error systematic in
greyscale. Colour marks the subject, greyscale the floors it is measured
against. This panel supersedes ``foreground_svd_residual.pdf``: the
foreground curve is the same one, now never shown without the signal
beside it.

Colouring is continuous rather than binned into classes on purpose.
Retention varies smoothly and no single statistic predicts it: trough
width tracks the retained *fraction* (Spearman -0.44) while the absolute
retained RMS is driven mainly by amplitude (+0.76), so any class edge
would cut a continuum and invite reading the bins as populations. The
norm is logarithmic -- 3.4 decades of range, where a linear norm would
put 47% of models in the bottom 10% of the colour range.

``N_ANCHOR = 10`` is the smallest N at which *both* floors -- the
foreground residual and the worst-case position systematic -- fall below
the median retained signal. At N = 8 (the previous operating point) the
foreground residual and the median signal are the same size, which is
exactly the referee's objection. Note there is no optimum to claim
beyond this: the margin keeps improving with N while the absolute signal
shrinks, and where to stop is set by thermal noise, which this
calculation does not model.

Run in the mock_analysis env (numpy + matplotlib + nbformat):
    uv run python horizon_position/make_paper_signal_loss_figure.py
"""

import os
from pathlib import Path

import nbformat as nbf
import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = Path(
    os.environ.get(
        "EIGSEP_PAPER_NOTEBOOKS",
        "/home/christian/Documents/research/papers/eigsep_instrument/notebooks",
    )
)
FG_NPZ = PAPER / "foreground_svd.npz"
SHIFT_NPZ = PAPER / "horizon_shift.npz"

# TODO(provenance): this ensemble predates the paper and no generating
# script survives alongside it. Confirm its origin (21cmGEM? the 21cmVAE
# training set?) before writing the caption citation, or regenerate from
# 21cmVAE/VeryAccurateEmulator/dataset_21cmVAE.h5. The npz stores the
# models as a plain array so swapping the ensemble touches only this
# constant and the loader cell.
MODELS_NPZ = Path(
    "/home/christian/Documents/research/eigsep/normalizing_flows/models_21cm.npz"
)

N_ANCHOR = 10  # modes filtered at the quoted operating point
# Class edges on retained RMS [mK] at N_ANCHOR. 1 and 5 mK split the
# ensemble roughly into thirds; 10 mK would catch 2.6% and 25 mK nothing
# at all (the most foreground-orthogonal model retains 16.4 mK).
RET_EDGES_MK = (1.0, 5.0)
CLASS_LABELS = ("< 1 mK", "1-5 mK", "> 5 mK")

# Distinguishability statistics (the paragraph on what the filter leaves
# *measurable*, as opposed to how much amplitude it leaves). Models shallower
# than DEEP_MIN_MK are not a target for any global-signal experiment and would
# otherwise dominate the pair statistics; the window is where the ensemble
# actually puts its absorption trough, so the result cannot be band-edge
# ringing from the projection.
DEEP_MIN_MK = 50.0
PAIR_WINDOW_MHZ = (70.0, 130.0)
PAIR_THRESH_MK = (1.0, 2.0, 5.0)

# From horizon_position/rotation_dimensionality.py, which pools the 36 x 36
# drive grid of horizon_chromaticity/output/chromaticity_eigsep.npz. Recorded
# here rather than recomputed because that needs the 3 GB simulation cube; the
# script cross-checks its zenith curve against this figure's numbers before
# reporting, so the two stay tied together.
ROT_N_ORIENTATIONS = 1296
ROT_N_ELEV = 36
ROT_N_AZ = 36
ROT_N_MODES_FULL_GRID = 16


def load_t21(freqs):
    """The 21 cm model ensemble interpolated onto `freqs` [K].

    models_21cm.npz holds (n_model, 200) in mK on a GHz-valued 50-249 MHz
    grid; the paper grid runs to 250 MHz, where the signal is ~0 (z < 4.7).
    """
    m = np.load(MODELS_NPZ)
    mod_f, models = m["freqs"] * 1e3, m["models"] * 1e-3
    return np.array([np.interp(freqs, mod_f, mm) for mm in models])


def retained_pct(T21, Vh, n_modes, pct=(5, 50, 95)):
    """Percentiles of the retained-signal RMS [K] vs modes filtered."""
    c = T21 @ Vh.T
    n_f = T21.shape[1]
    rms = np.array([np.sqrt(np.sum(c[:, N:] ** 2, axis=1) / n_f) for N in n_modes])
    return np.percentile(rms, pct, axis=1)


def classify(T21, Vh, n_anchor=N_ANCHOR):
    """Class index per model (0/1/2) from its retained RMS at `n_anchor`."""
    c = T21 @ Vh.T
    ret_mK = np.sqrt(np.sum(c[:, n_anchor:] ** 2, axis=1) / T21.shape[1]) * 1e3
    return np.digitize(ret_mK, RET_EDGES_MK), ret_mK


def build_data():
    fg = np.load(FG_NPZ)
    t_ant = fg["t_sys"] - fg["t_receiver"]  # antenna temperature, receiver dropped
    n_time = t_ant.shape[0]
    freqs = fg["freqs_MHz"]

    shift = np.load(SHIFT_NPZ, allow_pickle=True)
    assert np.array_equal(shift["freqs_MHz"], freqs), "frequency grid mismatch"
    Vh = shift["Vh"]  # same modes as foreground_svd.npz / Fig. 1
    s_fg = np.linalg.svd(t_ant, compute_uv=False)

    T21 = load_t21(freqs)  # (n_model, n_f) K
    cls, _ = classify(T21, Vh)

    np.savez_compressed(
        PAPER / "signal_loss.npz",
        freqs_MHz=freqs,
        Vh=Vh,
        s_fg=s_fg,
        n_time=n_time,
        dT_spectra=shift["dT_spectra"],
        labels=shift["labels"],
        T21_models=T21,
        cls=cls,
        class_labels=np.array(CLASS_LABELS),
        n_anchor=N_ANCHOR,
        description=(
            "Inputs for the 21 cm signal-loss figure. Vh (n_freq, n_freq) and "
            "s_fg (n_freq,) are the right singular vectors and singular values "
            "of the nominal antenna-temperature waterfall (foreground_svd.npz "
            "t_sys minus the constant receiver temperature, n_time LST samples) "
            "-- the same modes as Fig. 1. dT_spectra (3, n_lst, n_freq) [K] are "
            "the +1 m East/North/Up antenna-temperature differences. T21_models "
            "(n_model, n_freq) [K] is an ensemble of global 21 cm signal models "
            "interpolated onto freqs_MHz; cls bins each model by the RMS it "
            "retains at N_ANCHOR, for the summary only. All three quantities "
            "are filtered by the same "
            "projection onto the leading modes of Vh."
        ),
    )
    print(f"wrote {PAPER / 'signal_loss.npz'}  ({T21.shape[0]} models)")


# --- source shared by the notebook and the direct render (kept in sync) ---

IMPORTS_SRC = """import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm"""

LOAD_SRC = """d = np.load("signal_loss.npz", allow_pickle=True)
freqs = d["freqs_MHz"]           # (n_f,) MHz
Vh = d["Vh"]                     # (n_f, n_f) foreground spectral modes
s_fg = d["s_fg"]                 # (n_f,) singular values of the T_ant waterfall
n_time = int(d["n_time"])        # LST samples in that waterfall
dT = d["dT_spectra"]             # (3, n_lst, n_f) +1 m E/N/U systematic [K]
T21 = d["T21_models"]            # (n_model, n_f) global-signal ensemble [K]
cls = d["cls"]                   # (n_model,) retained-RMS bin, for the summary
class_labels = [str(x) for x in d["class_labels"]]
n_f = freqs.size
N_SHOW = 18                      # x-axis extent
N_ANCHOR = int(d["n_anchor"])    # modes filtered at the quoted operating point
CURVE_ALPHA = 0.10               # opacity of the curves in panel (b)
ALL_ALPHA = 0.08                 # opacity of the full ensemble in panel (a)
CONT_CMAP = "plasma"             # continuous variant: colour = retained RMS
print(f"{T21.shape[0]} 21 cm models on {n_f} channels, "
      f"{freqs[0]:.0f}-{freqs[-1]:.0f} MHz")"""

CALC_SRC = '''n_modes = np.arange(N_SHOW + 1)

# Foreground residual after filtering the leading N modes [K] -- the curve
# that used to be foreground_svd_residual.pdf.
tail = np.concatenate([np.cumsum(s_fg[::-1] ** 2)[::-1], [0.0]])
fg_resid = np.sqrt(tail / (n_time * n_f))[: N_SHOW + 1]


def filt_rms(x):
    """RMS over frequency after filtering the leading N modes, per row."""
    c = np.atleast_2d(x) @ Vh.T
    return np.array([np.sqrt(np.sum(c[:, N:] ** 2, axis=1) / n_f)
                     for N in n_modes])                    # (N_SHOW+1, n_row)


def filtered(x, N):
    """The part of x left after projecting out the leading N modes."""
    c = np.atleast_2d(x) @ Vh.T
    return (c[:, N:] @ Vh[N:]).reshape(np.shape(x))


sys_resid = filt_rms(dT.reshape(-1, n_f)).max(axis=1)      # worst axis/LST
t21_resid = filt_rms(T21)                                  # (N_SHOW+1, n_model)
t21_pct = np.percentile(t21_resid, [5, 50, 95], axis=1)    # (3, N_SHOW+1)
t21_filt = filtered(T21, N_ANCHOR)                         # (n_model, n_f) residuals'''

PLOT_SRC = r'''C_FG, C_SYS = "k", "0.45"


def make_figure(path):
    """Colour every model by what it retains, with no class boundaries.

    The three-class version cuts a continuum into bins, which invites
    reading the bins as populations; they are not. Here the colour is the
    retained RMS itself, on a log scale, and the colorbar replaces the
    class legend. Exemplars are dropped -- nothing distinguishes those
    three models once colour carries the quantity directly.
    """
    ret = t21_resid[N_ANCHOR] * 1e3                         # colour quantity [mK]
    norm = LogNorm(vmin=ret[ret > 0].min(), vmax=ret.max())
    order = np.argsort(-ret)                                # faintest drawn last

    def segs(x, Y):
        return np.stack([np.broadcast_to(np.asarray(x), Y.shape), Y], axis=-1)

    fig, ax = plt.subplot_mosaic(
        [["a1", "b"], ["a2", "b"]],
        figsize=(7.6, 3.2), layout="constrained",
        gridspec_kw=dict(width_ratios=[1, 1.2]),
    )
    for key, Y in (("a1", T21[order] * 1e3), ("a2", t21_filt[order] * 1e3)):
        lc = LineCollection(segs(freqs, Y), cmap=CONT_CMAP, norm=norm,
                            lw=0.4, alpha=ALL_ALPHA)
        lc.set_array(ret[order])
        ax[key].add_collection(lc)
        ax[key].set_xlim(freqs[0], freqs[-1])
        ax[key].set_ylim(Y.min() * 1.05, max(Y.max() * 1.05, 0.02 * abs(Y.min())))

    b = ax["b"]
    lc = LineCollection(segs(n_modes, t21_resid[:, order].T), cmap=CONT_CMAP,
                        norm=norm, lw=0.5, alpha=CURVE_ALPHA)
    lc.set_array(ret[order])
    b.add_collection(lc)
    ref = [b.plot(n_modes, fg_resid, color=C_FG, lw=1.5,
                  label="foreground residual")[0],
           b.plot(n_modes, sys_resid, color=C_SYS, lw=1.4, ls="--",
                  label="+1 m position error (worst LST)")[0]]

    for key, lab, ylab in (("a1", "input", r"$T_{21}$ [mK]"),
                           ("a2", f"after filtering {N_ANCHOR} modes",
                            "Residual [mK]")):
        ax[key].axhline(0, color="0.6", lw=0.6, ls="--", zorder=0)
        ax[key].set_ylabel(ylab, fontsize=8)
        ax[key].grid(alpha=0.2)
        ax[key].tick_params(labelsize=7)
        ax[key].text(0.03, 0.06, lab, transform=ax[key].transAxes, fontsize=7,
                     ha="left", va="bottom")
    ax["a1"].tick_params(labelbottom=False)
    ax["a2"].set_xlabel("Frequency [MHz]", fontsize=8)

    b.axvline(N_ANCHOR, color="0.6", lw=0.8, ls=":", zorder=0)
    b.text(N_ANCHOR - 0.3, 1e0, f"$N = {N_ANCHOR}$", fontsize=7, color="0.35",
           ha="right", va="center")
    b.set_yscale("log")
    b.set_xlim(0, N_SHOW)
    b.set_ylim(1e-5, 3e3)
    b.set_xlabel("Foreground modes filtered", fontsize=8)
    b.set_ylabel("RMS over band [K]", fontsize=8)
    b.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    b.tick_params(labelsize=7)
    b.legend(handles=ref, fontsize=6.5, loc="lower left", framealpha=0.92)

    sm = ScalarMappable(norm=norm, cmap=CONT_CMAP)
    cb = fig.colorbar(sm, ax=b, pad=0.015, fraction=0.045)
    cb.set_label(f"21 cm RMS retained at $N = {N_ANCHOR}$ [mK]", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    cb.solids.set_alpha(1.0)

    fig.savefig(path, bbox_inches="tight", dpi=600)


make_figure("signal_loss.pdf")'''

SUMMARY_SRC = """frac_above = (t21_resid > fg_resid[:, None]).mean(axis=1)
print(f"{'N':>3} {'fgnd':>9} {'pos err':>9} {'21cm p50':>9} {'21cm p95':>9} "
      f"{'frac>fgnd':>10}   (mK)")
for N in (6, 8, N_ANCHOR, 12, 15):
    print(f"{N:3d} {fg_resid[N]*1e3:9.3f} {sys_resid[N]*1e3:9.3f} "
          f"{t21_pct[1, N]*1e3:9.3f} {t21_pct[2, N]*1e3:9.3f} "
          f"{frac_above[N]:10.2f}")

keep = t21_resid[N_ANCHOR] / t21_resid[0]
print(f"\\nAt N = {N_ANCHOR}: median model keeps {np.median(keep)*100:.0f}% of its "
      f"RMS ({t21_pct[1, N_ANCHOR]*1e3:.2f} mK), while the foreground residual is "
      f"{fg_resid[N_ANCHOR]*1e3:.2f} mK and the worst-case +1 m position error is "
      f"{sys_resid[N_ANCHOR]*1e3:.2f} mK.")
print(f"{frac_above[N_ANCHOR]*100:.0f}% of the {t21_resid.shape[1]} models retain "
      f"more signal than the foreground residual.")

# What separates the classes: at matched depth it is trough width, not amplitude.
width = (T21 < T21.min(axis=1, keepdims=True) / 2).sum(axis=1) * (freqs[1] - freqs[0])
depth = -T21.min(axis=1) * 1e3
window = (depth > 80) & (depth < 160)
print()
for k, lab in enumerate(class_labels):
    m, mw = cls == k, (cls == k) & window
    print(f"{lab:>6s} retained: {m.sum():4d} models, median depth "
          f"{np.median(depth[m]):6.1f} mK; at matched depth (80-160 mK) "
          f"n={mw.sum():3d}, median trough width {np.median(width[mw]):3.0f} MHz")"""


def build_notebook():
    md = (
        "# 21 cm signal loss under foreground-mode filtering\n\n"
        "The eigenmode analysis shows that the simulated beam-weighted "
        "foregrounds occupy a low-dimensional spectral subspace. On its own "
        "that says nothing about whether the cosmological signal survives the "
        "same filter, so here we push an ensemble of global 21 cm models "
        "through the *identical* projection onto the leading $N$ foreground "
        "modes and read the retained signal off the same axes as the "
        "residuals.\n\n"
        "Panels (a1)/(a2) show all the models before and after filtering $N$ "
        "modes, each coloured by the RMS it retains at $N$ -- the same "
        "quantity plotted in panel (b), on the same log scale, so a curve's "
        "colour and its height in (b) agree. What survives is small but still "
        "structured; note it is band-edge ringing from projecting onto a "
        "truncated smooth basis, not a residual trough, so retained RMS is "
        "not retained signal *shape*. Panel (b) adds the foreground residual "
        "and the worst-case $+1$ m position-error systematic in greyscale: "
        "colour marks the subject, greyscale the floors it is measured "
        "against. It supersedes `foreground_svd_residual.pdf` -- the black "
        "curve is the same one, now never shown without the signal beside "
        "it.\n\n"
        "**Retention is a continuum and no single summary statistic predicts "
        "it**, which is why the models are coloured continuously rather than "
        "binned into classes. It is set by how much of a model's spectral "
        "shape lies in the leading foreground modes. Separating shape from "
        "amplitude over the ensemble (Spearman): trough width correlates with "
        "the retained *fraction* at -0.44 -- narrower troughs keep "
        "proportionally more -- while the *absolute* retained RMS is driven "
        "mainly by amplitude (depth, +0.76). Width is a trend, not a "
        "predictor: models retaining more than 5 mK and those retaining "
        "1-5 mK have similar depths (median 147 vs 134 mK) and heavily "
        "overlapping widths (p25-p75 of 15-34 vs 16-49 MHz). The dependence "
        "is also non-monotonic -- median retained fraction peaks near "
        "15-25 MHz and falls off on both sides, the narrowest bin being full "
        "of edge-of-band troughs pinned at 50 MHz. The summary cell bins the "
        "distribution at 1 and 5 mK for the caption; those bins are a "
        "reporting convenience, not populations.\n\n"
        "The colour scale is logarithmic. Retained RMS spans 3.4 decades "
        "(0.007-16.4 mK), and a linear norm would put 47% of the models in "
        "the bottom 10% of the colour range, against 2.9% for log.\n\n"
        "$N = 10$ is the smallest $N$ at which *both* floors fall below the "
        "median retained signal. At $N = 8$ the foreground residual and the "
        "median signal are the same size.\n\n"
        "**Limitations, to be stated wherever this result is used.** The "
        "modes come from a single simulated sky (GSM16) and beam, with no "
        "noise and no receiver systematics; in practice the basis would be "
        "estimated from data that already contain the signal, which costs "
        "additional signal loss not captured here. Filtering is a hard "
        "projection, whereas a joint signal-plus-foreground fit would recover "
        "some of what is removed. Signal loss is severe in absolute terms, "
        "and whether the retained amplitude is detectable is set by thermal "
        "noise and integration time, which this calculation does not model. "
        "This is a statement about spectral subspace overlap, not a "
        "sensitivity forecast.\n\n"
        "**TODO:** confirm the provenance of the 21 cm model ensemble before "
        "writing the caption citation."
    )
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(md),
        nbf.v4.new_code_cell(IMPORTS_SRC),
        nbf.v4.new_code_cell(LOAD_SRC),
        nbf.v4.new_code_cell(CALC_SRC),
        nbf.v4.new_code_cell(PLOT_SRC),
        nbf.v4.new_code_cell(SUMMARY_SRC),
    ]
    nbf.write(nb, PAPER / "signal_loss.ipynb")
    print(f"wrote {PAPER / 'signal_loss.ipynb'}")


def render_pdf():
    """Run the notebook's source here so the PDF and notebook stay identical."""
    ns = {}
    cwd = os.getcwd()
    os.chdir(PAPER)
    try:
        for src in (IMPORTS_SRC, LOAD_SRC, CALC_SRC, PLOT_SRC, SUMMARY_SRC):
            exec(src, ns)
    finally:
        os.chdir(cwd)
    print(f"wrote {PAPER / 'signal_loss.pdf'}")


def tex_stats():
    """Every number the draft paper text quotes, from signal_loss.npz itself.

    Deriving these from the figure's own npz rather than recomputing means the
    prose and the figure can never disagree, and re-running this script after
    the 21 cm ensemble is regenerated updates both together.
    """
    d = np.load(PAPER / "signal_loss.npz", allow_pickle=True)
    freqs, Vh, s_fg = d["freqs_MHz"], d["Vh"], d["s_fg"]
    n_time, n_f = int(d["n_time"]), freqs.size
    dT, T21 = d["dT_spectra"], d["T21_models"]
    N = int(d["n_anchor"])

    tail = np.concatenate([np.cumsum(s_fg[::-1] ** 2)[::-1], [0.0]])
    fg = np.sqrt(tail / (n_time * n_f))

    def filt_rms(x, n):
        c = np.atleast_2d(x) @ Vh.T
        return np.sqrt(np.sum(c[:, n:] ** 2, axis=1) / n_f)

    sys_at = {n: filt_rms(dT.reshape(-1, n_f), n).max() for n in (N - 1, N)}
    ret = filt_rms(T21, N)
    depth_mK = -T21.min(axis=1) * 1e3

    # Pairwise separation of the deep models inside the trough window: what the
    # filter leaves distinguishable, rather than how much amplitude it leaves.
    deep = depth_mK > DEEP_MIN_MK
    S = T21[deep]
    R = S - (S @ Vh[:N].T) @ Vh[:N]
    win = (freqs >= PAIR_WINDOW_MHZ[0]) & (freqs <= PAIR_WINDOW_MHZ[1])

    def pair_rms(A):
        g = A[:, win] @ A[:, win].T
        d2 = np.maximum(np.diag(g)[:, None] + np.diag(g)[None, :] - 2 * g, 0.0)
        return np.sqrt(d2[np.triu_indices(A.shape[0], 1)] / win.sum()) * 1e3

    before, after = pair_rms(S), pair_rms(R)

    return {
        "N": N,
        "NM1": N - 1,
        "N_MODEL": f"{T21.shape[0]}",
        "FG_ANCHOR": f"{fg[N] * 1e3:.2f}",
        "FG_NM1": f"{fg[N - 1] * 1e3:.1f}",
        "FG_NM2": f"{fg[N - 2] * 1e3:.1f}",
        "SYS_ANCHOR": f"{sys_at[N] * 1e3:.2f}",
        "SYS_NM1": f"{sys_at[N - 1] * 1e3:.1f}",
        "RET_P50": f"{np.median(ret) * 1e3:.1f}",
        "RET_P50_NM1": f"{np.median(filt_rms(T21, N - 1)) * 1e3:.1f}",
        "RET_PCT": f"{np.median(ret / filt_rms(T21, 0)) * 100:.0f}",
        "FRAC_ABOVE_FG": f"{(ret > fg[N]).mean() * 100:.0f}",
        "RET_150": f"{np.median(ret * 1e3 / depth_mK) * 150:.0f}",
        "DEEP_MIN": f"{DEEP_MIN_MK:.0f}",
        "N_DEEP": f"{deep.sum()}",
        "WIN_LO": f"{PAIR_WINDOW_MHZ[0]:.0f}",
        "WIN_HI": f"{PAIR_WINDOW_MHZ[1]:.0f}",
        "PAIR_BEFORE": f"{np.median(before):.0f}",
        "PAIR_AFTER": f"{np.median(after):.0f}",
        "PAIR_GT1": f"{(after > PAIR_THRESH_MK[0]).mean() * 100:.0f}",
        "PAIR_GT2": f"{(after > PAIR_THRESH_MK[1]).mean() * 100:.0f}",
        "ROT_N_ORI": f"{ROT_N_ORIENTATIONS}",
        "ROT_N_ELEV": f"{ROT_N_ELEV}",
        "ROT_N_AZ": f"{ROT_N_AZ}",
        "ROT_N_MODES": f"{ROT_N_MODES_FULL_GRID}",
    }


TEX_TEMPLATE = r"""% signal_loss_text.tex -- GENERATED, do not edit by hand.
%
% Draft replacement text for the 21 cm signal-loss result. Regenerate with
%     uv run python horizon_position/make_paper_signal_loss_figure.py
% in the mock_analysis repo; every number below is computed from
% signal_loss.npz, so re-running after the 21 cm ensemble is regenerated
% updates the prose and the figure together.
%
% Paste the three blocks into rasti_template.tex as marked. Nothing here is
% \input by the paper -- this file is a staging area, not a dependency.
%
% Numbers that depend on the 21 cm ensemble and will move when it is
% regenerated: the model count, retained amplitudes and percentages, and every
% distinguishability statistic. Numbers that will not move: the foreground
% residual, the position systematic, the choice of N, and the mode counts in
% block 3.


% ===================================================================
% BLOCK 1 -- section "Minimising Covariance with the 21-cm Signal".
% Replaces the sentence beginning "We emphasise that this analysis only
% quantifies the spectral complexity ...".
% ===================================================================

This analysis quantifies the spectral complexity of the beam-weighted
foregrounds; on its own it says nothing about whether the cosmological signal
survives the same filter. We therefore passed an ensemble of @N_MODEL@ global
21-cm models through the identical projection, so that the retained signal, the
foreground residual, and the antenna-position systematic of
section~\ref{subsec:fwd_modelling} are read off the same axes
(Fig.~\ref{fig:singular_values}). We adopt $N=@N@$ filtered modes as a
reference operating point: it is the smallest $N$ at which both the foreground
residual ($@FG_ANCHOR@$\,mK) and the worst-case systematic from a 1\,m antenna
displacement ($@SYS_ANCHOR@$\,mK) fall below the median retained signal. At
$N=@NM1@$ the position systematic is $@SYS_NM1@$\,mK against a median retained
signal of $@RET_P50_NM1@$\,mK, and at $N=8$ the foreground residual alone
exceeds it. At the reference point the median model retains @RET_PCT@ per cent
of its band RMS, or $@RET_P50@$\,mK, and @FRAC_ABOVE_FG@ per cent of the
ensemble retains more than the foreground residual. Retention scales
approximately with signal amplitude: a model with a 150\,mK absorption trough
retains $\sim@RET_150@$\,mK.

Retained RMS understates what such a filter leaves measurable. The projection
discards the components of a signal that lie along the foreground modes and
keeps the orthogonal complement, so the relevant question is not how much
amplitude survives but which models remain distinguishable within the subspace
that does. Restricting to the @N_DEEP@ ensemble members with absorption troughs
deeper than @DEEP_MIN@\,mK and scoring over @WIN_LO@--@WIN_HI@\,MHz, the median
separation between a pair of models falls from @PAIR_BEFORE@\,mK to
@PAIR_AFTER@\,mK under the filter, yet @PAIR_GT1@ per cent of pairs remain
separated by more than 1\,mK and @PAIR_GT2@ per cent by more than 2\,mK. A
measurement in the filtered subspace therefore constrains the signal to a
family of models differing by components that lie along the foreground modes,
rather than collapsing the ensemble to an indistinguishable residual.

These figures are a floor rather than a forecast. The filter used here is
maximally agnostic: its modes are estimated from the antenna temperature
itself, and no knowledge of the beam or the sky enters. EDGES
\citep{2025PASP..137l5002C} and MIST \citep{2024MNRAS.530.4125M} instead divide
out a simulated beam chromaticity correction before fitting, and REACH
\citep{2022JAI....1150001C} marginalises over a parametrised beam within its
forward model. EIGSEP is designed to do likewise, using the beam measurements
of section~\ref{subsec:beam_mapping} and the forward model of
section~\ref{subsec:fwd_modelling}; every mode of chromaticity that is modelled
rather than filtered is one fewer mode removed from the signal. Filtering is
also a hard projection, whereas a joint fit for the signal and the foregrounds
recovers part of what a projection discards, and the rotational degrees of
freedom described below provide further leverage that a per-spectrum filter
cannot use. Finally, the calculation uses a single simulated sky and beam and
contains no noise, so it describes spectral subspace overlap and not
sensitivity; whether a retained amplitude is detectable is set by the thermal
noise and integration time, which we do not model here.
Fig.~\ref{fig:singular_values} should accordingly not be read as a requirement
that EIGSEP calibrate at the millikelvin level to detect a typical signal. It
is the residual left by the most conservative foreground filter available, and
it sets the scale of the improvement that beam knowledge and joint fitting are
required to deliver.


% ===================================================================
% BLOCK 2 -- caption for signal_loss.pdf, replacing the
% foreground_svd_residual.pdf caption. Keep \label{fig:singular_values}:
% the horizon_shift caption already refers to it.
% ===================================================================

\caption{Signal loss under foreground-mode filtering. An ensemble of @N_MODEL@
global 21-cm models is passed through the same projection onto the leading $N$
eigenmodes of the simulated antenna temperature that is applied to the
foregrounds. Panels (a1) and (a2) show every model before and after filtering
$N=@N@$ modes, each coloured by the band RMS it retains at that operating
point; the colour scale is logarithmic and shared with panel (b), so a curve's
colour and its height in (b) agree. Panel (b) shows retained RMS against the
number of modes filtered, with the foreground residual (black) and the
worst-case systematic from a 1\,m antenna displacement (grey dashed,
Fig.~\ref{fig:horizon_shift}) for reference; colour marks the signal, greyscale
the floors it is measured against. At $N=@N@$ the foreground residual is
$@FG_ANCHOR@$\,mK and the position systematic $@SYS_ANCHOR@$\,mK, while the
median model retains $@RET_P50@$\,mK, or @RET_PCT@ per cent of its band RMS,
and @FRAC_ABOVE_FG@ per cent of the ensemble retains more than the foreground
residual. The structure surviving in (a2) is ringing from projecting onto a
truncated smooth basis rather than a residual absorption trough, so retained
RMS should not be read as retained signal shape. This filter uses no knowledge
of the beam or the sky and is therefore the most conservative case; see the
text.}


% ===================================================================
% BLOCK 3 -- section "Minimising Covariance", rotation paragraph.
% Insert after "... we aim to use both to improve constraints."
% Numbers from horizon_position/rotation_dimensionality.py.
% ===================================================================

Rotation adds spectral diversity to the data, and it is worth asking what that
diversity costs in model complexity. Repeating the eigenmode analysis above
over the full drive grid of @ROT_N_ORI@ pointings (@ROT_N_ELEV@ elevations
$\times$ @ROT_N_AZ@ azimuths) rather than the zenith pointing alone, the pooled
antenna temperature is described to the same residual by @ROT_N_MODES@ modes
instead of @N@. The foregrounds seen across every accessible orientation
therefore occupy a subspace only marginally larger than that of a single
pointing. Realising the benefit of that diversity requires a joint fit in which
the sky and the beam are shared parameters and the rotations are known; a
per-spectrum projection of the kind used in Fig.~\ref{fig:singular_values}
cannot exploit it, because the information lies in the correlation between
pointings rather than within any one spectrum. We defer this analysis to future
work.
"""


def build_tex():
    """Write the draft paper text with every number filled in from the data."""
    out = TEX_TEMPLATE
    for key, val in tex_stats().items():
        out = out.replace(f"@{key}@", str(val))
    leftover = [tok for tok in out.split("@")[1::2] if tok.isupper()]
    assert not leftover, f"unfilled tokens in the template: {leftover}"
    (PAPER / "signal_loss_text.tex").write_text(out)
    print(f"wrote {PAPER / 'signal_loss_text.tex'}")


def main():
    for p in (FG_NPZ, SHIFT_NPZ, MODELS_NPZ):
        if not p.exists():
            raise SystemExit(f"{p} not found")
    build_data()
    build_notebook()
    render_pdf()
    build_tex()


if __name__ == "__main__":
    main()
