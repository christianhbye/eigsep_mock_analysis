"""Build the paper 21 cm signal-loss figure.

Referee response: the eigenmode analysis shows only that the *simulated
foregrounds* occupy a low-dimensional spectral subspace; it says nothing
about whether the cosmological signal survives the same filter. This
figure answers that by pushing an ensemble of global-signal models
through the *identical* projection used for the foregrounds and the
position-error systematic, so the residual and the retained signal are
always read off the same axes.

Writes four artifacts into the eigsep_instrument paper notebooks dir
(repo convention: a gitignored npz + a committed notebook that
regenerates the PDF, data archived to Zenodo separately):

* ``signal_loss.npz``   -- foreground spectral modes ``Vh`` and singular
  values, and the 21 cm model ensemble interpolated onto the paper
  frequency grid.
* ``signal_loss.ipynb`` -- loads the npz and makes the figure.
* ``signal_loss.pdf``   -- the rendered figure.
* ``signal_loss_text.tex`` -- draft prose and caption for the paper,
  with every quoted number computed at generation time from the arrays
  above. It is a staging file to paste from, never included by the
  paper; nothing in this repo writes to the paper's own .tex sources.

Panels (a1)/(a2): a legibility subsample of the surviving ensemble
before and after filtering ``N_ANCHOR`` modes (panel (b) draws every
model), coloured by the RMS it retains -- the same quantity, scale and
colormap as panel (b), so a curve's colour and its height there agree.
What survives is band-edge ringing from projecting onto a truncated
smooth basis, not a residual trough; retained RMS is not retained signal
*shape*, which is why the panel is here at all. Panel (b): the same
retained-signal curves against the number of modes filtered, with the
foreground residual in black. Colour marks the subject, greyscale the
one floor it is measured against here. This panel supersedes
``foreground_svd_residual.pdf``: the foreground curve is the same one,
now never shown without the signal beside it.

Scope, deliberately narrow. This is the paper's *first* figure and it
makes the smallest claim that supports the design argument: the
beam-weighted foregrounds are spectrally low-dimensional, and a signal
survives projecting that subspace out. Nothing else is folded in --
no instrumental systematics, no noise, and no knowledge of the beam or
the sky. The antenna-position systematic used to be drawn here as a
second floor; it was removed because it is not defined until the
forward-modelling section, and its own figure (``horizon_shift.pdf``)
already carries the retained-signal band under the identical
projection. Folding it in raises the mode count, and that section shows
the cost is one mode. Read this figure as a statement about spectral
shape overlap, not as the analysis EIGSEP will actually run: the
planned analysis is a joint differentiable forward-model fit, in which
beam chromaticity is modelled rather than filtered.

The ensemble runs to z = 4.65 (251.4 MHz), below Zeus21's advertised
z = 5-35 validity range. This is deliberate: it is what covers the
250 MHz band edge with computed values rather than extrapolated ones.
Zero-padding above Zeus21's native top of range (z = 5, 236.7 MHz) was
rejected instead, because late-reionization models still carry up to
~14 mK of signal at 237 MHz, and the resulting step discontinuity would
survive a smooth-mode filter and inflate the retained-RMS statistic this
figure reports.

Colouring is continuous rather than binned into classes on purpose.
Retention varies smoothly and no single statistic predicts it: trough
width tracks the retained *fraction* (Spearman -0.59) while the absolute
retained RMS correlates only moderately with amplitude (depth, Spearman
+0.53), so any class edge would cut a continuum and invite reading the
bins as populations. The norm is logarithmic -- 1.54 decades of range
(0.60-21.0 mK), where a linear norm would put 44% of models in the
bottom 10% of the colour range against 2% for log.

Curve opacity ramps with retained RMS (``ALPHA_LO`` to ``ALPHA_HI``,
``RAMP_POW``) rather than being uniform. The high-retention tail is
about 2% of the ensemble drawn under ~1700 other curves, and at any
single alpha that reads as empty. Draw order was tried first and does
not solve it: sorting by retention in either direction merely chooses
which end of the colour scale to bury, and rendering the two changes
separately shows the ramp alone reproduces the full effect. Order is
therefore a seeded random permutation, which biases neither end.

The ramp's cost is that opacity echoes the quantity colour already
encodes, so the top of the scale looks more populated than it is. That
is why ``RAMP_POW = 3`` rather than a linear ramp -- the median model
takes under a tenth of the available range (alpha 0.12 against 0.63 at
the top), so the boost stays confined to the genuinely extreme tail --
and why the caption should
quote the tail fraction (5.5% above 10 mK) rather than let the reader
estimate it off the panel.

``N_ANCHOR = 9`` is the smallest N at which the foreground residual
falls below the median retained signal *and stays below* for every
larger N (1.82 vs 3.01 mK; at N = 8 it is still above, 7.75 vs 4.94).
The "stays below" clause is not decoration -- these curves cross more
than once, because the median retained signal falls with N too, so a
first-crossing rule can select an N the floor later climbs back above.
``make_paper_horizon_figure.py`` applies the same rule to the position
systematic, so the two figures cannot disagree about which N clears
which floor.

Note there is no optimum to claim beyond this: the margin keeps
improving with N while the absolute signal shrinks, and where to stop is
set by thermal noise, which this calculation does not model. N = 9 is
the floor of the range, not a requirement -- each systematic folded in
later can only push it up.

Run in the mock_analysis env (numpy + matplotlib + nbformat):
    uv run python horizon_position/make_paper_signal_loss_figure.py
"""

import os
import sys
from pathlib import Path

import nbformat as nbf
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PAPER = Path("/home/christian/Documents/research/papers/eigsep_instrument/notebooks")
FG_NPZ = PAPER / "foreground_svd.npz"
SHIFT_NPZ = PAPER / "horizon_shift.npz"

sys.path.insert(0, str(HERE.parent / "models_21cm"))
import selection  # noqa: E402

# Generated by models_21cm/generate.py from Zeus21 (Munoz 2023a,
# arXiv:2302.08506) with Pop III and Lyman-Werner feedback (Cruz+2024,
# arXiv:2407.18294). The npz carries its own regeneration recipe: read
# its `provenance`, `generator_source` and `env_lock` keys. Spec:
# docs/superpowers/specs/2026-08-19-zeus21-model-ensemble-design.md
MODELS_NPZ = Path(
    "/home/christian/Documents/research/eigsep/mock_analysis/"
    "models_21cm/output/zeus21_models.npz"
)

N_ANCHOR = 9  # modes filtered at the quoted operating point -- set by the
# foreground residual alone; see recompute_operating_point.py
# Class edges on retained RMS [mK] at N_ANCHOR. 2.0 and 4.5 mK split the
# Zeus21 ensemble roughly into thirds (29.6% / 37.1% / 33.2%); 10 mK
# would catch 5.5% and 25 mK nothing at all (the most foreground-
# orthogonal model retains 21.0 mK).
RET_EDGES_MK = (2.0, 4.5)
CLASS_LABELS = ("< 2 mK", "2-4.5 mK", "> 4.5 mK")

# Mirrors of two values that live inside the notebook source strings below.
# build_text needs them as Python and asserts they still match, so the tex
# cannot quote a figure width or mode range the figure no longer has.
N_SHOW_TEXT = 18  # LOAD_SRC's N_SHOW
FIG_W_IN = 7.6  # PLOT_SRC's figsize width [in]


def load_t21(freqs):
    """The 21 cm model ensemble on `freqs` [K], reionization cut applied.

    The npz stores T21 already spline-interpolated onto the paper grid, so
    this only checks the grid and converts mK -> K. The reionization cut is
    posterior and is applied here, once, so the figure and the quoted
    statistics can never disagree about which models are in.

    `reionized_across_band` (not plain `reionized`) is deliberate: it also
    requires the model to be reionized at the top of the band, excluding 43
    models whose Zeus21 Q-solution re-neutralises at low z; 27 of those 43
    carry more than 1 mK of unphysical signal at 250 MHz. Expect 1769 of
    4096 models to survive.
    """
    m = np.load(MODELS_NPZ, allow_pickle=False)
    assert np.array_equal(m["freqs_MHz"], freqs), "frequency grid mismatch"
    keep = selection.reionized_across_band(m["xHI"], m["z_xHI"])
    return m["T21_mK"][keep] * 1e-3


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

    # Curves are subsampled for legibility; every statistic below uses the
    # full surviving ensemble. Storing the index (not a pre-subsampled
    # array) keeps the figure reproducible and the statistics honest.
    draw_seed = 20260819
    n_draw = min(1000, T21.shape[0])
    draw_idx = selection.figure_subsample(T21.shape[0], n_draw, draw_seed)

    np.savez_compressed(
        PAPER / "signal_loss.npz",
        freqs_MHz=freqs,
        Vh=Vh,
        s_fg=s_fg,
        n_time=n_time,
        T21_models=T21,
        cls=cls,
        class_labels=np.array(CLASS_LABELS),
        n_anchor=N_ANCHOR,
        draw_idx=draw_idx,
        draw_seed=draw_seed,
        description=(
            "Inputs for the 21 cm signal-loss figure. Vh (n_freq, n_freq) and "
            "s_fg (n_freq,) are the right singular vectors and singular values "
            "of the nominal antenna-temperature waterfall (foreground_svd.npz "
            "t_sys minus the constant receiver temperature, n_time LST samples) "
            "-- the same modes as Fig. 1. T21_models "
            "(n_model, n_freq) [K] is the full surviving 21 cm model ensemble "
            "(reionization cut applied) interpolated onto freqs_MHz; cls bins "
            "each model by the RMS it retains at N_ANCHOR, for the summary "
            "only. draw_idx (n_draw,) selects the subset of T21_models rows "
            "the figure actually draws as curves, for legibility -- every "
            "percentile, classification and summary statistic uses the full "
            "T21_models, not the draw. Signal and foregrounds are filtered by "
            "the same projection onto the leading modes of Vh. The +1 m "
            "antenna-position systematic is deliberately not here: it belongs "
            "to horizon_shift.npz, which carries it against the same retained-"
            "signal benchmark."
        ),
    )
    print(
        f"wrote {PAPER / 'signal_loss.npz'}  ({T21.shape[0]} models, "
        f"{draw_idx.size} drawn)"
    )


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
T21 = d["T21_models"]            # (n_model, n_f) global-signal ensemble [K]
cls = d["cls"]                   # (n_model,) retained-RMS bin, for the summary
class_labels = [str(x) for x in d["class_labels"]]
draw_idx = d["draw_idx"]         # (n_draw,) curves the figure draws
n_f = freqs.size
N_SHOW = 18                      # x-axis extent
N_ANCHOR = int(d["n_anchor"])    # modes filtered at the quoted operating point
# Opacity ramps with retained RMS, from ALPHA_LO at the bottom of the colour
# scale to ALPHA_HI at the top. At a single uniform alpha the sparse
# high-retention tail is invisible: it is ~2% of the ensemble, drawn under
# ~1700 other curves. The ramp, not the draw order, is what makes it legible
# -- rendering the two effects separately shows the ramp alone reproduces the
# result. The cost is that opacity now echoes the quantity colour already
# carries, which exaggerates how much of the ensemble sits at the top end;
# RAMP_POW > 1 keeps the boost confined to the genuinely extreme tail.
ALPHA_LO, ALPHA_HI, RAMP_POW = 0.07, 0.63, 3.0   # panel (b)
ALL_ALPHA_LO, ALL_ALPHA_HI = 0.07, 0.55          # panels (a1)/(a2)
Z_SEED = 20260820                # seeds the draw order; see PLOT_SRC
CONT_CMAP = "plasma"             # colour = retained RMS, continuous
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


t21_resid = filt_rms(T21)                                  # (N_SHOW+1, n_model)
t21_pct = np.percentile(t21_resid, [5, 50, 95], axis=1)    # (3, N_SHOW+1)
t21_filt = filtered(T21, N_ANCHOR)                         # (n_model, n_f) residuals'''

PLOT_SRC = r'''C_FG = "k"


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

    # Draw order is randomised, not sorted by retention. Sorting either way
    # puts one end of the colour scale underneath the whole ensemble; a seeded
    # permutation biases neither end, so the rendered density is the
    # ensemble's own. (The opacity ramp below, not this, is what actually
    # rescues the high-retention tail -- order alone cannot, without burying
    # the other end instead.)
    rng = np.random.default_rng(Z_SEED)
    order = rng.permutation(ret.size)

    # Panels (a1)/(a2) draw only the n_draw-model subsample, for legibility;
    # panel (b) below still draws every surviving model. Colour and the
    # colorbar (norm, above) are always keyed to the full ensemble.
    draw_ret = ret[draw_idx]
    draw_order = rng.permutation(draw_idx.size)

    def alpha_for(r, lo, hi):
        """Opacity rising with retained RMS, on the colour scale's own norm."""
        return lo + (hi - lo) * np.clip(np.asarray(norm(r)), 0, 1) ** RAMP_POW

    def segs(x, Y):
        return np.stack([np.broadcast_to(np.asarray(x), Y.shape), Y], axis=-1)

    fig, ax = plt.subplot_mosaic(
        [["a1", "b"], ["a2", "b"]],
        figsize=(7.6, 3.2), layout="constrained",
        gridspec_kw=dict(width_ratios=[1, 1.2]),
    )
    panels = (("a1", T21[draw_idx][draw_order] * 1e3),
              ("a2", t21_filt[draw_idx][draw_order] * 1e3))
    for key, Y in panels:
        lc = LineCollection(
            segs(freqs, Y), cmap=CONT_CMAP, norm=norm, lw=0.4,
            alpha=alpha_for(draw_ret[draw_order], ALL_ALPHA_LO, ALL_ALPHA_HI))
        lc.set_array(draw_ret[draw_order])
        ax[key].add_collection(lc)
        ax[key].set_xlim(freqs[0], freqs[-1])
        ax[key].set_ylim(Y.min() * 1.05, max(Y.max() * 1.05, 0.02 * abs(Y.min())))

    b = ax["b"]
    lc = LineCollection(segs(n_modes, t21_resid[:, order].T), cmap=CONT_CMAP,
                        norm=norm, lw=0.5,
                        alpha=alpha_for(ret[order], ALPHA_LO, ALPHA_HI))
    lc.set_array(ret[order])
    b.add_collection(lc)
    ref = [b.plot(n_modes, fg_resid, color=C_FG, lw=1.5,
                  label="foreground residual")[0]]

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
print(f"{'N':>3} {'fgnd':>9} {'21cm p50':>9} {'21cm p95':>9} "
      f"{'frac>fgnd':>10}   (mK)")
for N in (6, 8, N_ANCHOR, 10, 12, 15):
    print(f"{N:3d} {fg_resid[N]*1e3:9.3f} "
          f"{t21_pct[1, N]*1e3:9.3f} {t21_pct[2, N]*1e3:9.3f} "
          f"{frac_above[N]:10.2f}")

keep = t21_resid[N_ANCHOR] / t21_resid[0]
print(f"\\nAt N = {N_ANCHOR}: median model keeps {np.median(keep)*100:.0f}% of its "
      f"RMS ({t21_pct[1, N_ANCHOR]*1e3:.2f} mK), while the foreground residual is "
      f"{fg_resid[N_ANCHOR]*1e3:.2f} mK.")
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
        "Panels (a1)/(a2) show a legibility subsample of the surviving "
        "ensemble before and after filtering $N$ modes (panel (b) draws "
        "every model), each coloured by the RMS it retains at $N$ -- the same "
        "quantity plotted in panel (b), on the same log scale, so a curve's "
        "colour and its height in (b) agree. What survives is small but still "
        "structured; note it is band-edge ringing from projecting onto a "
        "truncated smooth basis, not a residual trough, so retained RMS is "
        "not retained signal *shape*. Panel (b) adds the foreground residual "
        "in black: colour marks the subject, greyscale the floor it is "
        "measured against. It supersedes `foreground_svd_residual.pdf` -- the "
        "black curve is the same one, now never shown without the signal "
        "beside it.\n\n"
        "**Scope.** This figure makes the smallest claim that supports the "
        "design argument: the beam-weighted foregrounds are spectrally "
        "low-dimensional, and a signal survives projecting that subspace "
        "out. Nothing else is folded in -- no instrumental systematics, no "
        "noise, and no knowledge of the beam or the sky. The $+1$ m "
        "antenna-position systematic was previously drawn here as a second "
        "floor and has been removed: it is not defined until the "
        "forward-modelling section, and `horizon_shift.ipynb` already shows "
        "it against this same retained-signal benchmark, where folding it in "
        "costs one additional mode. Read this as a statement about spectral "
        "shape overlap, not as the analysis EIGSEP will run -- the planned "
        "analysis is a joint differentiable forward-model fit in which beam "
        "chromaticity is modelled rather than filtered.\n\n"
        "**Retention is a continuum and no single summary statistic predicts "
        "it**, which is why the models are coloured continuously rather than "
        "binned into classes. It is set by how much of a model's spectral "
        "shape lies in the leading foreground modes. Separating shape from "
        "amplitude over the ensemble (Spearman): trough width correlates with "
        "the retained *fraction* at -0.59 -- narrower troughs keep "
        "proportionally more -- while the *absolute* retained RMS correlates "
        "with amplitude only moderately (depth, +0.53). Neither predicts "
        "retention alone: the three classes' median depths rise with "
        "retained RMS (77, 92, 156 mK for < 2 mK, 2-4.5 mK and "
        "> 4.5 mK) but even at matched depth (80-160 mK) they still separate "
        "by width -- median trough widths of 36, 20 and 18 MHz respectively. "
        "The summary cell bins the distribution at 2 and 4.5 mK for the "
        "caption; those bins are a reporting convenience, not "
        "populations.\n\n"
        "The colour scale is logarithmic. Retained RMS spans 1.54 decades "
        "(0.60-21.0 mK), and a linear norm would put 44% of the models in "
        "the bottom 10% of the colour range, against 2% for log.\n\n"
        "Curve opacity ramps with retained RMS rather than being uniform: the "
        "high-retention tail is ~2% of the ensemble drawn under ~1700 other "
        "curves, and at any single alpha it reads as empty. Draw order alone "
        "does not fix this -- sorting by retention just chooses which end of "
        "the colour scale to bury -- so the order is a seeded random "
        "permutation and the ramp does the work. Its cost is that opacity "
        "echoes the quantity colour already encodes, making the top of the "
        "scale look more populated than it is; the ramp is cubic to keep "
        "the boost confined to the extreme tail, and the true tail fraction "
        "(5.5% of models above 10 mK) is quoted rather than eyeballed.\n\n"
        "$N = 9$ is the smallest $N$ at which the foreground residual falls "
        "below the median retained signal *and stays below* for every larger "
        "$N$ (1.82 vs 3.01 mK; at $N = 8$ it is still above, 7.75 vs "
        "4.94 mK). The 'stays below' clause matters because both curves fall "
        "with $N$ and cross more than once, so a first-crossing rule can "
        "select an $N$ the floor later climbs back above; "
        "`horizon_shift.ipynb` applies the same rule to the position "
        "systematic. $N = 9$ is the floor of the range rather than a "
        "requirement -- every systematic folded in later can only push it "
        "up.\n\n"
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
        "Ensemble: 1769 of 4096 Zeus21 models (Munoz 2023a, "
        "arXiv:2302.08506, with Pop III and Lyman-Werner feedback, "
        "Cruz+2024, arXiv:2407.18294) survive a posterior reionization "
        "cut requiring xHI below threshold both at the McGreer+2015 "
        "reference redshift and at the top of the observed band (z = "
        "4.6816, 250 MHz). The npz carries its own regeneration recipe "
        "(`provenance`, `generator_source`, `env_lock` keys); see "
        "`docs/superpowers/specs/2026-08-19-zeus21-model-ensemble-design.md`."
        "\n\n"
        "The ensemble runs to $z = 4.65$ (251.4 MHz), below Zeus21's "
        "advertised $z = 5$-35 validity range. This is deliberate, so "
        "the 250 MHz band edge is covered by computed values rather than "
        "extrapolation; zero-padding above Zeus21's native top of range "
        "($z = 5$, 236.7 MHz) was considered and rejected instead, "
        "because late-reionization models still carry up to ~14 mK of "
        "signal at 237 MHz and the resulting step discontinuity would "
        "survive a smooth-mode filter and inflate the retained-RMS "
        "statistic reported here."
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


TEXT_TEMPLATE = r"""% signal_loss_text.tex -- GENERATED, do not edit by hand.
%
% Draft replacement text for the 21 cm signal-loss result. Regenerate with
%     uv run python horizon_position/make_paper_signal_loss_figure.py
% in the mock_analysis repo; every number in blocks 1, 2, 4 and 5 below is
% computed at generation time from the same arrays the figures are drawn from,
% so re-running after the 21 cm ensemble changes updates the prose and the
% figures together and they cannot drift apart. Block 3 is the exception and
% says so.
%
% Framing to preserve if these are edited: neither figure is a proposed
% analysis. Both project onto eigenmodes of a *simulated* nominal instrument,
% and block 5 gives the reason that matters -- an unmodelled 1 m displacement
% puts signal-like power in the first mode past the filter. Nothing here may
% be phrased so as to license reading a residual excess as a detection.
%
% Paste the five blocks into rasti_template.tex as marked. Nothing here is
% \input by the paper -- this file is a staging area, not a dependency, and
% nothing in this repo writes to the paper .tex itself.
%
% NOTE: signal_loss.pdf is @@FIGW@@in wide and must go in a `figure*'
% (double-column) environment, not the `figure' that foreground_svd_residual
% .pdf used -- at \linewidth in a single rasti column its tick labels render
% at roughly 3pt.


% ===================================================================
% BLOCK 1 -- section "Minimising Covariance with the 21-cm Signal".
% Replaces the sentence beginning "We emphasise that this analysis only
% quantifies the spectral complexity ...".
% ===================================================================

This analysis quantifies the spectral complexity of the beam-weighted
foregrounds; on its own it says nothing about whether the cosmological signal
survives the same filter. We therefore passed an ensemble of @@NMODELS@@ global
21-cm models through the identical projection, so that the retained signal and
the foreground residual are read off the same axes
(Fig.~\ref{fig:singular_values}). We adopt $N=@@NA@@$ filtered modes as a
reference operating point: it is the smallest $N$ at which the foreground
residual (@@FG@@\,mK) falls below the median retained signal and stays below it
for every larger $N$; at $N=@@NAM1@@$ the residual is @@FGM1@@\,mK against a
median retained signal of @@MEDM1@@\,mK. At the reference point the median
model retains @@KEEPPCT@@ per cent of its band RMS, or @@MED@@\,mK, and
@@ABOVE@@ per cent of the ensemble retains more than the foreground residual.
Retention correlates with signal amplitude but is not determined by it
(Spearman @@RHODEPTH@@ against trough depth): of the @@N150@@ models with an
absorption trough of 140--160\,mK, the retained RMS has a median of
@@RET150@@\,mK but an interquartile range of @@RET150LO@@--@@RET150HI@@\,mK.
What separates them at matched depth is trough width, narrower troughs
retaining the larger fraction (Spearman @@RHOWIDTH@@ between width and
retained fraction), since a broad trough is the more nearly foreground-like
shape.

This operating point is set by the foregrounds alone, which is all that
Fig.~\ref{fig:singular_values} is asked to establish. Instrumental systematics
are folded in later and can only raise it. The 1\,m antenna displacement of
section~\ref{subsec:fwd_modelling} is one such term: its worst-case residual is
@@SYSNA@@\,mK at $N=@@NA@@$, comparable to the median retained signal there,
and one further mode brings it to @@SYSNP1@@\,mK against @@MEDNP1@@\,mK
retained (Fig.~\ref{fig:horizon_shift}). We report such crossings under a
`stays below' rule -- the smallest $N$ beyond which the floor never rises above
the median again -- because both quantities fall with $N$ and cross more than
once, so a first-crossing count would be optimistic. We stress that $N$ is used
here to characterise spectral overlap, and not as a filter depth we propose to
apply to data and then interpret the residual of: an unmodelled displacement
leaves signal-like power in precisely the first mode such a filter would keep,
as section~\ref{subsec:fwd_modelling} sets out.

Retained RMS understates what such a filter leaves measurable. The projection
discards the components of a signal that lie along the foreground modes and
keeps the orthogonal complement, so the relevant question is not how much
amplitude survives but which models remain distinguishable within the subspace
that does. Restricting to the @@NDEEP@@ ensemble members with absorption
troughs deeper than 50\,mK and scoring over 70--130\,MHz, the median separation
between a pair of models falls from @@SEP0@@\,mK to @@SEP1@@\,mK under the
filter, yet @@PAIR1@@ per cent of pairs remain separated by more than 1\,mK and
@@PAIR2@@ per cent by more than 2\,mK. A measurement in the filtered subspace
therefore constrains the signal to a family of models differing by components
that lie along the foreground modes, rather than collapsing the ensemble to an
indistinguishable residual.

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
% the horizon_shift caption already refers to it. Use figure*, not figure.
% ===================================================================

\caption{Signal loss under foreground-mode filtering. An ensemble of
@@NMODELS@@ global 21-cm models is passed through the same projection onto the
leading $N$ eigenmodes of the simulated antenna temperature that is applied to
the foregrounds. Panels (a1) and (a2) show a random subsample of the ensemble
before and after filtering $N=@@NA@@$ modes, each coloured by the band RMS it
retains at that operating point; the colour scale is logarithmic and shared
with panel (b), so a curve's colour and its height in (b) agree. Curve opacity
also rises with retained RMS, so that the sparse high-retention tail is
visible; it is a small minority, @@TAIL10@@ per cent of models above 10\,mK.
Panel (b) shows retained RMS against the number of modes filtered for every
model, with the foreground residual in black. At $N=@@NA@@$ the foreground
residual is @@FG@@\,mK while the median model retains @@MED@@\,mK, or
@@KEEPPCT@@ per cent of its band RMS, and @@ABOVE@@ per cent of the ensemble
retains more than the foreground residual. The structure surviving in (a2) is
ringing from projecting onto a truncated smooth basis rather than a residual
absorption trough, so retained RMS should not be read as retained signal shape.
This filter uses no knowledge of the beam or the sky and is therefore the most
conservative case; instrumental systematics are folded in at
Fig.~\ref{fig:horizon_shift}. See the text.}


% ===================================================================
% BLOCK 3 -- section "Minimising Covariance", rotation paragraph.
% Insert after "... we aim to use both to improve constraints."
%
% These numbers are NOT computed by this script. They come from a probe run
% on horizon_chromaticity/output/chromaticity_eigsep.npz (t_sys, 1296 ori x
% 1436 LST x 201 freq) on 2026-08-19, which is not checked in. Re-derive from
% that cube if the ensemble or the beam changes.
%
% Read the "16 versus 10" carefully: both are matched-residual counts at a
% 0.6 mK foreground residual, NOT operating points. The zenith figure happens
% to be 10 because that is where the zenith residual reaches 0.6 mK; the
% operating point in block 1 is N = @@NA@@, where the residual is @@FG@@ mK.
% Do not let a reader conflate the two.
% ===================================================================

Rotation adds spectral diversity to the data, and it is worth asking what that
diversity costs in model complexity. Repeating the eigenmode analysis over the
full drive grid of 1296 pointings (36 elevations $\times$ 36 azimuths) rather
than the zenith pointing alone, the pooled antenna temperature reaches a
foreground residual of 0.6\,mK with 16 modes, against 10 for the zenith
pointing at the same residual. The foregrounds seen across every accessible
orientation therefore occupy a subspace only marginally larger than that of a
single pointing. Realising the benefit of that diversity requires a joint fit
in which the sky and the beam are shared parameters and the rotations are
known; a per-spectrum projection of the kind used in
Fig.~\ref{fig:singular_values} cannot exploit it, because the information lies
in the correlation between pointings rather than within any one spectrum, and
pooling the pointings into the projection basis only enlarges the subspace
being removed. We defer this analysis to future work.


% ===================================================================
% BLOCK 4 -- caption for horizon_shift.pdf, replacing the existing one.
% Keep \label{fig:horizon_shift} and the figure* environment.
%
% The caption being replaced describes a figure that no longer exists: it
% promises a "dashed red line" at 10 mK, which was swapped for the retained
% 21 cm band, and counts crossings as "within six modes", which was a
% first-crossing count. Under the stays-below rule used throughout, the
% per-axis answers are @@CLEAR_E@@ (east), @@CLEAR_N@@ (north) and
% @@CLEAR_U@@ (up).
% ===================================================================

\caption{Change in the simulated antenna temperature,
$\Delta T_{\text{ant}}$, when the suspended antenna is displaced by 1\,m to
the east (left column), north (middle column), and up (right column). Each
curve corresponds to a different LST (one per sidereal hour, @@NLST@@ in
total), coloured as indicated by the colour bar. The displacement shifts the
local horizon, shown in Fig.~\ref{fig:horizon}(b), thereby changing the
fraction of sky occulted by the canyon walls and hence $T_{\text{ant}}$. The
bottom row shows, for each LST, the RMS over frequency of the difference
spectrum after filtering the eigenmodes of the unperturbed antenna temperature
-- the same modes as Fig.~\ref{fig:singular_values} -- as a function of the
number of modes filtered. The grey band is the 21 cm signal retained under
that identical projection (5--95 per cent of the model ensemble of
Fig.~\ref{fig:singular_values}, median dashed); the dotted vertical line marks
the $N=@@NA@@$ operating point adopted there. The displacements are large in
amplitude -- up to @@MAXDT@@\,K at 50\,MHz, @@RAW_U@@\,mK RMS for the upward
shift against @@RAW_E@@\,mK east and @@RAW_N@@\,mK north -- but they are
foreground-like, with @@LEADPCT@@ per cent of that power in the two leading
foreground modes, so the same low-order filtering that removes the sky removes
almost all of them: the east and north residuals stay below the median
retained signal from @@CLEAR_E@@ and @@CLEAR_N@@ modes on. What survives is
narrow rather than smooth. After @@NA@@ modes are filtered, @@SPIKEPCT@@ per
cent of the upward displacement's remaining power sits in mode @@SPIKE@@
alone, at @@SPIKESYS@@\,mK -- above that mode's nominal foreground content
(@@SPIKEFG@@\,mK) and above the median model's 21 cm content there
(@@SPIKE21@@\,mK), and resembling the retained signal in shape (cosine
similarity up to @@COSMAX@@). We therefore do not treat this filter as an
analysis that could be run on data and its residual attributed to cosmology;
see the text.}


% ===================================================================
% BLOCK 5 -- section "Forward Modelling", after the existing discussion of
% the horizon_shift results ("... Motion up or down produces the largest
% change in antenna temperature ...").
%
% This is the paragraph that keeps the two figures from being read as a
% proposed pipeline. Both of them project onto eigenmodes of a *simulated*
% nominal instrument; neither is the analysis EIGSEP intends to run, and the
% numbers below are the reason why.
% ===================================================================

Two things follow from Fig.~\ref{fig:horizon_shift}, and they pull in opposite
directions. The reassuring one is that a position error does not introduce a
new class of spectral structure. The induced $\Delta T_{\text{ant}}$ is large
in amplitude, but @@LEADPCT@@ per cent of its power lies in the two leading
eigenmodes of the unperturbed antenna temperature: it is, to that accuracy,
more foreground. Low-order filtering of the kind that motivates the EIGSEP
design in section~\ref{subsec:covariance} therefore removes almost all of it,
and the eastward and northward displacements fall below the median retained
21 cm signal after @@CLEAR_E@@ and @@CLEAR_N@@ modes respectively.

The cautionary one concerns what is left. Filtering @@NA@@ modes -- the
operating point Fig.~\ref{fig:singular_values} adopts from the foregrounds
alone -- leaves the upward displacement with @@SYSNA@@\,mK, of which
@@SPIKEPCT@@ per cent is concentrated in a single mode, mode @@SPIKE@@. That
one mode carries more power from a 1\,m displacement (@@SPIKESYS@@\,mK) than
it does from the nominal foregrounds (@@SPIKEFG@@\,mK) or from the median
21 cm model (@@SPIKE21@@\,mK); @@SPIKEBELOW@@ per cent of the ensemble has
less signal in that mode than a 1\,m shift would put there, and the leftover
resembles the retained signal in shape closely enough to be confused with it
(cosine similarity up to @@COSMAX@@, median @@COSMED@@). The response is
linear in displacement, so holding the injected power to a tenth of the median
retained signal requires the vertical position to be known to roughly
@@SPECM@@\,m -- a requirement on position monitoring, which we adopt.

The wider point is methodological. An unmodelled displacement of the size we
must anticipate deposits power that is signal-like in both amplitude and
shape, at exactly the mode boundary where filtering stops. A residual left by
a filter of fixed depth is therefore not evidence of a cosmological signal:
excess with respect to a foreground model is only as trustworthy as the
instrument model behind it, a lesson the field has already drawn from the
scrutiny of published detections
\citep{2018Natur.564E..32H, 2019ApJ...874..153B, 2019ApJ...880...26S}.
Figs.~\ref{fig:singular_values} and~\ref{fig:horizon_shift} should accordingly
be read as characterisations of spectral structure -- how many modes the
instrument-weighted foregrounds occupy, and how a position error compares with
the signal within that basis -- and not as a proposed pipeline. EIGSEP's
analysis instead forward-models the instrument (this section), with the
antenna position among the parameters that are fitted and marginalised over
rather than assumed. The differentiable forward model makes that
marginalisation tractable, and the position sensitivities computed here are
what set the priors it needs.
"""


def build_text():
    """Regenerate the draft paper text, every block-1/2 number computed here.

    The previous version of this file was written once and then went stale:
    it still quoted a 1135-model ensemble and an N = 10 operating point set
    partly by the position systematic, months after both had changed. Deriving
    the numbers here, from the same arrays `build_data` saves, is what stops
    that recurring -- the prose cannot drift from the figure without someone
    editing a file that says it is generated.
    """
    assert f"N_SHOW = {N_SHOW_TEXT}" in LOAD_SRC, "N_SHOW_TEXT is stale"
    assert f"figsize=({FIG_W_IN}," in PLOT_SRC, "FIG_W_IN is stale"

    d = np.load(PAPER / "signal_loss.npz", allow_pickle=True)
    freqs, Vh, s_fg = d["freqs_MHz"], d["Vh"], d["s_fg"]
    T21, n_time = d["T21_models"], int(d["n_time"])
    n_f = freqs.size
    n_modes = np.arange(N_SHOW_TEXT + 1)

    tail = np.concatenate([np.cumsum(s_fg[::-1] ** 2)[::-1], [0.0]])
    fg_resid = np.sqrt(tail / (n_time * n_f))[: N_SHOW_TEXT + 1] * 1e3  # mK

    c = T21 @ Vh.T
    ret_all = np.array(
        [np.sqrt(np.sum(c[:, N:] ** 2, axis=1) / n_f) * 1e3 for N in n_modes]
    )
    med = np.median(ret_all, axis=1)
    ret = ret_all[N_ANCHOR]

    # +1 m position systematic, from the horizon figure's own npz. Block 4
    # captions that figure, so the per-axis numbers are derived here too --
    # by index into dT_spectra, with the label order asserted, so the caption
    # cannot silently transpose east for up if TAGS is ever reordered.
    shift = np.load(SHIFT_NPZ, allow_pickle=True)
    dT3 = shift["dT_spectra"]  # (3, n_lst, n_f), axis order East/North/Up
    assert [str(s) for s in shift["labels"]] == [
        "East +1 m",
        "North +1 m",
        "Up +1 m",
    ], "horizon_shift.npz axis order changed; block 4 wording assumes E/N/U"

    def worst_lst(x):
        """Worst-LST residual RMS [mK] vs modes filtered, for one axis."""
        cx = np.atleast_2d(x) @ Vh.T
        return np.array(
            [np.sqrt(np.sum(cx[:, N:] ** 2, axis=1) / n_f).max() * 1e3 for N in n_modes]
        )

    per_axis = np.stack([worst_lst(dT3[k]) for k in range(3)])  # (3, N_SHOW+1)
    sys_r = worst_lst(dT3.reshape(-1, n_f))  # worst over axis and LST

    def stays_below(curve, ref):
        """Smallest N with curve < ref there and at every larger N on the axis."""
        below = curve < ref
        return next(int(N) for N in n_modes if below[N:].all())

    # Blocks 1 and 4 both say the fold-in "costs one mode" in prose, and quote
    # the N + 1 figures to match. Nothing recomputes that phrase, so gate it:
    # if the ensemble or the sims ever move the crossing, this fails here
    # rather than shipping a caption whose words and numbers disagree.
    cost = stays_below(sys_r, med) - N_ANCHOR
    assert cost == 1, (
        f"position systematic now clears the median {cost} modes after "
        f"N_ANCHOR={N_ANCHOR}, not 1 -- reword blocks 1 and 4, which say "
        "'one further mode' and 'costs one mode' in prose"
    )

    # --- anatomy of what the displacement leaves behind ------------------
    # Both halves of the horizon figure's message are quantified here. The
    # reassuring half: nearly all of a displacement's power lands in the two
    # leading foreground modes, so it is more foreground, not a new kind of
    # structure. The cautionary half: what survives the filter is not a smooth
    # tail but a spike in one mode, at the amplitude and roughly the shape of
    # the signal -- which is why a residual excess cannot be read as cosmology.
    up = dT3[2]  # worst axis at every N; see per_axis above
    cu = up @ Vh.T
    j = int(np.argmax(np.sqrt(np.sum(cu[:, N_ANCHOR:] ** 2, axis=1))))  # worst LST
    mode_mK = np.abs(cu[j]) / np.sqrt(n_f) * 1e3  # per-mode RMS contribution
    lead_frac = np.sum(cu[j, :2] ** 2) / np.sum(cu[j] ** 2)
    spike = int(np.argmax(mode_mK[N_ANCHOR:])) + N_ANCHOR  # 0-indexed
    tail_pow = np.sum(mode_mK[N_ANCHOR:] ** 2)

    # the same mode's nominal-foreground and 21 cm content, for comparison
    fg_mode_mK = s_fg / np.sqrt(n_time * n_f) * 1e3
    sig_mode_mK = np.abs(c[:, spike]) / np.sqrt(n_f) * 1e3

    # shape confusion between the leftover systematic and the retained signal
    resid_sys = cu[j, N_ANCHOR:] @ Vh[N_ANCHOR:]
    r21 = c[:, N_ANCHOR:] @ Vh[N_ANCHOR:]
    cos = np.abs(r21 @ resid_sys) / (
        np.linalg.norm(r21, axis=1) * np.linalg.norm(resid_sys)
    )

    # The effect is linear in displacement (checked against the 0.1/1/10 m
    # sims), so the position knowledge that holds the injected power to a tenth
    # of the median retained signal follows by scaling the 1 m case.
    spec_m = 0.1 * med[N_ANCHOR] / mode_mK[spike]

    depth = -T21.min(axis=1) * 1e3
    near150 = (depth > 140) & (depth < 160)
    # Trough width at half depth, the shape statistic that separates models of
    # equal amplitude; correlated against the retained *fraction*, not the
    # absolute RMS, so amplitude is divided out before shape is measured.
    width = (T21 < T21.min(axis=1, keepdims=True) / 2).sum(axis=1) * (
        freqs[1] - freqs[0]
    )
    rho_depth = spearmanr(depth, ret_all[N_ANCHOR]).statistic
    rho_width = spearmanr(width, ret_all[N_ANCHOR] / ret_all[0]).statistic

    # Distinguishability: pairwise RMS separation over 70-130 MHz, before and
    # after the filter, among models with a trough deeper than 50 mK. Computed
    # from the Gram matrix rather than an (n, n, n_band) difference array,
    # which would not fit in memory at this ensemble size.
    band = (freqs >= 70) & (freqs <= 130)
    deep = depth > 50

    def pair_sep(X):
        """Upper-triangle pairwise RMS separations over the scoring band [mK]."""
        A = X[deep][:, band] * 1e3
        g = A @ A.T
        dg = np.diag(g)
        d2 = np.maximum(dg[:, None] + dg[None, :] - 2 * g, 0.0)
        iu = np.triu_indices(A.shape[0], k=1)
        return np.sqrt(d2[iu] / band.sum())

    t21_filt = (c[:, N_ANCHOR:] @ Vh[N_ANCHOR:]).reshape(T21.shape)
    sep0, sep1 = pair_sep(T21), pair_sep(t21_filt)

    vals = {
        "FIGW": f"{FIG_W_IN:g}",
        "NMODELS": T21.shape[0],
        "NA": N_ANCHOR,
        "NAM1": N_ANCHOR - 1,
        "FG": f"{fg_resid[N_ANCHOR]:.2f}",
        "FGM1": f"{fg_resid[N_ANCHOR - 1]:.2f}",
        "MEDM1": f"{med[N_ANCHOR - 1]:.2f}",
        "MED": f"{med[N_ANCHOR]:.2f}",
        "MEDNP1": f"{med[N_ANCHOR + 1]:.2f}",
        "KEEPPCT": f"{np.median(ret_all[N_ANCHOR] / ret_all[0]) * 100:.0f}",
        "ABOVE": f"{(ret > fg_resid[N_ANCHOR]).mean() * 100:.0f}",
        "N150": int(near150.sum()),
        "RET150": f"{np.median(ret[near150]):.1f}",
        "RET150LO": f"{np.percentile(ret[near150], 25):.1f}",
        "RET150HI": f"{np.percentile(ret[near150], 75):.1f}",
        "RHODEPTH": f"{rho_depth:+.2f}",
        "RHOWIDTH": f"{rho_width:+.2f}",
        "SYSNA": f"{sys_r[N_ANCHOR]:.2f}",
        "SYSNP1": f"{sys_r[N_ANCHOR + 1]:.2f}",
        "TAIL10": f"{(ret > 10).mean() * 100:.1f}",
        "NDEEP": int(deep.sum()),
        "SEP0": f"{np.median(sep0):.0f}",
        "SEP1": f"{np.median(sep1):.1f}",
        "PAIR1": f"{(sep1 > 1).mean() * 100:.0f}",
        "PAIR2": f"{(sep1 > 2).mean() * 100:.0f}",
        "NLST": dT3.shape[1],
        "CLEAR_E": stays_below(per_axis[0], med),
        "CLEAR_N": stays_below(per_axis[1], med),
        "CLEAR_U": stays_below(per_axis[2], med),
        # RMS over every LST and channel, NOT the worst-LST value the residual
        # panels track. This is the statistic the surrounding body text already
        # quotes for these three displacements, and a caption that switched to
        # worst-LST would read as contradicting it.
        "RAW_E": f"{np.sqrt(np.mean(dT3[0] ** 2)) * 1e3:.0f}",
        "RAW_N": f"{np.sqrt(np.mean(dT3[1] ** 2)) * 1e3:.0f}",
        "RAW_U": f"{np.sqrt(np.mean(dT3[2] ** 2)) * 1e3:.0f}",
        "MAXDT": f"{np.abs(dT3[2]).max():.1f}",
        "LEADPCT": f"{lead_frac * 100:.1f}",
        "SPIKE": spike + 1,  # 1-indexed for the prose
        "SPIKEPCT": f"{mode_mK[spike] ** 2 / tail_pow * 100:.0f}",
        "SPIKESYS": f"{mode_mK[spike]:.2f}",
        "SPIKEFG": f"{fg_mode_mK[spike]:.2f}",
        "SPIKE21": f"{np.median(sig_mode_mK):.2f}",
        "SPIKEBELOW": f"{(sig_mode_mK < mode_mK[spike]).mean() * 100:.0f}",
        "COSMED": f"{np.median(cos):.2f}",
        "COSMAX": f"{cos.max():.2f}",
        "SPECM": f"{spec_m:.1f}",
    }
    out = TEXT_TEMPLATE
    for k, v in vals.items():
        out = out.replace(f"@@{k}@@", str(v))
    assert "@@" not in out, "unsubstituted token left in signal_loss_text.tex"

    (PAPER / "signal_loss_text.tex").write_text(out)
    print(f"wrote {PAPER / 'signal_loss_text.tex'}")
    for k, v in vals.items():
        print(f"    {k:9s} {v}")


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


def main():
    for p in (FG_NPZ, SHIFT_NPZ, MODELS_NPZ):
        if not p.exists():
            raise SystemExit(f"{p} not found")
    build_data()
    build_notebook()
    build_text()
    render_pdf()


if __name__ == "__main__":
    main()
