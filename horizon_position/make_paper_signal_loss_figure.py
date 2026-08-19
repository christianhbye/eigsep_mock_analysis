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

Panels (a1)/(a2): representative models before and after filtering
``N_ANCHOR`` modes. What survives is band-edge ringing from projecting
onto a truncated smooth basis, not a residual trough -- retained RMS is
not retained signal *shape*, which is why the panel is here at all.
Panel (b): the foreground residual, the worst-case position-error
systematic, and the retained 21 cm signal against the number of modes
filtered, drawn as individual model curves rather than a summary band so
the spread of trajectories is visible. This panel supersedes
``foreground_svd_residual.pdf``: the foreground curve is the same one,
now never shown without the signal beside it.

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
PAPER = Path("/home/christian/Documents/research/papers/eigsep_instrument/notebooks")
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
MATCH_DEPTH_MK = (80.0, 160.0)  # depth window the example models are drawn from


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


def trough_width(T21, freqs):
    """Width [MHz] of each absorption trough at half its depth."""
    return (T21 < T21.min(axis=1, keepdims=True) / 2).sum(axis=1) * (
        freqs[1] - freqs[0]
    )


def pick_models(T21, freqs, cls):
    """One model per class, drawn from a common depth window.

    At matched depth the classes separate almost entirely by trough
    *width*: the smooth low-order foreground modes absorb broad troughs
    and leave narrow ones (median width 73 / 30 / 17 MHz going from the
    destroyed class to the surviving one). Selecting inside a common
    depth window isolates that, rather than confounding it with
    amplitude -- most of the destroyed class is simply faint to begin
    with (median depth 2.7 mK).
    """
    depth = -T21.min(axis=1) * 1e3
    width = trough_width(T21, freqs)
    lo, hi = MATCH_DEPTH_MK
    window = (depth > lo) & (depth < hi)
    out = []
    for k in range(len(CLASS_LABELS)):
        m = np.nonzero(window & (cls == k))[0]
        out.append(m[np.argmin(np.abs(width[m] - np.median(width[m])))])
    return np.array(out)


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
    show_idx = pick_models(T21, freqs, cls)

    np.savez_compressed(
        PAPER / "signal_loss.npz",
        freqs_MHz=freqs,
        Vh=Vh,
        s_fg=s_fg,
        n_time=n_time,
        dT_spectra=shift["dT_spectra"],
        labels=shift["labels"],
        T21_models=T21,
        show_idx=show_idx,
        cls=cls,
        show_width_MHz=trough_width(T21, freqs)[show_idx],
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
            "interpolated onto freqs_MHz; show_idx selects the three drawn in "
            "panels (a1)/(a2). All three quantities are filtered by the same "
            "projection onto the leading modes of Vh."
        ),
    )
    print(f"wrote {PAPER / 'signal_loss.npz'}  ({T21.shape[0]} models)")


# --- source shared by the notebook and the direct render (kept in sync) ---

IMPORTS_SRC = """import numpy as np
import matplotlib.pyplot as plt"""

LOAD_SRC = """d = np.load("signal_loss.npz", allow_pickle=True)
freqs = d["freqs_MHz"]           # (n_f,) MHz
Vh = d["Vh"]                     # (n_f, n_f) foreground spectral modes
s_fg = d["s_fg"]                 # (n_f,) singular values of the T_ant waterfall
n_time = int(d["n_time"])        # LST samples in that waterfall
dT = d["dT_spectra"]             # (3, n_lst, n_f) +1 m E/N/U systematic [K]
T21 = d["T21_models"]            # (n_model, n_f) global-signal ensemble [K]
show_idx = d["show_idx"]         # the three models drawn in panels (a1)/(a2)
cls = d["cls"]                   # (n_model,) class 0/1/2 by retained RMS at N_ANCHOR
show_w = d["show_width_MHz"]     # trough width [MHz] of each drawn model
class_labels = [str(x) for x in d["class_labels"]]
n_f = freqs.size
N_SHOW = 18                      # x-axis extent
N_ANCHOR = int(d["n_anchor"])    # modes filtered at the quoted operating point
N_CURVES = 500                   # individual signals drawn in panel (b)
CURVE_ALPHA = 0.10               # opacity of those curves
ALL_ALPHA = 0.08                 # opacity of the full ensemble in panel (a)
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
CLASS_C = ["#cc79a7", "#e69f00", "#0072b2"]                 # destroyed -> survives
rng = np.random.default_rng(0)                              # fixed draw, reproducible
sub = rng.choice(t21_resid.shape[1], size=N_CURVES, replace=False)


def make_figure(path, show_all):
    """Panel (a) shows either the three exemplars alone or the whole ensemble."""
    fig, ax = plt.subplot_mosaic(
        [["a1", "b"], ["a2", "b"]],
        figsize=(7.3, 3.2), layout="constrained",
        gridspec_kw=dict(width_ratios=[1, 1.2]),
    )
    b = ax["b"]

    for k, lab in enumerate(class_labels):                  # colour by fate at N
        kk = np.nonzero(cls == k)[0]
        # The destroyed class is the sparsest and the faintest, so draw it
        # on top: class 0 gets the highest of these zorders.
        z = 0.1 * (len(class_labels) - k)
        if show_all:                                        # every model, left column
            ax["a1"].plot(freqs, T21[kk].T * 1e3, color=CLASS_C[k], lw=0.4,
                          alpha=ALL_ALPHA, zorder=z)
            ax["a2"].plot(freqs, t21_filt[kk].T * 1e3, color=CLASS_C[k], lw=0.4,
                          alpha=ALL_ALPHA, zorder=z)
        sk = sub[cls[sub] == k]                             # subsample, right panel
        b.plot(n_modes, t21_resid[:, sk], color=CLASS_C[k], lw=0.5,
               alpha=CURVE_ALPHA, zorder=z)
        b.plot([], [], color=CLASS_C[k], lw=1.2,            # legend proxy
               label=f"{lab} retained ({kk.size})")

    for i in show_idx:                                      # exemplars, both sides
        c = CLASS_C[cls[i]]
        ax["a1"].plot(freqs, T21[i] * 1e3, color=c, lw=0.9)
        ax["a2"].plot(freqs, t21_filt[i] * 1e3, color=c, lw=0.9)
        b.plot(n_modes, t21_resid[:, i], color=c, lw=1.0, zorder=2)

    for key, lab, ylab in (("a1", "input", r"$T_{21}$ [mK]"),
                           ("a2", f"after filtering {N_ANCHOR} modes",
                            "Residual [mK]")):
        ax[key].axhline(0, color="0.6", lw=0.6, ls="--", zorder=0)
        ax[key].set_ylabel(ylab, fontsize=8)
        ax[key].grid(alpha=0.2)
        ax[key].tick_params(labelsize=7)
        ax[key].text(0.03 if key == "a1" else 0.97, 0.06, lab,
                     transform=ax[key].transAxes, fontsize=7,
                     ha="left" if key == "a1" else "right", va="bottom")
    ax["a1"].tick_params(labelbottom=False)
    ax["a2"].set_xlabel("Frequency [MHz]", fontsize=8)

    ref = [b.plot(n_modes, fg_resid, color=C_FG, lw=1.5,
                  label="foreground residual")[0],
           b.plot(n_modes, sys_resid, color=C_SYS, lw=1.4, ls="--",
                  label="+1 m position error (worst LST)")[0]]
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
    cls_leg = b.legend(handles=[h for h in b.get_lines() if h.get_label()
                                in [f"{x} retained ({(cls == k).sum()})"
                                    for k, x in enumerate(class_labels)]],
                       title=f"21 cm signal retained at $N = {N_ANCHOR}$",
                       fontsize=6.5, title_fontsize=6.5, loc="upper right",
                       framealpha=0.92)
    b.add_artist(cls_leg)
    b.legend(handles=ref, fontsize=6.5, loc="lower left", framealpha=0.92)

    fig.savefig(path, bbox_inches="tight", dpi=600)


make_figure("signal_loss.pdf", show_all=False)              # three exemplars
make_figure("signal_loss_all.pdf", show_all=True)           # whole ensemble'''

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

print()
print("highlighted models: " + ", ".join(
    f"{class_labels[cls[i]]} / {w:.0f} MHz wide" for i, w in zip(show_idx, show_w)))

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
        "Panels (a1)/(a2) show one model per outcome class -- all three drawn "
        "from a common depth window (80-160 mK) so that trough *width*, not "
        "amplitude, is the visible difference -- before and after filtering "
        "$N$ modes. Each is highlighted as a thick curve of the same colour "
        "in panel (b), so the two sides can be read against each other; "
        "colour means outcome class throughout, so panel (a) carries no "
        "legend of its own. Their trough widths are printed by the summary "
        "cell, for the caption. "
        "`signal_loss_all.pdf` is the same figure with the whole ensemble "
        "drawn in the left column rather than the three exemplars. What "
        "survives is small but still "
        "structured -- note it is band-edge ringing from projecting onto a "
        "truncated smooth basis, not a residual trough, so retained RMS is "
        "not retained signal *shape*. Panel (b) "
        "puts the foreground residual, the worst-case $+1$ m position-error "
        "systematic, and 500 individual retained-signal curves coloured by "
        "outcome class on one set of axes. It supersedes "
        "`foreground_svd_residual.pdf` -- the black curve is the same one, now "
        "never shown without the signal beside it.\n\n"
        "Models are classed by the RMS they retain at $N = 10$: below 1 mK, "
        "1-5 mK, and above 5 mK, which splits the ensemble roughly into "
        "thirds. Higher cuts are not useful here -- 10 mK catches 2.6% of "
        "models and 25 mK none, since the most foreground-orthogonal model "
        "retains 16.4 mK. **The discriminator is trough width.** At matched "
        "depth the median width runs 73 / 30 / 17 MHz from the destroyed "
        "class to the surviving one: the smooth low-order foreground modes "
        "absorb broad troughs and leave narrow ones. Amplitude matters too, "
        "but most of the destroyed class is simply faint to begin with "
        "(median depth 2.7 mK).\n\n"
        "$N = 10$ is the smallest $N$ at which *both* floors fall below the "
        "median retained signal. At $N = 8$ the foreground residual and the "
        "median signal are the same size.\n\n"
        "**Limitations, to be stated wherever this result is used.** The "
        "modes come from a single simulated sky (GSM16) and beam, with no "
        "noise and no receiver systematics; in practice the basis would be "
        "estimated from data that already contain the signal, which costs "
        "additional signal loss not captured here. Filtering is a hard "
        "projection, whereas a joint signal-plus-foreground fit would recover "
        "some of what is removed. Signal loss is severe in absolute terms, and "
        "whether the retained amplitude is detectable is set by thermal noise "
        "and integration time, which this calculation does not model. This is "
        "a statement about spectral subspace overlap, not a sensitivity "
        "forecast.\n\n"
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


def main():
    for p in (FG_NPZ, SHIFT_NPZ, MODELS_NPZ):
        if not p.exists():
            raise SystemExit(f"{p} not found")
    build_data()
    build_notebook()
    render_pdf()


if __name__ == "__main__":
    main()
