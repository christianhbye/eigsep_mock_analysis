"""Two-panel horizon figure for the paper (wide + single-column).

Panel (a): the baseline horizon elevation profile alpha_h(az) (cf.
``nominal_horizon.png``). Panel (b): the horizon shift
Delta alpha_h(az) = alpha_h(shifted) - alpha_h(nominal) for +1 m antenna
displacements East / North / Up, sharing the azimuth axis as a residual
panel.

The figure makes the scale of the perturbation legible: a 1 m move
changes the horizon by <~0.1 deg over most azimuths and spikes to ~1 deg
only at steep cliff edges (where a lateral move slides a near-vertical
horizon edge sideways), while raising the antenna (Up +1 m) lowers the
horizon by a near-uniform small offset.

Standalone; numpy + matplotlib only, default env:

    uv run python horizon_position/make_horizon_figure.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
NPZ = HERE / "output" / "horizons_position.npz"

# +1 m shift directions, with colour-blind-safe (Okabe-Ito) colours that
# stay clear of the orange baseline fill.
SHIFTS = [
    ("x_p_1", "East +1 m", "#0072B2"),  # blue
    ("y_p_1", "North +1 m", "#CC79A7"),  # reddish purple
    ("z_p_1", "Up +1 m", "#009E73"),  # bluish green
]
FILL = "tab:orange"


def build_figure(az, alpha, names, base, figsize, legend_kw, resid_ylim=(-1.3, 1.3)):
    """Build the stacked baseline/residual figure at the given size.

    ``resid_ylim`` sets the residual-panel limits; the narrow
    single-column variant extends the top to make headroom for the
    in-panel legend.
    """
    fig, (axt, axb) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw=dict(height_ratios=[3, 1.4]),
        layout="constrained",
    )

    axt.fill_between(az, 0, base, color=FILL, alpha=0.6, lw=0)
    axt.plot(az, base, color="black", lw=1.1)
    axt.set_ylabel("Horizon Angle [deg]")
    axt.set_ylim(0, 40)

    for tag, lbl, c in SHIFTS:
        axb.plot(az, alpha[names.index(tag)] - base, color=c, lw=1.0, label=lbl)
    axb.axhline(0, color="0.6", lw=0.7, ls="--")
    axb.set_ylabel(r"$\Delta$ Horizon [deg]")
    axb.set_xlabel("Azimuthal Angle [deg]")
    axb.set_ylim(*resid_ylim)
    axb.legend(**legend_kw)

    for ax, tag in ((axt, "(a)"), (axb, "(b)")):
        ax.set_xlim(0, 360)
        ax.grid(alpha=0.3)
        ax.text(0.012, 0.93, tag, transform=ax.transAxes, va="top")
    return fig


def main():
    d = np.load(NPZ, allow_pickle=True)
    names = [str(n) for n in d["names"]]
    az = np.degrees(d["az_grid"])
    alpha = np.degrees(d["alpha_h"])
    base = alpha[names.index("nominal")]

    # Wide variant: in-panel 3-column legend.
    with plt.rc_context({"font.size": 10}):
        fig = build_figure(
            az,
            alpha,
            names,
            base,
            figsize=(6.5, 4.8),
            legend_kw=dict(
                ncol=3,
                loc="upper right",
                fontsize=8,
                columnspacing=1.0,
                handlelength=1.6,
            ),
        )
        fig.savefig(HERE / "horizon_perturbations.pdf")
        fig.savefig(HERE / "horizon_perturbations.png", dpi=200)
        plt.close(fig)

    # Single-column variant: smaller fonts; legend stays in panel (b) with
    # the residual top extended to make room above the curves.
    with plt.rc_context({"font.size": 8}):
        fig = build_figure(
            az,
            alpha,
            names,
            base,
            figsize=(3.4, 4.0),
            legend_kw=dict(
                ncol=3,
                loc="upper center",
                fontsize=6.5,
                columnspacing=0.8,
                handlelength=1.2,
                handletextpad=0.4,
            ),
            resid_ylim=(-2.1, 2.1),
        )
        fig.savefig(HERE / "horizon_perturbations_1col.pdf")
        fig.savefig(HERE / "horizon_perturbations_1col.png", dpi=200)
        plt.close(fig)

    print("wrote horizon_perturbations{,_1col}.{pdf,png}")


if __name__ == "__main__":
    main()
