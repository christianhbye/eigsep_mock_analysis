"""Figures from saved stage arrays; no sky generation or refitting."""

from pathlib import Path

import numpy as np


def comparison(
    geometry,
    data,
    model,
    out,
    title,
    caption,
    model_label="Compact + background",
    data_label="Map",
):
    """Shared-coordinate map/model/residual view in Galactic tangent offsets."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    out = Path(out)
    style = {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }
    with matplotlib.rc_context(style):
        fig, axes = plt.subplots(
            1, 3, figsize=(9.5, 3.5), sharex=True, sharey=True, layout="constrained"
        )
        lo = float(min(data.min(), model.min()))
        hi = float(max(data.max(), model.max()))
        if hi == lo:
            hi = lo + max(abs(lo), 1) * 1e-9
        norm = LogNorm(lo, hi) if lo > 0 else Normalize(lo, hi)
        residual = data - model
        bound = max(float(np.max(np.abs(residual))), np.finfo(float).eps)
        for j, (ax, values, name) in enumerate(
            zip(axes, [data, model, residual], [data_label, model_label, "Map − model"])
        ):
            image = ax.scatter(
                geometry["x_deg"],
                geometry["y_deg"],
                c=values,
                s=3,
                rasterized=True,
                cmap="RdBu_r" if j == 2 else "viridis",
                norm=Normalize(-bound, bound) if j == 2 else norm,
            )
            fig.colorbar(image, ax=ax, label="K_CMB", shrink=0.8, pad=0.03)
            ax.set(title=name, xlabel="East offset [deg]", aspect="equal")
        axes[0].set_ylabel("North offset [deg]")
        fig.suptitle(title + " (Galactic local offsets)", fontsize=10)
        fig.savefig(Path(str(out) + ".png"), dpi=145)
        fig.savefig(Path(str(out) + ".pdf"))
        Path(str(out) + ".txt").write_text(caption + "\n")
        plt.close(fig)


def encircled(radii_deg, fractions, out, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    with matplotlib.rc_context({"font.size": 8}):
        fig, ax = plt.subplots(figsize=(6.4, 3.2), layout="constrained")
        for label, values in fractions.items():
            ax.plot(radii_deg, values, label=label)
        ax.axhline(0.99, color="0.5", ls=":", label="99% of normalized profile")
        ax.set(
            xlabel="Angular radius [deg]",
            ylabel="Encircled profile flux fraction",
            title=title,
            ylim=(0, 1.03),
            xlim=(0, 5),
        )
        ax.legend(fontsize=8)
        fig.savefig(Path(str(out) + ".png"), dpi=145)
        fig.savefig(Path(str(out) + ".pdf"))
        Path(str(out) + ".txt").write_text(
            "Conditional nominal56arcmin Gaussian profiles in Galactic/RING pixels, "
            "normalized over5deg. The line checks numerical support, not a measured "
            "GSM source PSF. Horizontal99% line is the reviewed conservation check.\n"
        )
        plt.close(fig)
