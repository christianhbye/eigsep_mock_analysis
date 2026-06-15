"""SVD of the baseline (unperturbed) zenith waterfall -> two paper figures.

The baseline case is the nominal antenna position from
``output/position_sims.npz`` (``t_sys[0]``): bowtie beam, GSM16 sky,
T_ground = 300 K, zenith pointing, realistic horizon. We ground-loss
correct it to a sky temperature and take the SVD of the *uncentered*
time-frequency waterfall (a second-moment matrix in temperature units).
The mean spectrum is kept -- it is the dominant foreground and must be
removed by any per-spectrum cleaning, so it counts as mode 1.

Two quantities, both per-channel RMS temperatures (K):

* ``rms_i = s_i / sqrt(n_t * n_f)`` -- the RMS each mode contributes to a
  single channel of a single spectrum (Fig. 1).
* ``resid(N) = sqrt(sum_{i>N} s_i^2 / (n_t * n_f))`` -- the residual RMS
  left after projecting out the leading ``N`` modes (Fig. 2). The two
  add in quadrature: ``resid(N)^2 = sum_{i>N} rms_i^2``.

Usage (from the monorepo root):
    uv run python horizon_position/svd_baseline.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import glc  # noqa: E402

HERE = Path(__file__).resolve().parent
SIMS_FILE = HERE / "output" / "position_sims.npz"
FIG_DIR = HERE / "notebooks"

N_SHOW = 25  # modes shown in the singular-value spectrum
N_MAX = 25  # max modes removed in the residual curve
ONE_MK = 1e-3  # 1 mK reference threshold, in K


def load_baseline():
    """Ground-loss-corrected baseline waterfall (K); mean kept (uncentered)."""
    d = np.load(SIMS_FILE, allow_pickle=True)
    t_sys = d["t_sys"][0]  # (n_t, n_f), nominal position, zenith
    fgnd = d["fgnd"][0]  # (n_f,)
    t_gnd = float(d["t_ground"])
    t_rcvr = float(d["t_receiver"])
    freqs = d["freqs_mhz"]
    x_sky = glc(t_sys, fgnd, t_gnd, t_rcvr)  # sky temperature
    return x_sky, freqs


def fig_spectrum(rms_k, path):
    """Per-channel RMS contributed by each singular mode."""
    fig, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    modes = np.arange(1, N_SHOW + 1)
    ax.semilogy(modes, rms_k[:N_SHOW], "o-", color="k", ms=4, lw=1.3)
    ax.set_xlabel("Singular mode index")
    ax.set_ylabel("Per-channel RMS [K]")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.set_xlim(0, N_SHOW + 1)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_residual(resid_k, path):
    """Residual foreground RMS after projecting out the leading N modes."""
    fig, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    n = np.arange(N_MAX + 1)
    ax.semilogy(n, resid_k[: N_MAX + 1], "o-", color="k", ms=4, lw=1.3)
    ax.axhline(ONE_MK, color="0.5", ls="--", lw=1)
    ax.text(N_MAX, ONE_MK, " 1 mK", color="0.4", va="bottom", ha="right", fontsize=9)

    n_below = int(np.nonzero(resid_k < ONE_MK)[0][0])
    ax.axvline(n_below, color="0.7", ls=":", lw=1, zorder=0)
    ax.annotate(
        f"{n_below} modes\n$\\to$ < 1 mK",
        xy=(n_below, ONE_MK),
        xytext=(n_below + 1.5, 30 * ONE_MK),
        color="0.3",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8),
    )

    ax.set_xlabel("Foreground modes removed, $N$")
    ax.set_ylabel("Residual RMS [K]")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.set_xlim(0, N_MAX)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    if not SIMS_FILE.exists():
        raise SystemExit(f"{SIMS_FILE} not found - run run_sims.py first")

    x, _ = load_baseline()
    n_t, n_f = x.shape
    s = np.linalg.svd(x, compute_uv=False)  # (n_f,)
    norm = np.sqrt(n_t * n_f)

    rms_k = s / norm  # per-mode per-channel RMS [K]
    tail = np.concatenate([np.cumsum((s**2)[::-1])[::-1], [0.0]])  # tail[N]
    resid_k = np.sqrt(tail / (n_t * n_f))  # residual after N modes [K]

    FIG_DIR.mkdir(exist_ok=True)
    fig_spectrum(rms_k, FIG_DIR / "fig_svd_spectrum.pdf")
    fig_residual(resid_k, FIG_DIR / "fig_svd_residual.pdf")

    n_below = int(np.nonzero(resid_k < ONE_MK)[0][0])
    print(f"baseline waterfall: {n_t} LST x {n_f} freq, {len(s)} modes")
    print(f"mode-1 per-channel RMS: {rms_k[0]:.1f} K")
    print(f"residual after 10 modes: {resid_k[10] * 1e3:.2f} mK")
    print(f"modes to reach < 1 mK residual: {n_below}")
    print(f"wrote {FIG_DIR / 'fig_svd_spectrum.pdf'}")
    print(f"wrote {FIG_DIR / 'fig_svd_residual.pdf'}")


if __name__ == "__main__":
    main()
