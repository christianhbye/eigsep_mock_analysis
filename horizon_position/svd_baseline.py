"""SVD of the baseline (unperturbed) zenith waterfall -> two paper figures.

The baseline case is the nominal antenna position from
``output/position_sims.npz`` (``t_sys[0]``): bowtie beam, GSM16 sky,
T_ground = 300 K, zenith pointing, realistic horizon. We ground-loss
correct it to a sky temperature, subtract the mean spectrum over time
(so the SVD is of the time-frequency *covariance*, in temperature
units), and take the singular values ``s_i``.

Two quantities, both per-channel RMS temperatures (mK):

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
MK = 1e3  # K -> mK


def load_baseline():
    """Ground-loss-corrected, mean-subtracted baseline waterfall (K)."""
    d = np.load(SIMS_FILE, allow_pickle=True)
    t_sys = d["t_sys"][0]  # (n_t, n_f), nominal position, zenith
    fgnd = d["fgnd"][0]  # (n_f,)
    t_gnd = float(d["t_ground"])
    t_rcvr = float(d["t_receiver"])
    freqs = d["freqs_mhz"]
    x_sky = glc(t_sys, fgnd, t_gnd, t_rcvr)  # sky temperature
    x = x_sky - x_sky.mean(axis=0, keepdims=True)  # remove mean spectrum
    return x, freqs


def fig_spectrum(rms_mk, path):
    """Per-channel RMS contributed by each singular mode."""
    fig, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    modes = np.arange(1, N_SHOW + 1)
    ax.semilogy(modes, rms_mk[:N_SHOW], "o-", color="C0", ms=4, lw=1.3)
    ax.set_xlabel("singular mode index")
    ax.set_ylabel("per-channel RMS [mK]")
    ax.set_title("Baseline foreground mode spectrum (zenith)")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.set_xlim(0, N_SHOW + 1)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_residual(resid_mk, path):
    """Residual foreground RMS after projecting out the leading N modes."""
    fig, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    n = np.arange(N_MAX + 1)
    ax.semilogy(n, resid_mk[: N_MAX + 1], "o-", color="C3", ms=4, lw=1.3)
    ax.axhline(1.0, color="0.5", ls="--", lw=1)
    ax.text(N_MAX, 1.0, " 1 mK", color="0.4", va="bottom", ha="right", fontsize=9)

    n_below = int(np.nonzero(resid_mk < 1.0)[0][0])
    ax.axvline(n_below, color="0.7", ls=":", lw=1, zorder=0)
    ax.annotate(
        f"{n_below} modes\n$\\to$ < 1 mK",
        xy=(n_below, 1.0),
        xytext=(n_below + 1.5, 30.0),
        color="0.3",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8),
    )

    ax.set_xlabel("foreground modes removed, $N$")
    ax.set_ylabel("residual RMS [mK]")
    ax.set_title("Foreground suppression vs modes filtered (zenith)")
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

    rms_mk = s / norm * MK  # per-mode per-channel RMS
    tail = np.concatenate([np.cumsum((s**2)[::-1])[::-1], [0.0]])  # tail[N]
    resid_mk = np.sqrt(tail / (n_t * n_f)) * MK  # residual after N modes

    FIG_DIR.mkdir(exist_ok=True)
    fig_spectrum(rms_mk, FIG_DIR / "fig_svd_spectrum.pdf")
    fig_residual(resid_mk, FIG_DIR / "fig_svd_residual.pdf")

    n_below = int(np.nonzero(resid_mk < 1.0)[0][0])
    print(f"baseline waterfall: {n_t} LST x {n_f} freq, {len(s)} modes")
    print(f"mode-1 per-channel RMS: {rms_mk[0] / MK:.1f} K")
    print(f"residual after 10 modes: {resid_mk[10]:.2f} mK")
    print(f"modes to reach < 1 mK residual: {n_below}")
    print(f"wrote {FIG_DIR / 'fig_svd_spectrum.pdf'}")
    print(f"wrote {FIG_DIR / 'fig_svd_residual.pdf'}")


if __name__ == "__main__":
    main()
