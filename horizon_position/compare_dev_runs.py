"""One-off audit: what the croissant dev2 -> dev3 bump did to position_sims.npz.

Settled question, kept for the record. It backs one claim in the instrument
paper's regeneration notes -- that the croissant bump moves no number the
prose quotes -- and that claim is about a *filtered* residual, so the raw
difference does not answer it. This measures both.

The two runs differ only in croissant: `1384c8b` pinned v5.3.0.dev2, `b2824f5`
bumped to v5.3.0.dev3 (git rev 58bf56e, whose `croissant.__version__` still
reads 5.2.1 upstream). dev3 carries croissant #152, which takes the Euler
angles from the nearest true rotation and so slightly changes the Earth-frame
rotation behind the sky convolution.

INPUTS (both gitignored; `output/` is not in git):

  output/position_sims_phase1_pr18.npz  the PR #18 run, croissant dev2
      built 2026-09-14 from the feat/horizon-tilt-sensitivity branch, before
      it was squash-merged as `66946b8`; the npz stores no git sha
      sha256 c91d0880024541a4f97d05653ee7e4a2bbe84f71e6e18be807ecf150df7911a7

  output/position_sims.npz              the rerun on main, croissant dev3
      built 2026-09-15 from `36587e9` (main, with the run_beam_sims port and
      the compute_fgnd fix)
      sha256 30842c44f20e21b06bb70f1fe85bf01f82696c730a5283630925ec182c2fc337

Neither is reproducible cheaply -- each costs ~1 h of compute -- so the hashes
above, not the files, are the durable evidence. The script verifies them and
says so when they do not match.

CONVENTION, mirroring notebooks/horizon_shift.ipynb: the basis is the SVD of
the nominal row minus the receiver temperature (`t_ant = t_sys[0] - t_rcvr`),
a floor is the pooled RMS of the coefficients past mode N (`floor_in_basis`),
and N_ANCHOR = 10. The basis is built from the *new* run, because that is the
basis the regenerated numbers come from.

Usage (from the monorepo root):
    uv run python horizon_position/compare_dev_runs.py
"""

import hashlib
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEV2 = OUTPUT_DIR / "position_sims_phase1_pr18.npz"
DEV3 = OUTPUT_DIR / "position_sims.npz"
SHA256 = {
    DEV2.name: "c91d0880024541a4f97d05653ee7e4a2bbe84f71e6e18be807ecf150df7911a7",
    DEV3.name: "30842c44f20e21b06bb70f1fe85bf01f82696c730a5283630925ec182c2fc337",
}
N_ANCHOR = 10
# The two references the prose quotes at N_ANCHOR, both in mK.
FG_FLOOR_MK = 0.62
RETAINED_21CM_MK = 0.87


def check_hashes():
    """Verify both inputs are the runs this audit was written against."""
    for path in (DEV2, DEV3):
        if not path.exists():
            raise SystemExit(f"{path} not found; see this script's docstring")
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        ok = h == SHA256[path.name]
        print(f"{path.name:34s} sha256 {h[:16]}... {'OK' if ok else 'MISMATCH'}")
        if not ok:
            print(f"  expected {SHA256[path.name][:16]}...: this is a different run,")
            print("  so the numbers below are not the ones the notes quote")


def main():
    check_hashes()
    a = np.load(DEV2, allow_pickle=True)
    b = np.load(DEV3, allow_pickle=True)
    assert str(a["pos_sha"]) == str(b["pos_sha"]), "different position sets"

    print("\nARRAYS")
    for k in ("t_sys", "fgnd", "freqs_mhz", "times_jd"):
        same = np.array_equal(a[k], b[k])
        rel = np.max(np.abs(a[k] - b[k]) / np.maximum(np.abs(a[k]), 1e-30))
        print(f"  {k:10s} byte-equal={str(same):5s}  max rel diff={rel:.2e}")
    print("  fgnd must be byte-equal: no sky, and the zenith drive rotation is I")

    t_rcvr = float(b["t_receiver"])
    freqs = b["freqs_mhz"]
    n_f = freqs.size
    t_ant = b["t_sys"][0] - t_rcvr
    n_time = t_ant.shape[0]
    s, Vh = np.linalg.svd(t_ant, full_matrices=False)[1:]

    def floor_in_basis(mat, n):
        c = mat @ Vh.T
        return np.sqrt(np.sum(c[:, n:] ** 2) / mat.size)

    delta = a["t_sys"][0] - b["t_sys"][0]
    i = np.unravel_index(np.argmax(np.abs(delta)), delta.shape)
    print("\nRAW DIFFERENCE (nominal row)")
    print(
        f"  max |d| {np.abs(delta).max():.4f} K at {freqs[i[1]]:.0f} MHz"
        f"  ({a['t_sys'][0][i]:.3f} -> {b['t_sys'][0][i]:.3f} K)"
    )
    print(f"  RMS {np.sqrt((delta**2).mean()) * 1e3:.2f} mK")

    fg = np.array([np.sqrt(np.sum(s[N:] ** 2) / (n_time * n_f)) for N in range(19)])
    print("\nAFTER FILTERING (the quantity the mK numbers are)")
    print("   N   dev2-dev3 [mK]   fg floor [mK]   ratio")
    for N in (0, 5, 8, 9, N_ANCHOR, 12, 15, 18):
        d = floor_in_basis(delta, N) * 1e3
        print(f"  {N:2d}   {d:12.4f}   {fg[N] * 1e3:13.4f}   {d / (fg[N] * 1e3):6.3f}")

    d_anchor = floor_in_basis(delta, N_ANCHOR) * 1e3
    print(
        f"\n  at N={N_ANCHOR}: {d_anchor:.4f} mK"
        f" = {d_anchor / FG_FLOOR_MK:.3f}x the {FG_FLOOR_MK} mK foreground floor,"
        f" {d_anchor / RETAINED_21CM_MK:.3f}x the {RETAINED_21CM_MK} mK"
        " median retained signal"
    )

    print("\n  the same, for the rows section 4.7 quotes floors from:")
    for name in ("x_p_10", "z_p_1", "z_p_10"):
        j = [str(x) for x in a["names"]].index(name)
        dd = a["t_sys"][j] - b["t_sys"][j]
        print(
            f"    {name:8s} {floor_in_basis(dd, N_ANCHOR) * 1e3:.4f} mK"
            f"   (raw max {np.abs(dd).max():.3f} K)"
        )

    print(
        "\nSCOPE: this licenses transferring dev2-measured *filtered* values"
        " (residual\n  floors, induced floors) to dev3. Absolute quantities --"
        " a T_sys value, a\n  figure axis someone reads a number off -- still"
        f" move by up to {np.abs(delta).max():.2f} K."
    )


if __name__ == "__main__":
    main()
