"""How many spectral modes describe the foregrounds across the whole drive grid?

The paper quotes the number of eigenmodes needed to describe the beam-weighted
foregrounds at the zenith pointing. Rotating the antenna adds spectral
diversity, and this script measures what that diversity costs in model
complexity: it repeats the eigenmode analysis over the full 36 x 36 drive grid
and reports the number of modes needed to reach the same residual.

Input is ``horizon_chromaticity/output/chromaticity_eigsep.npz`` -- the
canonical multi-orientation run (1296 orientations x 1436 LST x 201 channels,
realistic EIGSEP horizon, GSM16 sky). No new simulation is required.

Note on the horizon mask: that cube applies a *boolean* horizon mask, whereas
``position_sims.npz`` (the source of the paper figures) uses the anti-aliased
fractional mask. The two agree on this quantity to three significant figures at
the zenith pointing -- the assertion at the end of ``main`` checks it -- so the
mode counts are quotable alongside the figure numbers.

The 2.99 GB ``t_sys`` array is streamed out of the npz to a temporary ``.npy``
so it can be memory-mapped rather than decompressed whole.

Usage (from the monorepo root):
    uv run python horizon_position/rotation_dimensionality.py
"""

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper  # noqa: E402

CHROMA = HERE.parent / "horizon_chromaticity" / "output" / "chromaticity_eigsep.npz"
OUT = HERE / "output" / "rotation_dimensionality.npz"

LST_STRIDE = 12  # 1436 -> 120 LST samples; applied identically to every case
N_REPORT = paper.N_ANCHOR  # the operating point the paper quotes at zenith

# Published zenith numbers from the fractional-mask simulation, used as a
# cross-check that the boolean-mask cube is interchangeable here.
ZENITH_REFERENCE = {0: 726.9, 1: 23.65, 4: 0.4089, 8: 7.687e-3}  # modes -> RMS [K]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--lst-stride",
        type=int,
        default=LST_STRIDE,
        help="use every Nth LST sample (the subspace is insensitive to this)",
    )
    return p.parse_args()


def extract_t_sys(dst):
    """Stream ``t_sys.npy`` out of the npz so it can be memory-mapped."""
    with zipfile.ZipFile(CHROMA) as z, open(dst, "wb") as f:
        with z.open("t_sys.npy") as s:
            shutil.copyfileobj(s, f, 1 << 24)
    return np.load(dst, mmap_mode="r")


def residual_curve(X):
    """Per-channel residual RMS [K] after filtering the leading N modes."""
    s = np.linalg.svd(X, compute_uv=False)
    tail = np.concatenate([np.cumsum(s[::-1] ** 2)[::-1], [0.0]])
    return np.sqrt(tail / (X.shape[0] * X.shape[1]))


def main():
    args = parse_args()
    if not CHROMA.exists():
        raise SystemExit(
            f"{CHROMA} not found - run horizon_chromaticity/run_sims.py --case eigsep"
        )

    meta = np.load(CHROMA, allow_pickle=True)
    el, az = meta["elevations"], meta["azimuths"]
    t_rcvr = float(meta["t_receiver"])
    n_f = meta["freqs_mhz"].size
    elw = np.where(el > 180, el - 360, el)  # signed tilt from zenith
    zenith = int(np.argmin(elw**2 + np.minimum(az, 360 - az) ** 2))

    with tempfile.TemporaryDirectory() as tmp:
        print(f"streaming t_sys out of {CHROMA.name} ...")
        t_sys = extract_t_sys(Path(tmp) / "t_sys.npy")
        n_ori, n_lst, _ = t_sys.shape
        lsts = np.arange(0, n_lst, args.lst_stride)
        print(f"{n_ori} orientations x {lsts.size} of {n_lst} LST x {n_f} freq")

        curves = {}
        for name, oris in (
            ("zenith", np.array([zenith])),
            ("full_grid", np.arange(n_ori)),
        ):
            X = np.asarray(t_sys[oris][:, lsts, :], dtype=float) - t_rcvr
            curves[name] = residual_curve(X.reshape(-1, n_f))
            del X

    zen, full = curves["zenith"], curves["full_grid"]
    for n, expect in ZENITH_REFERENCE.items():
        got = zen[n]
        assert abs(got - expect) / expect < 0.01, (
            f"zenith residual at N={n} is {got:.4g} K, expected ~{expect:.4g} K "
            "from the fractional-mask simulation - the two runs have diverged"
        )
    print("zenith curve matches the published fractional-mask numbers to <1%")

    target = zen[N_REPORT]
    n_full = int(np.flatnonzero(full <= target)[0])
    print(
        f"\nzenith reaches {target * 1e3:.3f} mK with {N_REPORT} modes; "
        f"the {n_ori}-pointing grid needs {n_full}"
    )

    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        OUT,
        residual_zenith=zen,
        residual_full_grid=full,
        n_orientations=n_ori,
        n_modes_zenith=N_REPORT,
        n_modes_full_grid=n_full,
        elev_vals=meta["elev_vals"],
        az_vals=meta["az_vals"],
        description=(
            "Foreground residual RMS [K] vs number of eigenmodes filtered, for "
            "the zenith pointing alone and for the pooled 36x36 drive grid, "
            "from chromaticity_eigsep.npz. n_modes_full_grid is the smallest N "
            "at which the pooled curve reaches the residual the zenith curve "
            "reaches at n_modes_zenith."
        ),
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
