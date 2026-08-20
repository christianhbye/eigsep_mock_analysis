# models_21cm/verify_ensemble.py
"""Validation gates for the generated ensemble.

    uv run --project models_21cm python models_21cm/verify_ensemble.py \
        models_21cm/output/zeus21_models.npz

Exits non-zero if any gate fails. Gate 1 and 2 together prove the
provenance header is sufficient rather than merely populated -- they are
the direct countermeasure to the problem this project exists to fix, so
treat them as a release gate for the npz.
"""

import json
import sys
from pathlib import Path

import generate
import numpy as np
import provenance
import selection as sel

PAPER = Path("/home/christian/Documents/research/papers/eigsep_instrument/notebooks")
# These are the PR #5 operating point, used here as a FIXED reference for the
# gates -- this file validates the ensemble, not the figure. Task 8 recomputes
# the figure's N_ANCHOR from the new ensemble and may move it; that does not
# make these stale, because these describe the point the gates were set at.
N_ANCHOR = 10  # modes filtered at the reference operating point
FG_FLOOR_MK = 0.62  # foreground residual at N = 10, from PR #5

failures = []


def gate(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def retained_rms_mK(T21_K, Vh, n_modes):
    """RMS retained after projecting out the leading `n_modes`, per model."""
    coeff = T21_K @ Vh.T
    n_f = T21_K.shape[1]
    return np.sqrt(np.sum(coeff[:, n_modes:] ** 2, axis=1) / n_f) * 1e3


def interpolation_budget(header, d, keep, Vh, params):
    """Gate 4: what the pb=3 -> pb=4 difference does to retained RMS.

    Expressed in induced retained-RMS error, not mK, because retained RMS
    is the only quantity the figure reports.
    """
    import zeus21

    user = zeus21.User_Parameters(precisionboost=4.0)
    cosmo_in = zeus21.Cosmo_Parameters_Input(**header["cosmo_params"])
    classy = zeus21.runclass(cosmo_in)
    cosmo = zeus21.Cosmo_Parameters(user, cosmo_in, classy)
    zeus21.Correlations(user, cosmo, classy)
    hmf = zeus21.HMF_interpolator(user, cosmo, classy)

    idx = np.random.default_rng(2).choice(np.flatnonzero(keep), size=20, replace=False)
    fine = []
    for i in idx:
        kwargs = {
            v["name"]: (
                10.0 ** params[i, j] if v["transform"] == "log10" else params[i, j]
            )
            for j, v in enumerate(header["varied"])
        }
        astro = zeus21.Astro_Parameters(user, cosmo, **header["astro_fixed"], **kwargs)
        coeff = zeus21.get_T21_coefficients(
            user, cosmo, classy, astro, hmf, zmin=header["zmin"]
        )
        fine.append(
            generate.interpolate_to_grid(
                coeff.zintegral, coeff.T21avg[None, :], d["freqs_MHz"]
            )[0]
        )
    fine = np.array(fine)
    rms_fine = retained_rms_mK(fine * 1e-3, Vh, N_ANCHOR)
    rms_prod = retained_rms_mK(d["T21_mK"][idx] * 1e-3, Vh, N_ANCHOR)
    err = np.abs(rms_fine - rms_prod)
    gate(
        "interpolation error well below the foreground floor",
        bool(err.max() < 0.1 * FG_FLOOR_MK),
        f"max induced retained-RMS error {err.max():.4f} mK vs floor {FG_FLOOR_MK} mK",
    )


def main(path):
    d = np.load(path, allow_pickle=False)
    header = json.loads(str(d["provenance"]))
    params, T21_mK = d["params"], d["T21_mK"]

    # --- Gate 1: header sufficiency, sampling half -----------------------
    # `kept_index` records which rows of the full draw survived, so this is
    # an exact comparison -- no tolerance, no fuzzy matching.
    rebuilt = provenance.rebuild_params(header)
    n_kept = params.shape[0]
    gate(
        "header rebuilds the Sobol draw",
        bool(np.array_equal(rebuilt[d["kept_index"]], params)),
        f"{n_kept} kept of {rebuilt.shape[0]} drawn",
    )

    # --- Gate 2: header sufficiency, physics half ------------------------
    import zeus21

    cfg_user = header["user_params"]
    cfg_cosmo = header["cosmo_params"]
    cfg_astro = header["astro_fixed"]
    user = zeus21.User_Parameters(precisionboost=cfg_user["precisionboost"])
    cosmo_in = zeus21.Cosmo_Parameters_Input(**cfg_cosmo)
    classy = zeus21.runclass(cosmo_in)
    cosmo = zeus21.Cosmo_Parameters(user, cosmo_in, classy)
    zeus21.Correlations(user, cosmo, classy)
    hmf = zeus21.HMF_interpolator(user, cosmo, classy)

    rng = np.random.default_rng(0)
    probe = rng.choice(n_kept, size=3, replace=False)
    worst = 0.0
    for i in probe:
        kwargs = {
            v["name"]: (
                10.0 ** params[i, j] if v["transform"] == "log10" else params[i, j]
            )
            for j, v in enumerate(header["varied"])
        }
        astro = zeus21.Astro_Parameters(user, cosmo, **cfg_astro, **kwargs)
        coeff = zeus21.get_T21_coefficients(
            user, cosmo, classy, astro, hmf, zmin=header["zmin"]
        )
        redone = generate.interpolate_to_grid(
            coeff.zintegral, coeff.T21avg[None, :], d["freqs_MHz"]
        )[0]
        worst = max(worst, float(np.max(np.abs(redone - T21_mK[i]))))
    gate(
        "header re-runs 3 models to machine precision",
        worst < 1e-9,
        f"max |diff| = {worst:.2e} mK",
    )

    # --- Gate 3: ensemble sanity -----------------------------------------
    gate("all T21 finite", bool(np.isfinite(T21_mK).all()))
    troughs = T21_mK.min(axis=1)
    gate(
        "trough depths physically plausible",
        bool(troughs.min() > -500.0 and troughs.max() < 0.0),
        f"[{troughs.min():.1f}, {troughs.max():.1f}] mK",
    )

    keep = sel.reionized(d["xHI"], d["z_xHI"])
    gate(
        "reionized models vanish at the top of the band",
        bool(np.abs(T21_mK[keep][:, -1]).max() < 1.0),
        f"max |T21(250 MHz)| = {np.abs(T21_mK[keep][:, -1]).max():.3f} mK",
    )
    print(
        f"       reionization cut keeps {keep.sum()}/{keep.size} "
        f"({100 * keep.mean():.1f}%)"
    )

    # --- Gate 4 + 5 need the foreground modes ----------------------------
    shift = np.load(PAPER / "horizon_shift.npz")
    Vh, freqs = shift["Vh"], shift["freqs_MHz"]
    if not np.array_equal(freqs, d["freqs_MHz"]):
        gate("paper grid matches the ensemble grid", False)
        return 1
    gate("paper grid matches the ensemble grid", True)

    rms = retained_rms_mK(T21_mK[keep] * 1e-3, Vh, N_ANCHOR)

    # --- Gate 4: interpolation error budget --------------------------------
    interpolation_budget(header, d, keep, Vh, params)

    # --- Gate 5: percentile convergence ----------------------------------
    print("\nPercentile convergence (retained RMS [mK] at N = 10):")
    print(f"{'n':>6}  {'p5':>8}  {'p50':>8}  {'p95':>8}")
    ref = np.percentile(rms, [5, 50, 95])
    for n in (256, 512, 1024, 2048, rms.size):
        sub = np.random.default_rng(1).choice(rms, size=min(n, rms.size), replace=False)
        p = np.percentile(sub, [5, 50, 95])
        print(f"{n:>6}  {p[0]:>8.3f}  {p[1]:>8.3f}  {p[2]:>8.3f}")
    drift = np.abs(
        np.percentile(
            np.random.default_rng(1).choice(
                rms, size=min(2048, rms.size), replace=False
            ),
            [5, 50, 95],
        )
        - ref
    )
    gate(
        "percentiles converged by 2048",
        bool(np.all(drift < 0.1 * ref)),
        f"drift {drift.round(4).tolist()} vs {ref.round(4).tolist()}",
    )

    print(f"\nForeground floor for reference: {FG_FLOOR_MK} mK at N = {N_ANCHOR}")
    print(f"Fraction of models above it: {100 * (rms > FG_FLOOR_MK).mean():.1f}%")

    if failures:
        print(f"\n{len(failures)} gate(s) failed: {failures}")
        return 1
    print("\nAll gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
