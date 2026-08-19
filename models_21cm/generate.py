"""Generate the Zeus21 global 21 cm ensemble.

    uv run --project models_21cm python models_21cm/generate.py \
        --n-log2 12 --precisionboost 3 --seed 20260819 \
        --out models_21cm/output/zeus21_models.npz

Only this module imports zeus21; priors/selection/provenance stay pure so
their tests run in the main env.
"""

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import priors
import provenance
from scipy.interpolate import CubicSpline

NU21 = 1420.405751  # 21 cm rest frequency [MHz]
PAPER_FREQS = np.arange(50.0, 251.0, 1.0)  # matches foreground_svd.npz
ZMIN = 4.65  # deliberately inside the spline domain
ZMIN_CLASS = 4.5
# Every fixed keyword is passed EXPLICITLY rather than left to Zeus21's
# defaults. The header promises that someone holding only the npz can
# regenerate it; if a future Zeus21 changed a default, a header that only
# recorded the non-default flags would be silently wrong and the
# regeneration would produce different models. Spec requires this.
COSMO_FIXED = {
    # Planck 2018 (Aghanim et al.)
    "omegab": 0.0223828,
    "omegac": 0.1201075,
    "h_fid": 0.67810,
    "As": 2.100549e-09,
    "ns": 0.9660499,
    "tau_fid": 0.05430842,
    "kmax_CLASS": 500.0,
    "zmax_CLASS": 50.0,
    "zmin_CLASS": ZMIN_CLASS,
    "Flag_emulate_21cmfast": False,
    "USE_RELATIVE_VELOCITIES": True,  # required by the Pop III machinery
    "HMF_CHOICE": "ST",
}
ASTRO_FIXED = {
    "astromodel": 0,  # GALUMI-like
    "accretion_model": 0,  # exponential
    "betastar": -0.5,  # high-mass rollover; UVLF bright end
    "Mc": 3e11,
    "Emax_xray_norm": 2000,
    "Nalpha_lyA_II": 9690,  # BL05
    "Nalpha_lyA_III": 17900,
    "Mturn_fixed": None,  # Pop II turnover follows M_atom(z)
    "FLAG_MTURN_SHARP": False,
    "sigmaUV": 0.5,
    "USE_POPIII": True,
    "alphastar_III": 0,
    "betastar_III": 0,
    "dlog10epsstardz_III": 0.0,
    "alphaesc_III": -0.3,
    "alpha_xray_III": -1.0,
    "USE_LW_FEEDBACK": True,
    "A_vcb": 1.0,  # Cruz+2024 calibration, not a free parameter
    "beta_vcb": 1.8,
    "C0dust": 4.43,
    "C1dust": 1.99,
}
assert not set(ASTRO_FIXED) & set(priors.PARAM_NAMES), "fixed/varied collision"
PACKAGES = (
    "numpy",
    "scipy",
    "classy",
    "astropy",
    "mcfit",
    "powerbox",
    "pyfftw",
    "zeus21",
)

_CTX = {}


def interpolate_to_grid(z_native, T21_native, freqs_MHz):
    """Cubic-spline ``(n_model, n_z)`` in log z onto ``freqs_MHz``.

    Linear interpolation would leave curvature discontinuities at every
    node, and in an analysis about what survives projection onto a smooth
    basis those kinks are exactly the structure that survives.
    """
    x = np.log10(np.asarray(z_native, dtype=float))
    if not np.all(np.diff(x) > 0):
        raise ValueError("z_native must be strictly ascending")
    z_target = NU21 / np.asarray(freqs_MHz, dtype=float) - 1.0
    x_target = np.log10(z_target)
    if x_target.min() < x[0] or x_target.max() > x[-1]:
        raise ValueError(
            "target grid falls outside the native z range; this would "
            f"extrapolate (native z [{10 ** x[0]:.4f}, {10 ** x[-1]:.4f}], "
            f"target z [{z_target.min():.4f}, {z_target.max():.4f}])"
        )
    native = np.atleast_2d(np.asarray(T21_native, dtype=float))
    return np.array([CubicSpline(x, row)(x_target) for row in native])


def _setup(precisionboost):
    """Build the cosmology once; workers inherit it through fork."""
    import zeus21

    user = zeus21.User_Parameters(precisionboost=precisionboost)
    cosmo_in = zeus21.Cosmo_Parameters_Input(**COSMO_FIXED)
    classy = zeus21.runclass(cosmo_in)
    cosmo = zeus21.Cosmo_Parameters(user, cosmo_in, classy)
    # Populates ClassCosmo.pars['xi_RR_CF']; Pop III fails without it.
    zeus21.Correlations(user, cosmo, classy)
    hmf = zeus21.HMF_interpolator(user, cosmo, classy)
    _CTX.update(zeus21=zeus21, user=user, cosmo=cosmo, classy=classy, hmf=hmf)


def _run_one(item):
    """Return ``(index, (z, T21, xHI), error)``; error is None on success."""
    index, row = item
    z21 = _CTX["zeus21"]
    try:
        astro = z21.Astro_Parameters(
            _CTX["user"],
            _CTX["cosmo"],
            **ASTRO_FIXED,
            **priors.to_astro_kwargs(row),
        )
        coeff = z21.get_T21_coefficients(
            _CTX["user"],
            _CTX["cosmo"],
            _CTX["classy"],
            astro,
            _CTX["hmf"],
            zmin=ZMIN,
        )
        z = np.asarray(coeff.zintegral, dtype=float)
        t21 = np.asarray(coeff.T21avg, dtype=float)
        xhi = np.asarray(coeff.xHI_avg, dtype=float)
    except Exception as exc:  # noqa: BLE001 - recorded, not raised
        return index, None, f"{type(exc).__name__}: {exc}"
    if not (np.isfinite(t21).all() and np.isfinite(xhi).all()):
        return index, None, "non-finite output"
    return index, (z, t21, xhi), None


def _batch_path(out_dir, start):
    return out_dir / f"batch_{start:05d}.npz"


def _run_batches(params, out_dir, batch_size, processes):
    """Run every batch, checkpointing each, and return per-model results."""
    ctx = mp.get_context("fork")  # workers inherit the built cosmology
    out_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(params), batch_size):
        path = _batch_path(out_dir, start)
        if path.exists():
            print(f"skip {path.name} (already done)", flush=True)
            continue
        items = [
            (i, params[i]) for i in range(start, min(start + batch_size, len(params)))
        ]
        with ctx.Pool(processes=processes) as pool:
            results = pool.map(_run_one, items)
        idx = np.array([r[0] for r in results])
        ok = np.array([r[1] is not None for r in results])
        good = [r[1] for r in results if r[1] is not None]
        np.savez_compressed(
            path,
            index=idx,
            ok=ok,
            z=good[0][0] if good else np.zeros(0),
            T21=np.array([g[1] for g in good]) if good else np.zeros((0, 0)),
            xHI=np.array([g[2] for g in good]) if good else np.zeros((0, 0)),
            errors=np.array([r[2] or "" for r in results]),
        )
        print(f"wrote {path.name}: {ok.sum()}/{len(results)} ok", flush=True)

    index, ok, t21, xhi, errors, z = [], [], [], [], [], None
    for start in range(0, len(params), batch_size):
        d = np.load(_batch_path(out_dir, start), allow_pickle=False)
        index.append(d["index"])
        ok.append(d["ok"])
        errors.append(d["errors"])
        if d["T21"].size:
            t21.append(d["T21"])
            xhi.append(d["xHI"])
            z = d["z"]
    return (
        np.concatenate(index),
        np.concatenate(ok),
        np.vstack(t21),
        np.vstack(xhi),
        np.concatenate(errors),
        z,
    )


def _read_text(path):
    return Path(path).read_text() if Path(path).exists() else ""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-log2", type=int, default=12, help="Sobol m; 2**m models")
    p.add_argument("--precisionboost", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=20260819)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--processes", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args(argv)

    here = Path(__file__).resolve().parent
    params = priors.sample(m=args.n_log2, seed=args.seed)
    print(f"{len(params)} models, precisionboost={args.precisionboost}", flush=True)

    _setup(args.precisionboost)
    work_dir = args.out.parent / f"batches_m{args.n_log2}_pb{args.precisionboost:g}"
    index, ok, t21_native, xhi, errors, z_native = _run_batches(
        params, work_dir, args.batch_size, args.processes
    )

    order = np.argsort(index)
    ok = ok[order]
    kept_index = index[order][ok]  # rows of the Sobol draw that survived
    kept = params[kept_index]
    n_failed = int((~ok).sum())
    if n_failed:
        print(f"dropped {n_failed} failed draws", flush=True)

    t21_grid = interpolate_to_grid(z_native, t21_native, PAPER_FREQS)

    header = provenance.build_header(
        user_params={"precisionboost": args.precisionboost},
        cosmo_params=dict(COSMO_FIXED),
        astro_fixed=dict(ASTRO_FIXED),
        zmin=ZMIN,
        sampler={
            "kind": "sobol",
            "scramble": True,
            "seed": args.seed,
            "m": args.n_log2,
            "n_models": int(len(params)),
            "n_kept": int(ok.sum()),
        },
        varied=[
            {"name": q.name, "transform": q.transform, "lo": q.lo, "hi": q.hi}
            for q in priors.PARAMS
        ],
        interpolation={
            "method": "scipy CubicSpline",
            "variable": "log10(z)",
            "target": "50-250 MHz at 1 MHz, 201 points",
        },
        selection={
            "applied": False,
            "recommended": "xHI(z=5.9) < 0.1",
            "reference": "McGreer et al. 2015 dark-pixel limit",
        },
        packages=provenance.package_versions(PACKAGES),
        code={"generator": provenance.git_info(here), "zeus21_pin": "see env_lock"},
        failures={"n_failed": n_failed, "messages": sorted({e for e in errors if e})},
        command_line=[str(a) for a in (sys.argv if argv is None else argv)],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        freqs_MHz=PAPER_FREQS,
        T21_mK=t21_grid,
        T21_native_mK=t21_native,
        z_native=z_native,
        xHI=xhi,
        z_xHI=z_native,
        params=kept,
        kept_index=kept_index,
        param_names=np.array(priors.PARAM_NAMES),
        provenance=np.array(header),
        generator_source=np.array(
            "\n\n# ---- priors.py ----\n"
            + _read_text(here / "priors.py")
            + "\n\n# ---- generate.py ----\n"
            + _read_text(here / "generate.py")
        ),
        env_lock=np.array(_read_text(here / "uv.lock")),
        regenerate_recipe=np.array(REGENERATE_RECIPE),
    )
    print(f"wrote {args.out} ({t21_grid.shape[0]} models)", flush=True)
    return args.out


REGENERATE_RECIPE = """\
To regenerate this file with no access to the originating repository:

1. Read the `provenance` key (plain JSON, no pickle needed):
       import numpy as np, json
       d = np.load("zeus21_models.npz")
       header = json.loads(str(d["provenance"]))
2. Recreate the environment from the `env_lock` key -- it is a uv lockfile
   pinning Zeus21 by commit along with classy, numpy and scipy.
3. Write out the `generator_source` key; it contains priors.py and
   generate.py verbatim as run.
4. Re-run the command in header["command_line"].

`params` holds the sampled values in TRANSFORMED units: header["varied"]
gives each column's name, transform ("log10" or "linear") and bounds, in
column order. A log10 column means Zeus21 received 10**value.

The reionization cut in header["selection"] has NOT been applied to the
stored arrays. Apply it with the stored `xHI` / `z_xHI`.
"""


if __name__ == "__main__":
    main()
