"""Bounded S4 pre-decision class pilot; scratch preview, never a result.

Run --help for explicit input paths. No downloads, sky synthesis or JAX
operations are called. The frozen S1 module is loaded for its established
small-matrix solvers; its import-time JAX configuration is recorded.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

for key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[key] = "1"

import numpy as np  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1", type=Path, required=True)
    parser.add_argument(
        "--team",
        type=Path,
        required=True,
        help="marjum-2026-07 directory containing derived/",
    )
    parser.add_argument("--skyground", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--cal-n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=1200.0,
        help="total pilot budget; 25 percent timing headroom",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="one real-data battery/equivalence check only",
    )
    args = parser.parse_args()
    if min(args.n, args.cal_n) < 1:
        parser.error("sample counts must be positive")
    start = time.monotonic()
    paths = {
        "columns": args.s1 / "columns/columns_eigsim.npz",
        "fits": args.s1 / "fits.npz",
        "noise": args.s1 / "noise.npz",
        "binned": args.team / "derived/ground_sky/binned_v0001/binned.npz",
        "f9": args.team / "derived/ground_sky/fit_v0009/fit.npz",
        "sol": args.team / "derived/tcal/v0003/solutions.npz",
    }
    manifest = {k: {"path": str(v), "sha256": sha(v)} for k, v in paths.items()}
    # A private namespace preserves relative imports (e.g. .ladder)
    # without executing package __init__ or touching an older installation.
    package_name = "s4_frozen_analysis"
    package = types.ModuleType(package_name)
    package.__path__ = [str(args.skyground.resolve().parent)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(
        package_name + ".skyground", args.skyground
    )
    sg = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sg
    spec.loader.exec_module(sg)
    dat = {k: dict(np.load(v, allow_pickle=False)) for k, v in paths.items()}
    b, f9, sol, ce = (dat[k] for k in ("binned", "f9", "sol", "columns"))
    t, f = b["t"], f9["freqs_mhz"]
    fs = np.array([np.argmin(abs(b["freqs_mhz"] - x)) for x in f])
    fitted = f9["fitted"]
    train = sg.team_split(t, fitted)
    test = fitted & ~train
    blocks = sg.block_ids(t)
    bids = np.unique(blocks[fitted])
    tbids = np.unique(blocks[test])
    tau = (t - t[train].mean()) / 3600
    y, sb = f9["data_k"], f9["sigma_k"]
    ok = sg.ok_mask(y, sb)
    sigma = np.sqrt(sb**2 + (dat["noise"]["eps_w"][:, None] * y) ** 2)
    nf, nt = y.shape
    s, fg = ce["S_gsm"][:, fs].T, ce["fgnd"][:, fs].T
    base = np.stack((s + 2.725 * (1 - fg), fg, np.ones_like(s)), -1)
    prior = ("theta_g", "theta", 290.0, 30.0)
    names = ["theta", "theta_g", "O"]

    def free(data, rows, extra=False):
        model, coeff = [], []
        for i in range(nf):
            a = base[i]
            nm = names
            if extra:
                a = np.column_stack((a, tau))
                nm = names + ["tau"]
            z = sg.fit_linear(
                a,
                data[i],
                sigma[i],
                rows,
                test,
                rows,
                ok[i],
                prior=prior,
                names=nm,
            )
            if not z["prior_converged"]:
                raise RuntimeError("free prior did not converge")
            model.append(z["model_full"])
            coeff.append(z["x_full"])
        return np.array(model), np.array(coeff)

    def tied(data, rows, col):
        # Profile each frequency after appending its prior pseudo-row.
        # Widths follow the changing theta, exactly as S1 _prior_fit.
        initial, ix = free(data, rows)
        theta = ix[:, 0].copy()
        for it in range(200):
            qrs = []
            xx, xy = 0.0, 0.0
            for i in range(nf):
                m = rows & ok[i]
                a = np.vstack(
                    (
                        base[i, m] / sigma[i, m, None],
                        np.array([-290.0, 1.0, 0.0]) / (30 * abs(theta[i])),
                    )
                )
                v = np.append(data[i, m] / sigma[i, m], 0.0)
                x = np.append(col[i, m] / sigma[i, m], 0.0)[:, None]
                q, _ = np.linalg.qr(a, mode="reduced")
                rx, rv = x - q @ (q.T @ x), v - q @ (q.T @ v)
                xx += float((rx.T @ rx).item())
                xy += float((rx.T @ rv).item())
                qrs.append((a, v, x))
            p = xy / xx
            coeff = np.array(
                [np.linalg.lstsq(a, v - x[:, 0] * p, rcond=1e-12)[0] for a, v, x in qrs]
            )
            new = coeff[:, 0]
            change = np.max(abs(new - theta) / np.maximum(abs(new), 1e-300))
            theta = new
            if change < 1e-6:
                break
        else:
            raise RuntimeError("tied prior did not converge")
        return np.einsum("fbc,fc->fb", base, coeff) + p * col, p, coeff

    mg, cg = free(y, fitted)
    mp, cp = free(y, fitted, True)
    # Independent equivalence control against canonical outputs.
    err = float(
        np.max(
            abs(cp - dat["fits"]["R1P1__x_full"])
            / (abs(dat["fits"]["R1P1__x_full"]) + 1e-12)
        )
    )
    if err > 1e-7:
        raise RuntimeError(f"canonical free-fit mismatch {err}")
    sweeps = {
        k: (sol[f"s11_{k}_t"], sol[f"s11_{k}_epoch"], sol[f"s11_{k}"])
        for k in ("ant", "amb", "rec")
    }
    cl = np.zeros_like(y)
    cl[:, fitted] = sg.scale_error_column(
        t[fitted], sol["therm_t"], sol["therm_k"], sweeps, b["channels"][fs]
    )["c_l"]

    def mean(x):
        return np.array([np.mean(x[i, fitted & ok[i]]) for i in range(nf)])

    sky_slope = []
    for i in range(nf):
        m = fitted & ok[i]
        sky_slope.append(
            np.linalg.lstsq(
                np.column_stack((np.ones(m.sum()), tau[m])),
                s[i, m],
                rcond=None,
            )[0][1]
            * cg[i, 0]
        )
    signatures = {
        "H1": -mean(mg - cl),
        "loss_ambient": mean(np.einsum("fbc,fc->fb", base[:, :, :2], cg[:, :2]))
        - cg[:, 0] * 290,
        "sky_slope": np.array(sky_slope),
        "sky_level": cg[:, 0] * mean(s),
        "control": mean(mg),
        "flat": np.ones(nf),
    }
    cols = {k: v[:, None] * tau for k, v in signatures.items()}
    residual = np.where(ok & fitted[None], y - mp, 0.0)
    band1 = (f >= 50) & (f <= 90)
    band2 = (f >= 109) & (f <= 165)
    # Primary pilot contrast only; high-band/crossing/model selection and
    # driver gates require the formal notebook battery.

    def slopes(data):
        _, cc = free(data, fitted, True)
        rep = []
        for bid in bids:
            _, cr = free(data, fitted & (blocks != bid), True)
            rep.append(cr[:, 3])
        return cc[:, 3], np.array(rep).T

    def spectral(g, reps, sig):
        nr = reps.shape[1]
        delta = reps - reps.mean(1, keepdims=True)
        se2 = (nr - 1) / nr * np.sum(delta**2, 1)
        w = 1 / np.maximum(se2, 1e-30)

        def ratio(a, mask):
            return np.sum(w[mask] * a[mask] * sig[mask]) / np.sum(
                w[mask] * sig[mask] ** 2
            )

        value = ratio(g, band1) - ratio(g, band2)
        rr = np.array(
            [ratio(reps[:, j], band1) - ratio(reps[:, j], band2) for j in range(nr)]
        )
        se = np.sqrt((nr - 1) / nr * np.sum((rr - rr.mean()) ** 2))
        return abs(value) / max(se, 1e-30)

    def battery(data):
        gm, _ = free(data, train)
        pm, _ = free(data, train, True)

        def score(model):
            r = np.where(ok & test[None], ((data - model) / sigma) ** 2, 0.0)
            return np.array([r[:, blocks == k].sum() for k in tbids])

        r0, r1 = score(gm), score(pm)
        denom = (r0 - r1).sum()
        g, reps = slopes(data)
        out = {}
        for key, col in cols.items():
            hm, p, coeff = tied(data, train, col)
            # A separate one-parameter full-joint augmented least-squares
            # check at converged widths catches profiling errors.
            if args.probe:
                rows, vals = [], []
                for i in range(nf):
                    mask = train & ok[i]
                    a = np.zeros((mask.sum() + 1, 3 * nf + 1))
                    a[:-1, 3 * i : 3 * i + 3] = base[i, mask] / sigma[i, mask, None]
                    a[:-1, -1] = col[i, mask] / sigma[i, mask]
                    a[-1, 3 * i : 3 * i + 3] = np.array([-290.0, 1.0, 0.0]) / (
                        30 * abs(coeff[i, 0])
                    )
                    rows.append(a)
                    vals.append(np.append(data[i, mask] / sigma[i, mask], 0.0))
                a, v = np.vstack(rows), np.concatenate(vals)
                scale = np.linalg.norm(a, axis=0)
                joint = np.linalg.lstsq(a / scale, v, rcond=1e-12)[0] / scale
                if not np.isclose(joint[-1], p, rtol=1e-5, atol=1e-8):
                    raise RuntimeError("profiled vs full joint disagreement")
            rh = score(hm)
            gain = r0 - rh
            # The target remains summed held-out chi-square. No block is
            # silently dropped. Nonpositive generic denominator is invalid.
            rho = float(gain.sum() / denom) if denom > 1.0 else None
            leave = [
                (gain.sum() - gain[j]) / (denom - (r0 - r1)[j])
                for j in range(len(gain))
                if denom - (r0 - r1)[j] > 1.0
            ]
            robust = len(leave) == len(gain) and min(leave) >= 0.5
            out[key] = dict(
                criterion1=bool(np.all(gain > 0)),
                rho=rho,
                replacement=bool(rho is not None and rho >= 0.8 and robust),
                min_leave_share=float(min(leave)) if leave else None,
                spectral_stat=float(spectral(g, reps, signatures[key])),
                parameter=float(p),
            )
        return out

    battery_start = time.monotonic()
    real = battery(y)
    first_battery_seconds = time.monotonic() - battery_start
    probe_seconds = time.monotonic() - start
    record = dict(
        label="diagnostic scratch preview",
        canonical_error=err,
        source_sha256=sha(__file__),
        skyground_sha256=sha(args.skyground),
        frozen_package_manifest={
            p.name: sha(p) for p in sorted(args.skyground.parent.glob("*.py"))
        },
        manifest=manifest,
        args=vars(args).copy(),
        versions={k: importlib.metadata.version(k) for k in ("numpy", "scipy", "jax")},
        probe_seconds=probe_seconds,
        real_preview=real,
        limitations=[
            "low/middle spectral contrast only",
            "criterion1 sign arm only (conservative)",
            "no driver 3/P/T3/T4/T5 gates; driver power unmeasured",
            "no H0 sensitivity, two-component or covariance variants",
            "small conditional wild-sign ensemble; cannot certify tails",
            "calibration does not certify familywise selection error",
        ],
    )
    record["args"] = {
        k: str(v) if isinstance(v, Path) else v for k, v in record["args"].items()
    }
    args.out.mkdir(parents=True, exist_ok=True)
    gate_only = False
    if not args.probe:
        # The numerical battery is unchanged from the passed probe. This
        # first measurement omits probe-only independent joint checks.
        gate_path = args.out / "GATE.json"
        elapsed = time.monotonic() - start
        if gate_path.exists():
            raise RuntimeError("pilot gate already exists; use a fresh run")
        chosen = None
        for count in range(min(args.n, args.cal_n, 8), 3, -1):
            remaining = 8 * (2 * count) * first_battery_seconds
            if elapsed + 1.25 * remaining <= args.budget_seconds:
                chosen = count
                break
        gate = dict(
            source_sha256=sha(__file__),
            first_battery_seconds=first_battery_seconds,
            elapsed_seconds=elapsed,
            budget_seconds=args.budget_seconds,
            remaining_headroom_factor=1.25,
            requested_n=args.n,
            requested_cal_n=args.cal_n,
            selected_n=chosen,
            selected_cal_n=chosen,
            number_of_truths=8,
            projected_remaining_seconds=(
                8 * 2 * chosen * first_battery_seconds if chosen is not None else None
            ),
            decision=(
                "balanced preview count selected"
                if chosen
                else "gate only: balanced 4+4 does not fit"
            ),
        )
        gate_path.write_text(json.dumps(gate, indent=2))
        record["gate"] = gate
        gate_only = chosen is None
        if chosen is not None:
            args.n = args.cal_n = chosen
    if not args.probe and not gate_only:
        rng = np.random.default_rng(args.seed)
        truths = {k: cols[k] * real[k]["parameter"] for k in cols if k != "flat"}
        truths["measured_generic"] = cp[:, 3, None] * tau
        # Nonlinear smooth generic uses the measured slope scale, but a
        # deterministic 3 h smooth time shape independent of thermal logs.
        smooth = np.sin(tau / 3.0) + 0.3 * (tau / 3.0) ** 2
        smooth /= np.std(smooth[fitted])
        truths["nonlinear_generic"] = cp[:, 3, None] * smooth * np.std(tau[fitted])
        truths["null"] = np.zeros_like(y)
        records = {}
        for name, truth in truths.items():
            rows = []
            for n in range(args.cal_n + args.n):
                if time.monotonic() - start >= args.budget_seconds:
                    record["budget_stopped"] = True
                    record["wall_seconds"] = time.monotonic() - start
                    (args.out / "summary.json").write_text(
                        json.dumps(record, indent=2, allow_nan=False)
                    )
                    print("Budget stop; incomplete draws, no power table")
                    return
                signs = rng.choice([-1.0, 1.0], len(bids))
                noise = residual.copy()
                for bid, sign in zip(bids, signs):
                    noise[:, blocks == bid] *= sign
                ans = battery(mg + truth + noise)
                rows.append(ans)
                # Atomic per-draw progress; resume must match source/input
                # hashes and a deterministic prefix (formal runner later).
                path = args.out / f"{name}_{n:03d}.json"
                path.write_text(json.dumps(ans, indent=2, allow_nan=False))
            records[name] = rows
        # True-signature contrast thresholds from independent calibration
        # draws, then evaluate each gate and the joint class proposal.
        thresholds = {
            k: float(
                np.quantile(
                    [r[k]["spectral_stat"] for r in records[k][: args.cal_n]],
                    0.95,
                )
            )
            for k in truths
            if k in cols
        }
        tables = {}
        for truth, rows in records.items():
            table = {}
            for model in thresholds:
                ev = rows[args.cal_n :]
                tests = {
                    "criterion1": [r[model]["criterion1"] for r in ev],
                    "replacement": [r[model]["replacement"] for r in ev],
                    "spectral": [
                        r[model]["spectral_stat"] <= thresholds[model] for r in ev
                    ],
                }
                tests["joint_partial_preview"] = [all(x) for x in zip(*tests.values())]
                rates = {}
                for test, flags in tests.items():
                    n, k = len(flags), int(sum(flags))
                    z, phat = 1.96, k / n
                    den = 1 + z * z / n
                    centre = (phat + z * z / (2 * n)) / den
                    half = (
                        z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / den
                    )
                    rates[test] = dict(
                        successes=k,
                        n=n,
                        rate=phat,
                        wilson95=[centre - half, centre + half],
                    )
                table[model] = rates
            tables[truth] = table
        record.update(thresholds=thresholds, power=tables)
    record["wall_seconds"] = time.monotonic() - start
    (args.out / "summary.json").write_text(
        json.dumps(record, indent=2, allow_nan=False)
    )
    print(
        json.dumps(
            {
                "summary": str(args.out / "summary.json"),
                "wall_seconds": record["wall_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
