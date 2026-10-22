"""S4 T6 diagnostic set: recovery, degeneracy and mimicry injections.

Plan ``docs/plans/2026-10-22-s4-night-trend-hunt.md`` Task 8 and section 4 in
christianhbye/eigsep_analysis_code (spec section 6.4 T6; D8, D13, D17, D18, D20).
Diagnostic only: it checks numerical recovery, uncertainty behaviour and
generic/null mimicry of the required-change fits; it identifies no cause.

Stages:

(i) sealed amplitude stage (D20): fits on the real data at the primary H0 k the
    required changes that set the truth amplitudes, the comparators and the
    degenerate directions; writes ``truths.npz`` and logs only its sha256;
(ii) per-draw units (set, truth, draw) with checkpoints keyed by the params,
    source, input and truth hashes;
(iii) summary ``injections.npz`` / ``injections.json``.

The analysis code is the frozen ``eigsep_analysis`` source named in
``s4_t6_source_manifest.json`` (commit and sha256 per file), loaded as a
private package namespace as ``s4_revision_pilot.py`` does; any hash mismatch
stops the run. ``--probe`` runs the solver-equivalence check and one draw per
group and writes ``gate.json``; the main run needs that gate. ``--smoke`` runs
two draws per truth on separate seeds; its outputs are not results.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import sys
import time
import types
from pathlib import Path

for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_key] = "1"

import numpy as np  # noqa: E402

#: Seed sets (plan section 4.2): spawn_key = (set, truth row, draw).
SEED_ENTROPY = 20261022
SET_TRUTH, SET_CAL, SET_EVAL, SET_SMOKE, SET_PROBE = 0, 1, 2, 8, 9
BATCH_SIZE = 10
SOURCE_FILES = ("__init__.py", "nighttrend.py", "skyground.py", "noise_waves.py")
MANIFEST = Path(__file__).with_name("s4_t6_source_manifest.json")
PACKAGE = "s4_t6_frozen_analysis"
T_CMB = 2.725

#: Input files and the keys read (notebook 023's product layout; the Part II
#: smoke run checks every one against the canonical S1 run).
S1_FILES = {
    "columns": "columns/columns_eigsim.npz",
    "gsm_maps": "columns/gsm_maps.npz",
    "fits": "fits.npz",
    "noise": "noise.npz",
    "scale": "scale.npz",
    "generic_hunt": "generic_hunt.npz",
}
TEAM_FILES = {
    "binned": "derived/ground_sky/binned_v0001/binned.npz",
    "fit_v0009": "derived/ground_sky/fit_v0009/fit.npz",
    "tcal": "derived/tcal/v0003/solutions.npz",
}
T_SW_DRIVER = "T_sw (switch board)"


# --- pure helpers (tested without the frozen source) -----------------------


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_for(set_id, truth, draw, entropy=SEED_ENTROPY):
    """SeedSequence(entropy, spawn_key=(set, truth, draw)) (plan section 4.2)."""
    return np.random.SeedSequence(entropy, spawn_key=(set_id, truth, draw))


def seed_batch(draw):
    """Evaluation draws fall in batches of ten for the seed-batch spread."""
    return draw // BATCH_SIZE


def block_signs(rng, blocks):
    """One independent +-1 per original 30-min block (ids >= 0)."""
    ids = np.unique(np.asarray(blocks)[np.asarray(blocks) >= 0])
    return {int(b): float(s) for b, s in zip(ids, rng.choice([-1.0, 1.0], ids.size))}


def apply_block_signs(residual, mask, blocks, signs):
    """Multiply every cell of a block, at all frequencies, by its sign.

    Times, sizes and masks are unchanged; a cell off ``mask`` stays NaN, so no
    missing cell becomes data (spec section 6.4 T6).
    """
    out = np.where(mask, np.asarray(residual, dtype=float), np.nan)
    blocks = np.asarray(blocks)
    for b, s in signs.items():
        out[:, blocks == b] *= s
    return out


def rebuild_data(base_model, truth, signed_residual, mask):
    """Full-night base + truth + signed residual on the existing cells."""
    return np.where(mask, base_model + truth + signed_residual, np.nan)


def weakest_direction(design, floor=1e-12):
    """The weakest right singular vector above zero of a design.

    Columns are scaled to unit norm; singular values with s / s_max <= floor
    count as exactly null and are skipped. Returns the coefficient vector in
    the design's own units and its s / s_max.
    """
    A = np.asarray(design, dtype=float)
    norm = np.linalg.norm(A, axis=0)
    d = np.where(norm > 0, 1.0 / np.where(norm > 0, norm, 1.0), 0.0)
    _, sv, vt = np.linalg.svd(A * d, full_matrices=False)
    rel = sv / sv[0]
    i = int(np.flatnonzero(rel > floor)[-1])
    return d * vt[i], float(rel[i])


def wilson(k, n, z=1.96):
    """Wilson score interval for k successes in n."""
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def confusion(records):
    """Rates per (truth, model) with numerators, denominators and Wilson."""
    tab = {}
    for r in records:
        cell = tab.setdefault(r["truth"], {}).setdefault(
            r["model"], {"successes": 0, "n": 0}
        )
        cell["n"] += 1
        cell["successes"] += int(bool(r["passes"]))
    for row in tab.values():
        for cell in row.values():
            cell["rate"] = cell["successes"] / cell["n"]
            cell["wilson95"] = wilson(cell["successes"], cell["n"])
    return tab


def order_stat_interval(n, q=0.95, conf=0.95):
    """Distribution-free interval for the q quantile: 1-based order statistics.

    Indices (l, u) with P(X_(l) <= x_q <= X_(u)) >= conf, as near the median
    rank n q as the binomial allows (n = 100: about the 90th to 99th draw).
    """
    from scipy.stats import binom

    best = None
    for lo in range(1, n + 1):
        for hi in range(lo + 1, n + 1):
            cov = binom.cdf(hi - 1, n, q) - binom.cdf(lo - 1, n, q)
            if cov >= conf and (best is None or hi - lo < best[1] - best[0]):
                best = (lo, hi)
                break
    return best


def log_event(log, **rec):
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **rec}
    with open(log, "a") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")


def sealed_stage(fit_truths, out_dir, log):
    """D20: run ``fit_truths`` and write ``truths.npz``; log only its hash.

    ``fit_truths()`` returns (arrays, params): name -> truth array, name ->
    true parameter vector. Nothing human-readable about them is written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays, params = fit_truths()
    payload = {f"truth__{k}": np.asarray(v) for k, v in arrays.items()}
    payload.update({f"params__{k}": np.asarray(v) for k, v in params.items()})
    path = out_dir / "truths.npz"
    tmp = out_dir / "truths.tmp.npz"
    np.savez(tmp, **payload)
    tmp.replace(path)
    digest = sha256(path)
    log_event(log, event="sealed", file=str(path), sha256=digest, n_truths=len(arrays))
    return digest


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_jsonable))
    tmp.replace(path)


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# --- frozen source and inputs ---------------------------------------------


def load_source(source_dir, manifest_path=MANIFEST):
    """Load the frozen analysis package after checking every file's sha256."""
    man = json.loads(Path(manifest_path).read_text())
    src = Path(source_dir)
    got = {f: sha256(src / f) for f in SOURCE_FILES}
    bad = {f: g for f, g in got.items() if g != man["files"][f]}
    if bad:
        raise RuntimeError(f"frozen source hash mismatch: {sorted(bad)}")
    pkg = types.ModuleType(PACKAGE)
    pkg.__path__ = [str(src.resolve())]
    sys.modules[PACKAGE] = pkg
    mods = {}
    for name in ("skyground", "noise_waves", "nighttrend"):
        spec = importlib.util.spec_from_file_location(
            f"{PACKAGE}.{name}", src / f"{name}.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods["nighttrend"], mods["skyground"], {"commit": man["commit"], **got}


def load_problem(nt, sg, s1, team, params):
    """The night problem at the primary H0 k from S1 and team products."""
    paths = {k: Path(s1) / v for k, v in S1_FILES.items()}
    paths.update({k: Path(team) / v for k, v in TEAM_FILES.items()})
    inputs = {k: {"path": str(p), "sha256": sha256(p)} for k, p in paths.items()}
    z = {k: dict(np.load(p, allow_pickle=False)) for k, p in paths.items()}
    b, f9, sol = z["binned"], z["fit_v0009"], z["tcal"]
    t, freqs = b["t"], f9["freqs_mhz"]
    fsel = np.array([int(np.argmin(np.abs(b["freqs_mhz"] - x))) for x in freqs])
    chan = b["channels"][fsel]
    y, sb = f9["data_k"], f9["sigma_k"]
    sigma = sg.sigma_tot(sb, y, z["noise"]["eps_w"])
    S = z["columns"]["S_gsm"][:, fsel].T
    f = z["columns"]["fgnd"][:, fsel].T
    base = np.stack([S + T_CMB * (1 - f), f, np.ones_like(S)], -1)
    era = b["era"].astype(str)
    masks = nt.masks_for(t, era, f9["fitted"], z["fits"]["e85"], y, sigma)
    if not np.array_equal(masks["fitted_ok"], z["fits"]["bins_used_fitted_ok"]):
        raise RuntimeError("fitted & ok differs from S1's bins_used_fitted_ok")
    k = params["h0"]["primary"]["k"]
    if k is None:
        raise RuntimeError("h0.primary.k is not filled (Task 1; D19, C39)")
    k = nt.as_k(np.asarray(k, float), freqs.size)
    use = np.flatnonzero(masks["fitted_ok"].any(0) | masks["e85_ok"].any(0))
    sweeps = {
        x: (sol[f"s11_{x}_t"], sol[f"s11_{x}_epoch"], sol[f"s11_{x}"])
        for x in ("ant", "amb", "rec")
    }
    cl = nt.c_load(t[use], sol["therm_t"], sol["therm_k"], sweeps, chan)
    full = {key: np.full(y.shape, np.nan) for key in ("c_l", "m_ant", "m_load")}
    for key in full:
        full[key][:, use] = cl[key]
    gam = {}
    for x, name in (("ant", "gamma_ant"), ("amb", "gamma_l"), ("rec", "gamma_rec")):
        g = np.full(y.shape, np.nan + 0j)
        g[:, use] = cl["gamma"][x].mean(axis=2).T
        gam[name] = g
    gns = params["h5"]["gamma_ns"]
    if gns is None:
        raise RuntimeError("h5.gamma_ns is not set (a Part II input; spec 6.3)")
    gns = np.asarray(gns["re"], float) + 1j * np.asarray(gns["im"], float)
    gns = gns[:, None] if gns.ndim == 1 else gns
    gam["gamma_ns"] = gns * np.ones(y.shape)
    t_l = np.full(t.size, np.nan)
    t_l[use] = cl["t_load"]
    names = [str(n) for n in z["generic_hunt"]["driver_names"]]
    t_sw = z["generic_hunt"]["drivers"][names.index(T_SW_DRIVER)]
    t_sw = t_sw[0] if t_sw.ndim == 2 else t_sw
    h0 = nt.h0_column(full["c_l"], k, masks["fitted_ok"])
    tr = masks["train_ok"]
    dy = sg.y_drift_delta(
        sol["non_t"],
        sol["non_p"],
        sol["amb_t"],
        sol["amb_p"],
        chan,
        t[use],
        t_l[use],
        full["m_load"][:, use],
        k,
        tr[:, use],
        on_gap="nan",
    )
    delta = np.full(y.shape, np.nan)
    delta[:, use] = dy["delta"]
    mk = z["gsm_maps"]
    return {
        "t": t,
        "freqs": freqs,
        "y": y,
        "sigma": sigma,
        "base": base,
        "S": S,
        "f": f,
        "masks": masks,
        "k": k,
        "h0": h0,
        "c_l": full["c_l"],
        "m_ant": full["m_ant"],
        "ybar": dy["ybar"],
        "t_l": t_l,
        "t_sw": t_sw,
        "delta_y": delta,
        "g_sw": z["scale"]["g_sw"],
        "g_bar": z["scale"]["gbar"],
        "w_bins": b["n_good"] if "n_good" in b else np.ones(t.size),
        "mckay": (mk["mckay_t_offset_k"][fsel], mk["mckay_f_scale"][fsel]),
        "t_out": params["h4"]["t_out_k"],
        "inputs": inputs,
        **gam,
    }


# --- stage (i): the sealed truths -------------------------------------------


def _rms(x, mask):
    return float(np.sqrt(np.mean(np.asarray(x)[mask] ** 2)))


def _gp(rng, t, freqs, ell_h, dex=0.3):
    ht = (t - t.mean()) / 3600.0
    Kt = np.exp(-0.5 * ((ht[:, None] - ht[None]) / ell_h) ** 2)
    lf = np.log10(freqs)
    Kf = np.exp(-0.5 * ((lf[:, None] - lf[None]) / dex) ** 2)
    Lt = np.linalg.cholesky(Kt + 1e-9 * np.eye(t.size))
    Lf = np.linalg.cholesky(Kf + 1e-9 * np.eye(freqs.size))
    return Lf @ rng.normal(size=(freqs.size, t.size)) @ Lt.T


def build_truths(nt, prob):
    """Fit the real data at the primary k; return (table, arrays, params).

    The table (plan section 4.1) lists 44 truths: name, group, own model, the
    models fitted per draw and the recovery quantity. Arrays are the injected
    columns on every bin; params the true parameter vectors.
    """
    y, sig, base, m, h0 = (prob[x] for x in ("y", "sigma", "base", "masks", "h0"))
    fo, tau = m["fitted_ok"], m["tau"]
    fB = nt.fit_shared(y, sig, base, None, fo, nt.PRIOR_MOD100, h0)
    sr = nt.slope_replicates(y, sig, base, tau, m, h0)
    ctx = {
        "freqs": prob["freqs"],
        "tau": tau,
        "t": prob["t"],
        "w_bins": prob["w_bins"],
        "support": fo.any(0),
        "k": prob["k"],
        "yhat": fB["model"],
        "c_l": prob["c_l"],
        "m_ant": prob["m_ant"],
        "ybar": prob["ybar"],
        "g_sw": prob["g_sw"],
        "g_bar": prob["g_bar"],
        "t_out": prob["t_out"],
        "t_sw": prob["t_sw"],
        "t_l": prob["t_l"],
        "delta_y": prob["delta_y"],
        "S": prob["S"],
        "f": prob["f"],
        "a_base": fB["a"],
        "fitted_ok": fo,
        **{g: prob[g] for g in ("gamma_ant", "gamma_l", "gamma_ns", "gamma_rec")},
    }
    first = nt.candidate_models(ctx)
    ranks = nt.measure_ranks(y, sig, base, first, m, h0)
    f10c = nt.fit_model(y, sig, base, first["10c_H5tau21"], fo, h0)
    tt = np.where(tau == 0, 1.0, tau)[None]
    pattern = np.einsum("fbp,p->fb", first["10c_H5tau21"]["shared"], f10c["p"]) / tt
    M = nt.candidate_models(ctx, ranks=ranks, h5_pattern=pattern)
    missing = M.pop("_missing")
    if missing:
        raise RuntimeError(f"roster incomplete: {missing}")
    g = sr["g"]
    trend = g[:, None] * tau[None]
    trend_rms = _rms(trend, fo)
    arrays, params, table, real_ranks = {}, {}, [], {}

    def add(name, group, own, models, arr, p=None, kind="params"):
        arrays[name] = np.where(np.isfinite(arr), arr, 0.0)
        params[name] = np.atleast_1d(np.asarray(p if p is not None else [], float))
        table.append(
            {"name": name, "group": group, "own": own, "models": models, "kind": kind}
        )

    cand = [n for n in M if n[:2] in _N_MODELS or n[:3] in _N_MODELS]
    cand = [n for n in cand if not n.startswith("_")]
    # N: null and generic truths
    add("null", "N", None, cand, np.zeros_like(y))
    add("g1tau", "N", None, cand, trend)
    r1p2 = {
        **M["2_G"],
        "extra": np.stack([tau, tau**2], -1)[None] * np.ones((y.shape[0], 1, 1)),
    }
    f2 = nt.fit_model(y, sig, base, r1p2, fo, h0)
    add("R1P2", "N", None, cand, np.einsum("fbq,fq->fb", r1p2["extra"], f2["a"][:, 3:]))
    for row, ell in zip(range(3, 6), (2.0, 3.0, 4.0)):
        rng = np.random.default_rng(seed_for(SET_TRUTH, row, 0))
        gp = _gp(rng, prob["t"], prob["freqs"], ell)
        add(f"gp{int(ell)}h", "N", None, cand, gp * trend_rms / _rms(gp, fo))
    # P: own-model recovery at the measured full-night amplitude
    for name in _P_MODELS:
        mod = M[name]
        fit = nt.fit_model(y, sig, base, mod, fo, h0)
        arr = np.einsum("fbp,p->fb", mod["shared"], fit["p"])
        if mod.get("fixed") is not None:
            arr = arr + mod["fixed"]
        add(name, "P", name, _own_models(name, M), arr, fit["p"])
        real_ranks[name] = int(fit["rank"])
    # degenerate companions (H4-sw with H0 and H5 sw3; each H5 design; H6)
    wid = fB["widths"]
    sw = M["9_H4sw"]
    joint = np.concatenate(
        [
            sw["shared"],
            (prob["c_l"] - np.nanmean(np.where(fo, prob["c_l"], np.nan), 1)[:, None])[
                ..., None
            ],
            M["11a_H5t_sw3"]["shared"],
        ],
        axis=-1,
    )
    joint = np.where(np.isfinite(joint), joint, 0.0)
    jmod = {
        **sw,
        "shared": joint,
        "p_names": ["alpha_sw", "dk_H0", "unc", "cos", "sin"],
    }
    M["9j_H4sw_joint"] = jmod
    v, _ = weakest_direction(nt.residualised_design(sig, base, joint, fo, wid))
    arr = np.einsum("fbp,p->fb", joint, v)
    s = trend_rms / _rms(arr, fo)
    add(
        "9_H4sw_degenerate",
        "P",
        "9j_H4sw_joint",
        _own_models("9j_H4sw_joint", M),
        arr * s,
        v * s,
        "degenerate",
    )
    for name in [n for n in M if n[:3] in ("10a", "10b", "10c", "11a", "11b", "11c")]:
        X = M[name]["shared"]
        v, _ = weakest_direction(nt.residualised_design(sig, base, X, fo, wid))
        arr = np.einsum("fbp,p->fb", X, v)
        s = trend_rms / _rms(arr, fo)
        add(
            f"{name}_degenerate",
            "P",
            name,
            _own_models(name, M),
            arr * s,
            v * s,
            "degenerate",
        )
    f12 = nt.fit_model(y, sig, base, M["12a_H6"], fo, h0)
    add(
        "12a_H6",
        "P",
        "12a_H6",
        _own_models("12a_H6", M),
        np.einsum("fbp,p->fb", M["12a_H6"]["shared"], f12["p"]),
        f12["p"],
    )
    real_ranks["12a_H6"] = int(f12["rank"])
    fg = np.zeros_like(y)
    n_near = 0
    for i in range(y.shape[0]):
        A = base[i, fo[i]][:, 1:] / sig[i, fo[i], None]
        vi, rel = weakest_direction(A)
        n_near += int(rel < 1e-3)
        fg[i] = vi[0] * base[i, :, 1] + vi[1]
    add(
        "12a_H6_degenerate",
        "P",
        "12a_H6",
        _own_models("12a_H6", M),
        fg * trend_rms / _rms(fg, fo),
        [0.0],
        "degenerate",
    )
    # its weak direction is the base's no-prior [f, 1] near-null, not in p
    table[-1]["flag_static"] = n_near
    for name in ("13_skylevel", "14a_skyslope", "15_antgain"):
        fit = nt.fit_model(y, sig, base, M[name], fo, h0)
        add(
            name,
            "P",
            name,
            _own_models(name, M),
            np.einsum("fbp,p->fb", M[name]["shared"], fit["p"]),
            fit["p"],
        )
        real_ranks[name] = int(fit["rank"])
    mk = nt.sky_variant_columns(
        prob["S"], prob["f"], *prob["mckay"], prob["freqs"], 1.0, 1.0
    )
    av = mk["available"]
    fmk = nt.fit_shared(
        y[av], sig[av], mk["base"][av], None, fo[av], nt.PRIOR_MOD100, h0[av]
    )
    diff = np.zeros_like(y)
    diff[av] = fmk["model"] - fB["model"][av]
    M["1_B_mckay"] = {**M["1_B"], "base_override": "mckay"}
    add(
        "mckay_change", "P", "1_B_mckay", ["1_B", "1_B_mckay"], diff, None, "prediction"
    )
    # R: declared reference amplitudes (labelled references)
    night_h = float(np.ptp(tau[fo.any(0)]))
    n7 = 7
    coef = np.zeros(3 * n7)
    coef[[0, n7, 2 * n7]] = 50.0 / night_h
    add(
        "ref_H5tau21_50K",
        "R",
        "10c_H5tau21",
        ["1_B", "2_G", "10c_H5tau21"],
        np.einsum("fbp,p->fb", M["10c_H5tau21"]["shared"], coef),
        coef,
    )
    for sgn, nm in ((1.0, "ref_H6_plus1"), (-1.0, "ref_H6_minus1")):
        add(
            nm,
            "R",
            "12a_H6",
            ["1_B", "2_G", "12a_H6"],
            sgn * M["12a_H6"]["shared"][..., 0],
            [sgn],
        )
    # C: the fitted same-freedom comparator per (r, h)
    for name in [n for n in M if M[n].get("is_comparator")]:
        fit = nt.fit_model(y, sig, base, M[name], fo, h0)
        rh = M[name]["is_comparator"]
        sharing = [
            n for n in M if not n.startswith("_") and M[n].get("comparator") == rh
        ]
        add(
            name,
            "C",
            name,
            sharing + [name],
            np.einsum("fbp,p->fb", M[name]["shared"], fit["p"]),
            fit["p"],
        )
    resid = np.where(fo, y - sr["fit"]["model"], np.nan)
    ctxinfo = {"ranks": ranks, "real_ranks": real_ranks, "trend_rms": trend_rms}
    return table, arrays, params, M, fB, resid, ctxinfo, mk


#: Models fitted on every N truth (plan section 4.1): 3-7, 8a, 9, 10a-c, 11a-c,
#: 12a, 13, 14a, 15, 20-28, with B and G.
_N_MODELS = (
    "1_",
    "2_",
    "3_",
    "4_",
    "5_",
    "6_",
    "7_",
    "8a",
    "9_",
    "10a",
    "10b",
    "10c",
    "11a",
    "11b",
    "11c",
    "12a",
    "13",
    "14a",
    "15",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
)
_P_MODELS = (
    "3_H1",
    "4_H1a",
    "5_H1b",
    "6_H3",
    "7_H3sw",
    "8a_H4out290",
    "9_H4sw",
    "10a_H5tau3",
    "10b_H5tau6",
    "10c_H5tau21",
    "11a_H5t_sw3",
    "11b_H5t_sw6",
    "11c_H5t_sw21",
)
_T4_OF = {
    "4_H1a": "16a_T4_H1a",
    "5_H1b": "16b_T4_H1b",
    "7_H3sw": "16c_T4_H3sw",
    "9_H4sw": "16d_T4_H4sw",
    "11c_H5t_sw21": "16e_T4_H5",
}


def _own_models(name, M):
    out = ["1_B", "2_G", name]
    comp = M.get(name, {}).get("comparator")
    if comp is not None:
        out += [n for n in M if M[n].get("is_comparator") == tuple(comp)]
    if name in _T4_OF and _T4_OF[name] in M:
        out.append(_T4_OF[name])
    return out


# --- stage (ii): one draw ----------------------------------------------------


def run_unit(nt, prob, ctx, unit):
    """Fit one draw's models; return a JSON-able record (no verdict)."""
    set_id, row, draw = unit
    T = ctx["table"][row]
    rng = np.random.default_rng(seed_for(set_id, row, draw))
    m, sig, h0 = prob["masks"], prob["sigma"], prob["h0"]
    fo = m["fitted_ok"]
    signs = block_signs(rng, m["blocks"])
    noise = apply_block_signs(ctx["resid"], fo, m["blocks"], signs)
    data = rebuild_data(ctx["yB"], ctx["arrays"][T["name"]], noise, fo)
    M, base = ctx["models"], prob["base"]
    rec = {"set": set_id, "truth": T["name"], "row": row, "draw": draw, "models": {}}
    t0 = time.process_time()
    sr = nt.slope_replicates(data, sig, base, m["tau"], m, h0)
    w = 1.0 / np.maximum(np.var(sr["reps"], axis=1) * (sr["reps"].shape[1] - 1), 1e-300)
    bat = {}
    for name in T["models"]:
        mod = M[name]
        b = base if mod.get("base_override") != "mckay" else ctx["mckay_base"]
        parts = ("score",)
        if name == T["own"] or name in _T4_OF.values():
            parts = ("full", "score") if name == T["own"] else ("full",)
        if T["group"] == "C":
            parts = ()
        if name.startswith("1_B_mckay"):
            av = ctx["mckay_avail"]
            fk = nt.fit_shared(
                data[av], sig[av], b[av], None, fo[av], nt.PRIOR_MOD100, h0[av]
            )
            fB = nt.fit_shared(data, sig, base, None, fo, nt.PRIOR_MOD100, h0)
            chg = fk["model"] - fB["model"][av]
            tru = ctx["arrays"][T["name"]][av]
            rec["models"][name] = {
                "rel_rms_change_error": float(
                    np.sqrt(np.mean((chg - tru)[fo[av]] ** 2))
                    / max(np.sqrt(np.mean(tru[fo[av]] ** 2)), 1e-300)
                ),
                "converged": bool(fk["converged"]),
            }
            continue
        out = {}
        if parts:
            bt = nt.battery(data, sig, b, mod, m, h0, parts=parts)
            bat[name] = bt
            if "full" in bt:
                fb = bt["full"]
                out.update(
                    p=fb["p"],
                    p_se=fb["p_se"],
                    rank=fb["rank"],
                    rank_sens=fb["rank_sens"],
                    sv=fb["sv"],
                    converged=fb["converged"],
                    ident_values=fb["identified"]["values"],
                    ident_var=fb["identified"]["var"],
                    ident_vectors=fb["identified"]["vectors"],
                    d=fb["identified"]["d"],
                )
        if mod.get("shared") is not None and not name.startswith("16"):
            X = mod["shared"]
            S_H = nt.filtered_slope(X, sig, b, m["tau"], fo, sr["widths"])
            gg = sr["g"]
            if mod.get("fixed") is not None:
                gg = gg - nt.filtered_slope(
                    mod["fixed"], sig, b, m["tau"], fo, sr["widths"]
                )
            comp = mod.get("comparator")
            cname = [
                n
                for n in M
                if comp is not None and M[n].get("is_comparator") == tuple(comp)
            ]
            if cname:
                S_C = nt.filtered_slope(
                    M[cname[0]]["shared"], sig, b, m["tau"], fo, sr["widths"]
                )
                dq = nt.delta_q(gg, S_H, S_C, w)
                out.update(delta_q=dq["delta_q"], dq_available=dq["available"])
            if "p" in out:
                pred = S_H @ np.atleast_1d(out["p"])
                tru = nt.filtered_slope(
                    ctx["arrays"][T["name"]], sig, base, m["tau"], fo, sr["widths"]
                )
                out["slope_rel_rms_error"] = float(
                    np.sqrt(np.mean((pred - tru) ** 2))
                    / max(np.sqrt(np.mean(tru**2)), 1e-300)
                )
        rec["models"][name] = out
    if "1_B" in bat and "2_G" in bat:
        for name, bt in bat.items():
            if name in ("1_B", "2_G") or "score" not in bt:
                continue
            cmp = nt.compare(bt, bat["1_B"], bat["2_G"], seed=int(rng.integers(2**31)))
            rec["models"][name].update(
                crit1=cmp["criterion1"]["passes"],
                R=cmp["R"]["passes"],
                rho=cmp["R"]["rho"],
                R_available=cmp["R"]["available"],
            )
    rec["cpu_s"] = time.process_time() - t0
    return rec


# --- stage (iii): summary ----------------------------------------------------


def summarise(records, table, params_truth, real_ranks, pre):
    """Recovery, degeneracy and mimicry tables (plan section 4.3), no verdicts."""
    by = {}
    for r in records:
        by.setdefault((r["set"], r["truth"]), []).append(r)
    out = {"recovery": {}, "mimicry": None, "delta_q_reference": {}, "labels": {}}
    # Delta-Q references from the calibration draws of each comparator truth
    for T in table:
        if T["group"] != "C":
            continue
        cal = by.get((SET_CAL, T["name"]), [])
        for model in T["models"]:
            vals = [r["models"].get(model, {}).get("delta_q") for r in cal]
            vals = np.sort([v for v in vals if v is not None])
            if vals.size == 0:
                continue
            lo_hi = order_stat_interval(vals.size) if vals.size >= 20 else None
            out["delta_q_reference"][model] = {
                "q95": float(np.quantile(vals, 0.95)),
                "n": int(vals.size),
                "order_stat_interval": None
                if lo_hi is None
                else [float(vals[lo_hi[0] - 1]), float(vals[lo_hi[1] - 1])],
                "label": "approximate",
            }
    mim = []
    for T in table:
        ev = by.get((SET_EVAL, T["name"]), [])
        if T["group"] == "N":
            for r in ev:
                for model, rec in r["models"].items():
                    ref = out["delta_q_reference"].get(model, {}).get("q95")
                    dq = rec.get("delta_q")
                    mim.append(
                        {
                            "truth": T["name"],
                            "model": model + ":crit1",
                            "passes": bool(rec.get("crit1")),
                        }
                    )
                    mim.append(
                        {
                            "truth": T["name"],
                            "model": model + ":R",
                            "passes": bool(rec.get("R")),
                        }
                    )
                    if ref is not None and dq is not None:
                        mim.append(
                            {
                                "truth": T["name"],
                                "model": model + ":dQ",
                                "passes": dq > ref,
                            }
                        )
        if T["group"] in ("P", "R") and ev:
            own = T["own"]
            recs = [r["models"].get(own, {}) for r in ev]
            truth = np.asarray(params_truth.get(T["name"], []), float)
            rec = _recovery(T, recs, truth, real_ranks.get(own), pre)
            # seed-batch spread: the same checks per batch of ten draws
            batches = {}
            for r in ev:
                batches.setdefault(seed_batch(r["draw"]), []).append(
                    r["models"].get(own, {})
                )
            rec["batches"] = {
                str(b): _recovery(T, rb, truth, real_ranks.get(own), pre).get(
                    "covered68"
                )
                for b, rb in sorted(batches.items())
            }
            if T["own"] in _T4_OF and truth.size:
                p4 = [r["models"].get(_T4_OF[own], {}).get("p") for r in ev]
                ratio = [np.atleast_1d(x)[1] / truth[0] for x in p4 if x is not None]
                sd = float(np.std(ratio, ddof=1)) if len(ratio) > 1 else None
                rec["t4"] = {
                    "sigma_pf_over_ppred": sd,
                    "testable": sd is not None and sd <= pre["t4_sigma_ratio_max"],
                }
            out["recovery"][T["name"]] = rec
            if T["group"] == "P":
                out["labels"][T["name"]] = {
                    "model": own,
                    "label": (
                        "required-change diagnostic"
                        if rec["passes"]
                        else "not testable"
                    ),
                }
    out["mimicry"] = confusion(mim) if mim else {}
    return out


def _recovery(T, recs, truth, real_rank, pre):
    n = len(recs)
    if T["kind"] == "prediction":
        err = [r.get("rel_rms_change_error", np.inf) for r in recs]
        ok = int(np.sum(np.asarray(err) <= pre["slope_rel_rms_max"]))
        return {
            "n": n,
            "rel_rms_change_error_median": float(np.median(err)),
            "within": ok,
            "passes": ok >= pre["coverage68_min"],
        }
    conv = all(r.get("converged", False) for r in recs)
    ranks = [r.get("rank") for r in recs]
    p = np.array([np.atleast_1d(r.get("p", np.nan)) for r in recs], float)
    se = np.array([np.atleast_1d(r.get("p_se", np.nan)) for r in recs], float)
    slope_err = [r.get("slope_rel_rms_error", np.nan) for r in recs]
    res = {
        "n": n,
        "converged_all": conv,
        "ranks": sorted(set(map(int, filter(None, ranks)))),
        "real_rank": real_rank,
    }
    full_rank = p.shape[1] > 0 and all(rk == p.shape[1] for rk in ranks)
    if T["kind"] == "degenerate" and T.get("flag_static") is not None:
        flagged = [T["flag_static"] > 0] * n
        cover, wrong = _ident_cover(recs, truth)
        if full_rank and truth.size == p.shape[1]:
            z = np.abs(p - truth) / se
            cover = int(np.sum(np.all(z <= 1, axis=1)))
            wrong = int(np.sum(np.any(z > 2, axis=1)))
        res.update(flagged=int(np.sum(flagged)), covered68=cover, confident_wrong=wrong)
        res["passes"] = bool(
            all(flagged)
            and cover >= pre["coverage68_min"]
            and wrong <= pre["degenerate_confident_wrong_max"]
        )
        return res
    if T["kind"] == "degenerate":
        flagged = [
            (r.get("rank", 0) < len(np.atleast_1d(r.get("p", []))))
            or (
                np.min(
                    np.asarray(r.get("sv", [1.0]))
                    / np.max(np.asarray(r.get("sv", [1.0])))
                )
                < 1e-3
            )
            for r in recs
        ]
        if truth.size and p.shape[1] == truth.size and full_rank:
            z = np.abs(p - truth) / se
            cover = int(np.sum(np.all(z <= 1, axis=1)))
            wrong = int(np.sum(np.any(z > 2, axis=1)))
        else:
            cover, wrong = _ident_cover(recs, truth)
        res.update(flagged=int(np.sum(flagged)), covered68=cover, confident_wrong=wrong)
        res["passes"] = bool(
            np.all(flagged)
            and cover >= pre["coverage68_min"]
            and wrong <= pre["degenerate_confident_wrong_max"]
        )
        return res
    if full_rank and truth.size == p.shape[1]:
        bias = np.mean(p - truth, axis=0)
        med = np.median(se, axis=0)
        cover = int(np.sum(np.all(np.abs(p - truth) <= se, axis=1)))
        ok_bias = bool(np.all(np.abs(bias) <= pre["bias_max_sigma"] * med))
        res.update(bias=bias, median_se=med, covered68=cover, bias_ok=ok_bias)
        res["passes"] = bool(
            ok_bias
            and cover >= pre["coverage68_min"]
            and conv
            and (real_rank is None or set(ranks) == {real_rank})
        )
    else:
        within = int(np.sum(np.asarray(slope_err) <= pre["slope_rel_rms_max"]))
        cover, wrong = _ident_cover(recs, truth)
        res.update(slope_within=within, covered68=cover, confident_wrong=wrong)
        res["passes"] = bool(
            conv and within >= pre["coverage68_min"] and cover >= pre["coverage68_min"]
        )
    res["slope_rel_rms_error_median"] = float(np.nanmedian(slope_err)) if n else None
    return res


def _ident_cover(recs, truth):
    cover = wrong = 0
    for r in recs:
        vec = np.asarray(r.get("ident_vectors", np.zeros((0, 0))))
        if vec.size == 0 or truth.size != vec.shape[1]:
            continue
        d = np.asarray(r["d"])
        tz = vec @ np.where(d > 0, truth / np.where(d > 0, d, 1), 0)
        z = np.abs(np.asarray(r["ident_values"]) - tz) / np.sqrt(r["ident_var"])
        cover += int(np.all(z <= 1))
        wrong += int(np.any(z > 2))
    return cover, wrong


# --- driver --------------------------------------------------------------------


def _units(table, n_eval, n_cal, eval_set, cal_set):
    units = []
    for row, T in enumerate(table):
        if T["group"] == "C":
            units += [(cal_set, row, d) for d in range(n_cal)]
        else:
            units += [(eval_set, row, d) for d in range(n_eval)]
    return units


_G = {}


def _work(unit):
    return run_unit(_G["nt"], _G["prob"], _G["ctx"], unit)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--s1", type=Path, required=True, help="canonical products/023_s1_night_fit"
    )
    ap.add_argument(
        "--team",
        type=Path,
        required=True,
        help="marjum-2026-07 directory holding derived/",
    )
    ap.add_argument(
        "--source", type=Path, required=True, help="frozen src/eigsep_analysis"
    )
    ap.add_argument(
        "--params", type=Path, required=True, help="scripts/029_s4_params.json"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--budget-seconds", type=float, default=None)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    start = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / "run.log"
    nt, sg, src = load_source(args.source)
    params = json.loads(args.params.read_text())
    hashes = {
        "params": sha256(args.params),
        "source": src,
        "manifest": sha256(MANIFEST),
        "producer": sha256(__file__),
    }
    prob = load_problem(nt, sg, args.s1, args.team, params)
    hashes["inputs"] = prob["inputs"]
    log_event(
        log,
        event="start",
        hashes=hashes,
        mode="probe" if args.probe else "smoke" if args.smoke else "main",
    )
    sealed = args.out / "sealed"
    built = {}

    def fit_truths():
        built["v"] = build_truths(nt, prob)
        _, arrays, params_t, *_ = built["v"]
        return arrays, params_t

    truth_sha = sealed_stage(fit_truths, sealed, log)
    table, arrays, params_t, M, fB, resid, info, mk = built["v"]
    ctx = {
        "table": table,
        "arrays": arrays,
        "models": M,
        "yB": fB["model"],
        "resid": resid,
        "mckay_base": mk["base"],
        "mckay_avail": mk["available"],
    }
    key = hashlib.sha256(
        json.dumps(
            {**hashes, "truths": truth_sha}, sort_keys=True, default=str
        ).encode()
    ).hexdigest()
    t6p = params["t6"]
    if args.probe:
        return _probe(nt, prob, ctx, M, args, log, key, t6p, start)
    gate = args.out / "gate.json"
    if not args.smoke:
        if not gate.exists():
            raise RuntimeError("no gate.json: run --probe first (plan Task 9.4-9.5)")
        g = json.loads(gate.read_text())
        if g["key"] != key or g["decision"] != "go":
            raise RuntimeError("gate.json does not match this run or is not 'go'")
    n_eval = t6p["smoke"]["n_evaluation"] if args.smoke else t6p["n_evaluation"]
    n_cal = t6p["smoke"]["n_calibration"] if args.smoke else t6p["n_calibration"]
    es, cs = (SET_SMOKE, SET_SMOKE) if args.smoke else (SET_EVAL, SET_CAL)
    units = _units(table, n_eval, n_cal, es, cs)
    udir = args.out / "units"
    udir.mkdir(exist_ok=True)
    todo, records = [], []
    for u in units:
        path = udir / f"{u[0]}_{u[1]:02d}_{u[2]:03d}.json"
        if path.exists():
            rec = json.loads(path.read_text())
            if rec.get("key") == key:
                records.append(rec)
                continue
        todo.append(u)
    _G.update(nt=nt, prob=prob, ctx=ctx)
    import multiprocessing as mp

    status = "done"
    with mp.get_context("fork").Pool(args.workers) as pool:
        for rec in pool.imap_unordered(_work, todo):
            rec["key"] = key
            write_json(
                udir / f"{rec['set']}_{rec['row']:02d}_{rec['draw']:03d}.json", rec
            )
            records.append(json.loads(json.dumps(rec, default=_jsonable)))
            if args.budget_seconds and time.monotonic() - start > args.budget_seconds:
                status = "budget"
                pool.terminate()
                break
    if status != "done":
        write_json(args.out / "RUN_FAILED", {"reason": status, "n_done": len(records)})
        log_event(log, event="failed", reason=status)
        return 2
    if args.smoke:
        records = [
            dict(r, set=SET_EVAL if table[r["row"]]["group"] != "C" else SET_CAL)
            for r in records
        ]
    summ = summarise(records, table, params_t, info["real_ranks"], t6p["precision"])
    summ.update(
        label="diagnostic",
        smoke=bool(args.smoke),
        key=key,
        hashes=hashes,
        truths_sha256=truth_sha,
        n_units=len(records),
        seeds={"entropy": SEED_ENTROPY, "eval_set": es, "cal_set": cs},
        limits=(
            "conditional on this night's residuals and the block-sign assumption (D17)"
        ),
    )
    write_json(args.out / "injections.json", summ)
    np.savez_compressed(
        args.out / "injections.npz",
        records=np.array(json.dumps(records, default=_jsonable)),
    )
    write_json(
        args.out / "RUN_DONE",
        {"n_units": len(records), "wall_s": time.monotonic() - start},
    )
    log_event(log, event="done", n_units=len(records))
    return 0


def _probe(nt, prob, ctx, M, args, log, key, t6p, start):
    """Solver equivalence (B, G, 3, 10c, 12a) and one draw per group."""
    y, sig, base, m, h0 = (prob[x] for x in ("y", "sigma", "base", "masks", "h0"))
    eq = {}
    for name in ("1_B", "2_G", "3_H1", "10c_H5tau21", "12a_H6"):
        b, fixed, prior = nt.model_inputs(M[name], base, h0)
        fit = nt.fit_shared(y, sig, b, M[name]["shared"], m["fitted_ok"], prior, fixed)
        ref = nt.joint_reference(
            y, sig, b, M[name]["shared"], m["fitted_ok"], fit["widths"], prior, fixed
        )
        eq[name] = nt.solver_equivalence(fit, ref)
    if not all(e["passed"] for e in eq.values()):
        write_json(
            args.out / "RUN_FAILED", {"reason": "solver equivalence", "equivalence": eq}
        )
        raise RuntimeError("profiled and joint solves disagree: abort")

    # one draw per group, of its costliest truth (most shared parameters)
    def cost(T):
        return sum(
            0 if M[n]["shared"] is None else M[n]["shared"].shape[-1]
            for n in T["models"]
        )

    pick = {}
    for row, T in enumerate(ctx["table"]):
        g = T["group"]
        if g not in pick or cost(T) > cost(ctx["table"][pick[g]]):
            pick[g] = row
    per_group, probe_truth = {}, {}
    for g, row in pick.items():
        t0 = time.process_time()
        run_unit(nt, prob, ctx, (SET_PROBE, row, 0))
        per_group[g] = time.process_time() - t0
        probe_truth[g] = ctx["table"][row]["name"]
    counts = {}
    for T in ctx["table"]:
        n = t6p["n_calibration"] if T["group"] == "C" else t6p["n_evaluation"]
        counts[T["group"]] = counts.get(T["group"], 0) + n
    P = sum(counts[g] * per_group[g] for g in counts)
    W = args.workers
    gate = {
        "key": key,
        "equivalence": eq,
        "t_draw_cpu_s": per_group,
        "probe_truths": probe_truth,
        "draws": counts,
        "projected_cpu_h": P / 3600.0,
        "workers": W,
        "budget_h": t6p["budget_factor"] * P / (3600.0 * W),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "wall_s": time.monotonic() - start,
        "decision": "go"
        if P / 3600.0 <= t6p["budget_stop_cpu_h"]
        else "stop: return to CHB (D17)",
    }
    path = args.out / "gate.json"
    if path.exists():
        raise RuntimeError("gate.json exists: it is written once")
    write_json(path, gate)
    log_event(
        log,
        event="gate",
        decision=gate["decision"],
        projected_cpu_h=gate["projected_cpu_h"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
