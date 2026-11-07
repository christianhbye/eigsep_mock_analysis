"""Range-explicit monostatic adapter; existing physical kernels are reused.

PO is deterministic per physical draw. Diffuse fields are Gaussian random
measures on each chosen grid: compare expected powers/covariance across
refinements, never uncoupled complex realizations.
"""

import numpy as np
from eigsep_analysis import ground_reflect as gr
from eigsep_analysis import s11_diagnostics as sd


def frequency_grids(native_hz, step_hz, max_true_delay_s=0.0):
    """Separate regular fine and exact native working-band frequencies."""
    native = np.asarray(native_hz)
    native = native[(native >= 50e6) & (native <= 250e6)]
    if len(native) < 2:
        raise ValueError("no native working band")
    while max_true_delay_s >= 0.4 / step_hz:
        step_hz /= 2
    fine = np.arange(50e6, 250e6 + step_hz / 4, step_hz)
    return native.copy(), fine


def r0_normalization(nside, reference_nside=64):
    """Historical R0 raw sum to a common reference-pixel convention."""
    return (reference_nside / nside) ** 2


def r0_echo(freq_hz, points, antenna_enu, nside, chunk=4096):
    """Export raw historical phase sum and NSIDE64-normalized comparison."""
    ranges = np.linalg.norm(np.asarray(points) - antenna_enu, axis=1)
    # Explicit exact phase evaluation also supports stored rounded axes.
    raw = np.zeros(len(freq_hz), complex)
    f = np.asarray(freq_hz)
    for start in range(0, len(f), 128):
        fs = slice(start, start + 128)
        for i in range(0, len(ranges), chunk):
            r = ranges[i : i + chunk]
            raw[fs] += np.exp(-4j * np.pi * f[fs, None] * r / sd.C_LIGHT) @ (1 / r)
    return dict(
        raw=raw,
        reference64=raw * r0_normalization(nside),
        ranges_m=ranges,
        nside=int(nside),
        normalization=r0_normalization(nside),
    )


def terminal_taper(ranges_m, radius_m, fraction=0.1):
    """Explicit slant-domain taper; fraction0 is the hard boundary variant."""
    r = np.asarray(ranges_m)
    if fraction == 0:
        return (r <= radius_m).astype(float)
    return np.where(
        r <= radius_m, gr.slant_taper(r, radius_m * (1 - fraction), radius_m), 0
    )


def shell_index(ranges_m, edges_m):
    """First shell owns zero; subsequent shells are (left,right]."""
    r = np.asarray(ranges_m)
    edges = np.asarray(edges_m)
    idx = np.searchsorted(edges[1:], r, side="left")
    return np.where((r >= 0) & (r <= edges[-1]), idx, -1)


def _slice(arrays, pick):
    return {key: np.asarray(value)[pick] for key, value in arrays.items()}


def facet_spectra(
    freq_hz,
    cells,
    antenna_enu,
    beam,
    rotation,
    choices,
    radius_m,
    sigma_h_m,
    member=0,
    bin_deg=1.0,
    covariance=False,
):
    """PO, diffuse realization and expected diffuse signed power/covariance.

    No hidden190–220m taper. Common random numbers couple fine/native for
    this geometry only. Fresnel/roughness/polarization/facet formulas are
    ground_reflect's; incidence is quantized explicitly for its convergence
    ladder, or exact with bin_deg=None. Cell and frequency chunks bound RAM.
    """
    f = np.asarray(freq_hz, float)
    geom = gr.member_geometry(cells, antenna_enu)
    taper = terminal_taper(
        geom["range_m"], radius_m, choices["terminal_taper_fraction"]
    )
    shell = shell_index(geom["range_m"], choices["shell_edges_m"])
    ns = len(choices["shell_edges_m"]) - 1
    po = np.zeros((ns, len(f)), complex)
    diffuse = np.zeros_like(po)
    near_po = np.zeros(len(f), complex)
    near_diffuse = np.zeros(len(f), complex)
    diffuse_freq_power = np.zeros((ns, len(f)))
    delay_power = np.zeros_like(diffuse_freq_power) if covariance else None
    tau = sd.native_delays(len(f), (f[-1] - f[0]) / (len(f) - 1))
    target_tau = np.array([-1.3, -1.0, -0.8, -0.4, -0.2, 0, 0.6, 1, 1.3, 1.7]) * 1e-6
    ti = np.unique([np.argmin(abs(tau - t)) for t in target_tau])
    delay_cov = np.zeros((ns, len(ti), len(ti)), complex) if covariance else None
    taper_f = sd.make_taper("bh7", len(f))
    rng = np.random.default_rng([choices["seed"], member, 991])
    draws = rng.normal(size=(2, 2, len(shell))) / np.sqrt(2)
    phases = draws[:, 0] + 1j * draws[:, 1]
    # Diffuse covariance uses native FFT coefficients with an explicit bound
    # on stored-frequency roundoff; direct signed operator validation in tests.
    uniform = f[0] + np.arange(len(f)) * (f[-1] - f[0]) / (len(f) - 1)
    roundoff_bound = 2 * np.pi * np.max(abs(f - uniform)) * np.max(abs(tau))
    if covariance and roundoff_bound > 1e-6:
        raise ValueError("native roundoff exceeds declared covariance tolerance")
    cc = min(choices["cell_chunk"], 128 if covariance else 4096)
    for lo in range(0, len(shell), cc):
        sl = slice(lo, lo + cc)
        c = _slice(cells, sl)
        g = _slice(geom, sl)
        nc = len(g["range_m"])
        zs = phases[:, sl]
        dn, wn = gr.facet_beam(beam, rotation, c, antenna_enu)
        a_s = np.zeros((len(f), nc), complex) if covariance else None
        a_p = np.zeros_like(a_s) if covariance else None
        for start in range(0, len(f), choices["frequency_chunk"]):
            fs = slice(start, start + choices["frequency_chunk"])
            ff = f[fs]
            i0, t = gr.freq_weights(beam["f_hz"], ff)
            d = ((1 - t) * dn[:, i0] + t * dn[:, i0 + 1]).T
            ws = ((1 - t) * wn[:, i0] + t * wn[:, i0 + 1]).T
            cos = g["cos_inc"]
            if bin_deg is not None:
                bins, _, _ = gr.theta_bins(cos, bin_deg)
                cos = np.cos(np.radians((bins + 0.5) * bin_deg))
            eps = gr.eps_complex(
                ff, choices["material_eps"], choices["material_sigma_s_m"]
            )
            rs, rp = gr.fresnel_rs_rp(cos[None], eps[:, None])
            chi = gr.ament_chi(ff[:, None], sigma_h_m, cos[None])
            share = chi**2
            refl = gr.r_mono(rs, rp, ws)
            weight = d * refl * chi * taper[sl][None]
            phase = np.exp(-4j * np.pi * ff[:, None] * g["range_m"] / sd.C_LIGHT)
            fc = gr.facet_factor(
                2 * np.pi * ff[:, None] / sd.C_LIGHT,
                np.sqrt(c["area_m2"])[None],
                g["g_e"][None],
                g["g_n"][None],
            )
            coherent = 1j / (4 * np.pi) * weight * g["d_omega"] * fc * phase
            # Existing4b amplitudes: lambda/(4pi)^1.5 a D sqrt(alpha_q).
            base = (
                sd.C_LIGHT
                / ff[:, None]
                / (4 * np.pi) ** 1.5
                * gr.incoherent_amplitude(g)[None]
                * d
                * taper[sl][None]
            )
            qs = base * np.sqrt(ws * (1 - share)) * abs(rs) * phase
            qp = base * np.sqrt((1 - ws) * (1 - share)) * abs(rp) * phase
            if covariance:
                a_s[fs], a_p[fs] = qs, qp
            near = gr.slant_taper(g["range_m"])
            near_po[fs] += (coherent * near[None]).sum(axis=1)
            near_diffuse[fs] += (qs * near[None]) @ zs[0] + (qp * near[None]) @ zs[1]
            for k in range(ns):
                pick = shell[sl] == k
                if pick.any():
                    po[k, fs] += coherent[:, pick].sum(axis=1)
                    diffuse[k, fs] += (
                        qs[:, pick] @ zs[0, pick] + qp[:, pick] @ zs[1, pick]
                    )
                    diffuse_freq_power[k, fs] += (
                        abs(qs[:, pick]) ** 2 + abs(qp[:, pick]) ** 2
                    ).sum(axis=1)
        if covariance:
            # Different cells/polarizations are independent Gaussian measures.
            for a in [a_s, a_p]:
                x = np.fft.fftshift(np.fft.ifft(a * taper_f[:, None], axis=0), axes=0)
                x *= len(f) / taper_f.sum()
                for k in range(ns):
                    pick = shell[sl] == k
                    if pick.any():
                        delay_power[k] += (abs(x[:, pick]) ** 2).sum(axis=1)
                        q = x[ti][:, pick]
                        delay_cov[k] += q @ q.conj().T
    return dict(
        frequency_hz=f,
        po=po,
        diffuse=diffuse,
        legacy_near_po=near_po,
        legacy_near_diffuse=near_diffuse,
        diffuse_frequency_expected_power=diffuse_freq_power,
        diffuse_delay_expected_power=delay_power,
        diffuse_delay_covariance=delay_cov,
        covariance_tau_s=tau[ti],
        tau_s=tau,
        covariance_roundoff_operator_bound=float(roundoff_bound),
        shell_edges_m=np.asarray(choices["shell_edges_m"]),
        visible_cell_counts=np.bincount(
            shell[(shell >= 0) & (taper > 0)], minlength=ns
        ),
        zero_weight_cell_count=int(np.sum(taper == 0)),
        farthest_visible_range_m=float(geom["range_m"].max()),
        radius_m=radius_m,
        sigma_h_m=sigma_h_m,
    )


def closure(previous, current, predictive_spread, choices):
    """Report every window, last shell and MC floor; two radius steps needed."""
    a, b, spread = map(np.asarray, [previous, current, predictive_spread])
    change = abs(b - a)
    fractional = change / np.maximum(abs(b), 1e-300)
    pass_cells = (fractional < choices["closure_power_fraction"]) & (
        change < choices["closure_spread_fraction"] * spread
    )
    return dict(
        passed=bool(pass_cells.all()),
        change=change.tolist(),
        fractional=fractional.tolist(),
        threshold_spread=(choices["closure_spread_fraction"] * spread).tolist(),
        n_pass=int(pass_cells.sum()),
        n_total=int(pass_cells.size),
    )


def convergence_report(exports, choices):
    """Common-parameter refinements before any independent amplitude refit.

    PO complex changes and diffuse expected-power/covariance changes are
    different convergence questions. Raw physical units are retained.
    """
    import h5py

    records = {}
    for row in exports:
        with h5py.File(row["path"], "r") as h:
            f = h["frequency_hz"][:]
            po = h["po"][:].sum(axis=0)
            diff = h["diffuse_delay_expected_power"][:]
            cov = h["diffuse_delay_covariance"][:].sum(axis=0)
            tau = h["tau_s"][:]
            coeff = sd.signed_transform(f, po, taper=sd.make_taper("bh7", len(f))).coeff
            shells = h["diffuse_delay_expected_power"][:]
            active_last = min(
                int(
                    np.searchsorted(
                        np.asarray(choices["shell_edges_m"])[1:],
                        row["radius_m"],
                        side="left",
                    )
                ),
                len(shells) - 1,
            )
        windows = [
            (tau < -0.8e-6),
            (tau >= 1.2e-6),
            (tau > -0.4e-6) & (tau < 0),
            (tau >= -1.3e-6) & (tau < -0.7e-6),
            np.ones(len(tau), bool),
        ]
        key = (row["configuration"], row["epoch"])
        rec = records.setdefault(
            key,
            dict(
                parameters={k: row[k] for k in ("radius_m", "stride", "bin_deg")},
                members=[],
            ),
        )
        rec["members"].append(
            dict(
                member=row["member"],
                po=coeff,
                po_power=np.array([np.mean(abs(coeff[w]) ** 2) for w in windows]),
                diffuse=np.array([np.mean(diff.sum(axis=0)[w]) for w in windows]),
                last_shell=np.array([np.mean(shells[active_last][w]) for w in windows]),
                covariance=cov,
            )
        )
    comparisons = []
    for (ka, epoch), a in records.items():
        for (kb, epoch_b), b in records.items():
            if epoch != epoch_b or ka >= kb:
                continue
            varied = [
                k for k in a["parameters"] if a["parameters"][k] != b["parameters"][k]
            ]
            if len(varied) != 1:
                continue
            aa = sorted(a["members"], key=lambda r: r["member"])
            bb = sorted(b["members"], key=lambda r: r["member"])
            if [r["member"] for r in aa] != [r["member"] for r in bb]:
                raise ValueError("refinement physical draws do not match")
            row = dict(
                first=ka,
                second=kb,
                epoch=epoch,
                varied=varied[0],
                common_members=len(aa),
                normalization="same physical parameters; no refitted scale",
            )
            for quantity in ["po_power", "diffuse"]:
                av = np.array([r[quantity] for r in aa])
                bv = np.array([r[quantity] for r in bb])
                spread = bv.std(axis=0, ddof=1) if len(bb) > 1 else np.zeros(5)
                row[quantity] = closure(
                    av.mean(axis=0), bv.mean(axis=0), spread, choices
                )
                row[quantity]["parameter_mc_se"] = (spread / np.sqrt(len(bb))).tolist()
            ac, bc = np.array([r["po"] for r in aa]), np.array([r["po"] for r in bb])
            row["po_complex_rms_change"] = float(np.sqrt(np.mean(abs(bc - ac) ** 2)))
            row["po_complex_member_spread"] = float(
                np.sqrt(np.mean(np.var(bc, axis=0)))
            )
            row["expected_covariance_frobenius_change"] = float(
                np.linalg.norm(
                    np.mean([r["covariance"] for r in aa], axis=0)
                    - np.mean([r["covariance"] for r in bb], axis=0)
                )
            )
            comparisons.append(row)
    tails = [
        dict(
            configuration=key[0],
            epoch=key[1],
            parameters=r["parameters"],
            last_shell_mean=np.mean(
                [x["last_shell"] for x in r["members"]], axis=0
            ).tolist(),
        )
        for key, r in records.items()
    ]
    return dict(
        comparisons=comparisons,
        last_shells=tails,
        status="refinements reported; terminal-domain audit required"
        if comparisons
        else "domain/resolution limited: no independent refinement available",
        limitation=(
            "600m/AC5 is domain limited; uncoupled diffuse fields "
            "do not test coherent convergence"
        ),
    )
