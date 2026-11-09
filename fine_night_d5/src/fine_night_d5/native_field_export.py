"""Opt-in native harmonic-field scalar export, with native-grid closure.

This stage uses the fitted producer's existing DPSS/real-SH convention.
Both nativeL180 and modelL129 grids use their actual MWSS quadrature.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .assets import atomic_npz, sha256, write_json
from .native_field import harmonic_power


def load_basis(path):
    from eigsep_data.beam_mapping import DPSSSpectralBasis

    with np.load(path, allow_pickle=False) as z:
        return DPSSSpectralBasis(
            z["native_modes"],
            z["native_freqs"],
            z["grid_mhz"],
            z["dpss_modes"],
            z["concentrations"],
            float(z["half_bandwidth_cycles_per_sample"]),
            z["qr_r"],
            float(z["delay_halfwidth_ns"]),
        )


def grid(L, lmax):
    from eigsep_data.beam_mapping.beam_basis import real_spherical_harmonics
    from s2fft.utils.quadrature import quad_weights

    theta, phi = np.meshgrid(
        np.linspace(0, np.pi, L + 1), np.arange(2 * L) * np.pi / L, indexing="ij"
    )
    directions = np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    ).reshape(3, -1)
    weights = (
        np.broadcast_to(
            np.asarray(quad_weights(L=L, sampling="mwss"))[:, None], theta.shape
        )
        .ravel()
        .copy()
    )
    angular = real_spherical_harmonics(theta.ravel(), phi.ravel(), lmax)
    return angular, directions, weights


def run(prepared, fit, frequencies, out, fit_record_sha256, prior=False):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    source = json.loads((prepared / "RUN_DONE.json").read_text())
    for n, h in source["output_hashes"].items():
        if sha256(prepared / n) != h:
            raise ValueError("prepared native representation changed")
    with np.load(prepared / "representation.npz", allow_pickle=False) as z:
        lmax = int(z["lmax"])
        c0 = z["c0"]
    record_path = fit / "RUN_RECORD.json"
    if sha256(record_path) != fit_record_sha256:
        raise ValueError("fitted record identity changed")
    parent_record = json.loads(record_path.read_text())
    if parent_record["prepared_done_sha256"] != sha256(prepared / "RUN_DONE.json"):
        raise ValueError("fit was produced from a different native preparation")
    fit_record = json.loads((fit / "RUN_DONE.json").read_text())
    if not prior and not fit_record["fit_converged"]:
        raise ValueError("unconverged field cannot be exported")
    if fit_record["output_hashes"]["fit.npz"] != sha256(fit / "fit.npz"):
        raise ValueError("fitted coefficients changed")
    with np.load(fit / "fit.npz", allow_pickle=False) as z:
        coeff = c0 if prior else z["coeff"]
    f = np.asarray(frequencies, float)
    basis = load_basis(prepared / "spectral_basis.npz")
    if (
        f.ndim != 1
        or not len(f)
        or not np.isfinite(f).all()
        or np.any(np.diff(f) <= 0)
        or f.min() < basis.freqs.min()
        or f.max() > basis.freqs.max()
    ):
        raise ValueError("unique requested frequencies inside native support required")
    y, r, w = grid(129, lmax)
    yn, rn, wn = grid(180, lmax)
    maps, norms, errors = [], [], []
    for start in range(0, len(f), 4):
        at = f[start : start + 4]
        cf = basis.evaluate(at) @ coeff
        power, norm = harmonic_power(cf, y, r, w)
        _, native_norm = harmonic_power(cf, yn, rn, wn)
        np.testing.assert_allclose(norm, native_norm, rtol=1e-11, atol=1e-12)
        target = out / f"part_{start:04d}.npz"
        atomic_npz(
            target,
            frequencies_mhz=at,
            bm=power.reshape(-1, 130, 258),
            transverse_mean_power=norm,
            native_mean_power=native_norm,
        )
        maps.append(power.reshape(-1, 130, 258))
        norms.append(norm)
        errors.append(abs(norm / native_norm - 1))
    atomic_npz(
        out / "beam.npz",
        freqs=f * 1e6,
        bm=np.concatenate(maps),
        lmax=128,
        transverse_mean_power=np.concatenate(norms),
        normalization="unit MWSS spherical mean transverse power",
        frequency_scope="explicit requested coordinates inside fitted native band",
        parent_fit_done_sha256=sha256(fit / "RUN_DONE.json"),
        fit_record_sha256=fit_record_sha256,
        parent_prepared_done_sha256=sha256(prepared / "RUN_DONE.json"),
        role="represented_hfss_control" if prior else "raster_hybrid",
    )
    write_json(
        out / "RUN_DONE.json",
        dict(
            label="diagnostic; shape-only; not antenna efficiency",
            beam_sha256=sha256(out / "beam.npz"),
            fit_done_sha256=sha256(fit / "RUN_DONE.json"),
            fit_record_sha256=fit_record_sha256,
            prior_control=prior,
            prepared_done_sha256=sha256(prepared / "RUN_DONE.json"),
            exporter_sha256=sha256(__file__),
            frequencies_mhz=f.tolist(),
            native_vs_output_normalization_max_relative=float(
                np.max(np.concatenate(errors))
            ),
            native_L=180,
            output_L=129,
            field_lmax=lmax,
            analytic_power_bandlimit_upper_bound=2 * (lmax + 2),
            gain_renormalization=(
                "tooth gain times transverse_mean_power at same frequency"
            ),
            elapsed_s=time.monotonic() - started,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--fit", type=Path, required=True)
    p.add_argument("--frequencies", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fit-record-sha256", required=True)
    p.add_argument("--prior", action="store_true")
    a = p.parse_args()
    run(
        a.prepared,
        a.fit,
        json.loads(a.frequencies.read_text()),
        a.out,
        a.fit_record_sha256,
        a.prior,
    )
