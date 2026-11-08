#!/usr/bin/env python
"""Build an opt-in v003 power candidate from the pinned dense complex authority.

Source preparation does not authorize this real-data SHT. Root first approves
a three-frequency --probe on private compute, then a measured full-run gate.
The old v002 generator and every old file/config/default remain unchanged.
"""

import argparse
import copy
import datetime
import importlib.metadata
import json
import os
import resource
import subprocess
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

# Reuse the old transform verbatim; leave that generator's numerical behavior.
import numpy as np  # noqa: E402
import s2fft  # noqa: E402
from eigsim.hfss_pin import (  # noqa: E402
    NATIVE_PIN,
    POWER_PIN,
    directivity,
    load_native_fields,
    sha256,
)
from make_bowtie_v002 import native_to_mwss  # noqa: E402
from scipy.constants import epsilon_0, mu_0  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe", action="store_true", help="only 50/150/250 MHz")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("commit the generator before recording a candidate")
    candidate = copy.deepcopy(json.loads(POWER_PIN.read_text()))
    if args.probe:
        candidate["artifact"] = "eigsep_bowtie_v003_1mhz_probe_mwss.npz"
        candidate["schema"]["bm_shape"][0] = 3
        candidate["schema"]["frequency_hz"]["step"] = 100000000
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / candidate["artifact"]
    manifest = output.with_suffix(".json")
    if output.exists() or manifest.exists():
        raise FileExistsError("use a fresh candidate directory; do not overwrite a run")
    start = time.perf_counter()
    arrays, parent = load_native_fields(args.native_source)
    loaded = time.perf_counter()
    indices = np.array([0, 100, 200]) if args.probe else np.arange(201)
    freqs = arrays["freqs_mhz"][indices] * 1e6
    L, lmax = int(arrays["L"]), 128
    weights = np.asarray(s2fft.utils.quadrature_jax.quad_weights(L=L, sampling="mwss"))
    maps, dropped, efficiency, seconds = [], [], [], []
    for i in indices:
        t = time.perf_counter()
        power = (abs(arrays["E_mV"][i]) ** 2).sum(axis=0)
        normalized, integral = directivity(power, weights)
        bm, lost = native_to_mwss(normalized[None], L, lmax)
        maps.append(bm[0])
        dropped.append(lost[0])
        efficiency.append(integral * 1e-6 / (2 * np.sqrt(mu_0 / epsilon_0)))
        seconds.append(time.perf_counter() - t)
        print(f"{freqs[len(maps) - 1] / 1e6:g} MHz: {seconds[-1]:.3f} s", flush=True)
    beam = np.stack(maps)
    w_out = np.asarray(
        s2fft.utils.quadrature_jax.quad_weights(L=lmax + 1, sampling="mwss")
    )
    norm = np.einsum("ftp,t->f", beam, w_out) / (4 * np.pi)
    provenance = dict(
        time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        generator_commit=commit,
        generator_dirty=False,
        uv_lock_sha256=sha256(repo / "uv.lock"),
        parent_path=str(args.native_source.resolve()),
        parent=parent,
        native_pin_sha256=sha256(NATIVE_PIN),
        reused_transform="eigsim/scripts/make_bowtie_v002.py:native_to_mwss",
        reused_transform_file_sha256=sha256(
            Path(__file__).with_name("make_bowtie_v002.py")
        ),
        packages={
            p: importlib.metadata.version(p)
            for p in ("numpy", "scipy", "jax", "jaxlib", "s2fft")
        },
        input_load_seconds=loaded - start,
        frequency_seconds=seconds,
        prewrite_seconds=time.perf_counter() - start,
        prewrite_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        probe=bool(args.probe),
        normalization_integral_over_4pi=norm.tolist(),
        harmonic_dropped_fraction=np.asarray(dropped).tolist(),
        minimum_power=float(beam.min()),
        realized_efficiency_label=(
            "diagnostic: original 1W excitation; absolute normalization "
            "not independently validated"
        ),
        physical_port_plane="unresolved Q-BK-06; no balun/coax transfer",
    )
    with output.open("xb") as stream:
        np.savez(
            stream,
            freqs=freqs,
            bm=beam,
            lmax=lmax,
            realized_efficiency=np.asarray(efficiency),
            harmonic_dropped_fraction=np.asarray(dropped),
            provenance_json=json.dumps(provenance),
            **candidate["metadata"],
        )
    candidate.update(
        sha256=sha256(output),
        bytes=output.stat().st_size,
        status="generated candidate; review/adoption pending",
        provenance=provenance,
        execution=dict(
            wall_seconds=time.perf_counter() - start,
            maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            stage="including artifact serialization and checksum",
        ),
    )
    with manifest.open("x") as stream:
        json.dump(candidate, stream, indent=2)
        stream.write("\n")
    print(json.dumps(candidate, indent=2), flush=True)


if __name__ == "__main__":
    main()
