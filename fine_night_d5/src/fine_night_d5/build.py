"""Opt-in scalar beam builder; requires a separately approved real run.

Example: python -m fine_night_d5.build --native INPUT --sha256 HASH
--symmetry body-x --out NEW_STAGE. It never writes eigsim/data or defaults.
Per-frequency arrays/QC are atomically checkpointed before final assembly.
"""

import argparse
import hashlib
import importlib.metadata
import json
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import beam


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024**2), b""):
            h.update(part)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--symmetry", required=True, choices=("raw", "body-x"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        if sha256(args.native) != args.sha256:
            raise ValueError("native complex source checksum differs")
        import s2fft

        with np.load(args.native, allow_pickle=False) as z:
            f, e, L = z["freqs_mhz"], z["E_mV"], int(z["L"])
            if (
                str(z["sampling"]) != "mwss"
                or e.dtype != np.complex128
                or e.shape != (len(f), 3, L + 1, 2 * L)
                or not np.isfinite(e).all()
                or not np.array_equal(z["theta_deg"], np.linspace(0, 180, L + 1))
                or not np.array_equal(z["phi_deg"], np.arange(2 * L) * 180 / L)
            ):
                raise ValueError("native complex coordinates/schema differ")
        if f.ndim != 1 or not np.isfinite(f).all() or np.any(np.diff(f) <= 0):
            raise ValueError("invalid native frequency axis")
        runtime = {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "jax", "jaxlib", "s2fft")
        }
        record = {
            "label": "diagnostic",
            "source_sha256": args.sha256,
            "symmetry": args.symmetry,
            "lmax": 128,
            "normalization": "native MWSS integral4pi before scalar SHT",
            "frequency_interpolation": "none; native slices",
            "python": sys.version,
            "runtime": runtime,
            "beam_module_sha256": sha256(beam.__file__),
            "builder_sha256": sha256(__file__),
        }
        atomic_json(args.out / "RUN_START.json", record)
        w = np.asarray(s2fft.utils.quadrature_jax.quad_weights(L=L, sampling="mwss"))
        wo = np.asarray(s2fft.utils.quadrature_jax.quad_weights(L=129, sampling="mwss"))
        maps, qc = [], []
        for i, freq in enumerate(f):
            power, integral = beam.normalize_native(e[i], w, args.symmetry)
            bm, dropped = beam.scalar_truncate(power, L, 128)
            checks = {
                "frequency_mhz": float(freq),
                "raw_integral_mV2": integral,
                "dropped_harmonic_fraction": dropped,
                "output_integral_over_4pi": float(
                    np.einsum("tp,t->", bm, wo) / (4 * np.pi)
                ),
                "minimum_power": float(bm.min()),
            }
            tmp = args.out / f"slice_{i:04}.tmp"
            final = args.out / f"slice_{i:04}.npz"
            with tmp.open("wb") as stream:
                np.savez_compressed(stream, bm=bm, frequency_mhz=freq)
            tmp.replace(final)
            checks["slice_sha256"] = sha256(final)
            atomic_json(args.out / f"slice_{i:04}.json", checks)
            maps.append(bm)
            qc.append(checks)
        tmp = args.out / "beam.tmp"
        with tmp.open("wb") as stream:
            np.savez_compressed(
                stream,
                freqs=f * 1e6,
                bm=np.stack(maps),
                lmax=128,
                field_symmetry=args.symmetry,
                parent_native_sha256=args.sha256,
                normalization=record["normalization"],
            )
        tmp.replace(args.out / "beam.npz")
        record.update(
            {
                "status": "DONE",
                "elapsed_s": time.monotonic() - start,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "beam_sha256": sha256(args.out / "beam.npz"),
                "slices": qc,
            }
        )
        atomic_json(args.out / "RUN_DONE.json", record)
    except BaseException as exc:
        atomic_json(
            args.out / "RUN_FAILED.json",
            {
                "status": "FAILED",
                "error": type(exc).__name__,
                "message": str(exc),
                "elapsed_s": time.monotonic() - start,
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
