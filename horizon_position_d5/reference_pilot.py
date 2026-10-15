"""Run the first full-resolution DEM reference and reproduction controls.

Run with python -m horizon_position_d5.reference_pilot --inputs DIR --output DIR.
This phase does not compute temperature derivatives or validate an interpolator.
"""

import argparse
import json
import os
import platform
import resource
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import reference
from .cache import BlockCache, atomic_json, file_sha256
from .inputs import verify_inputs
from .reference import BilinearDEM, trace_horizon


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    inputs = verify_inputs(args.inputs)
    with np.load(args.inputs / "team_horizon.npz") as team:
        p = team["antenna_enu_m"].copy()
        bearings = team["bearings_deg"].copy()
        nominal = np.stack([team[f"elev_rad_{era}"] for era in ("30m", "87.5m", "91m")])
    identity = {
        "reference_sha256": file_sha256(reference.__file__),
        "numpy_version": np.__version__,
        "dem_sha256": inputs["dem.npz"]["sha256"],
    }
    cache = BlockCache(args.output / "traces", identity)
    record = {
        "phase": "reference-pilot",
        "label": "diagnostic",
        "input_hashes": {k: v["sha256"] for k, v in inputs.items()},
        "reference_identity": identity,
        "host": platform.node(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "parameters": {
            "inner_cutoff_m": 0.25,
            "earth_radius_m": 6371e3,
            "height_offsets_m": [-61.0, -3.5, 0.0],
        },
    }
    atomic_json(args.output / "reference_run.json", record)
    print("INPUTS VERIFIED", flush=True)
    dem = BilinearDEM.from_npz(args.inputs / "dem.npz")
    results, durations = {}, {}
    for method in ("sampled", "continuous"):
        params = {
            **record["parameters"],
            "position_enu_m": p.tolist(),
            "bearings_deg": bearings.tolist(),
            "method": method,
            "march_step_m": 0.25,
        }
        key = f"nominal-{method}"
        start = time.monotonic()
        arrays = cache.load(key, params)
        reused = arrays is not None
        if arrays is None:
            arrays = asdict(
                trace_horizon(
                    dem, p, bearings, height_offsets_m=(-61.0, -3.5, 0.0), method=method
                )
            )
            cache.write(key, params, arrays)
        durations[method] = time.monotonic() - start
        results[method] = arrays
        print(
            json.dumps(
                {"checkpoint": key, "seconds": durations[method], "reused": reused}
            ),
            flush=True,
        )
    sampled, continuous = results["sampled"], results["continuous"]
    reproduction = float(np.max(np.abs(sampled["alpha_rad"] - nominal)))
    lower_violation = float(np.max(sampled["alpha_rad"] - continuous["alpha_rad"]))
    # Dense samples are lower bounds, not a certified error bound on the maximum.
    selected = np.unique(
        np.r_[
            np.linspace(0, len(bearings) - 1, 8, dtype=int),
            np.argmax(continuous["alpha_rad"] - sampled["alpha_rad"], axis=1),
        ]
    )
    dense = trace_horizon(
        dem,
        p,
        bearings[selected],
        height_offsets_m=(-61.0, -3.5, 0.0),
        method="sampled",
        march_step_m=0.01,
    )
    gap = continuous["alpha_rad"][:, selected] - dense.alpha_rad
    report = {
        "label": "diagnostic",
        "phase": "reference-pilot",
        "source_commit": record["source_commit"],
        "reference_identity": identity,
        "team_reproduction_max_abs_rad": reproduction,
        "sampled_above_continuous_max_rad": lower_violation,
        "dense_selected_bearing_indices": selected.tolist(),
        "dense_gap_rad": gap.tolist(),
        "continuous_minus_product_max_rad_by_era": np.max(
            continuous["alpha_rad"] - nominal, axis=1
        ).tolist(),
        "edge_limited_counts": continuous["edge_limited"].sum(axis=1).tolist(),
        "trace_seconds": durations,
        "wall_seconds": time.monotonic() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "pass_reproduction": reproduction <= 1e-14,
        "pass_lower_bounds": lower_violation <= 1e-13 and float(gap.min()) >= -1e-13,
        "scope": (
            "full 3600-bearing nominal horizons, all three heights; "
            "no temperature or inference result"
        ),
    }
    atomic_json(args.output / "reference_pilot.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["pass_reproduction"] or not report["pass_lower_bounds"]:
        raise RuntimeError("reference pilot failed; inspect reference_pilot.json")


if __name__ == "__main__":
    main()
