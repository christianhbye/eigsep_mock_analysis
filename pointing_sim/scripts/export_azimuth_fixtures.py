"""Export independent B8 azimuth-window references; no analysis imports."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from eigsep_pointing_sim.azimuth import ramp_window_means, simulate_azimuth_window


def main(output):
    root = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("Commit generator before exporting references")
    edges = np.arange(13) * 0.537
    time = (edges[:-1] + edges[1:]) / 2
    u = np.array([-10.7, -8, -5.3, -2.7, 0, 0, 0, 0, -2.7, -5.3, -8, -10.7])
    start, duration = 2.21, 0.936
    az = 5 * ramp_window_means(edges, start, duration)
    cases = []
    rng = np.random.default_rng(953)
    for tau in [0, 0.2, 0.4]:
        for realization in range(50):
            tone = simulate_azimuth_window(
                edges,
                count_start=start,
                duration=duration,
                tau_s=tau,
                step=0.113,
                contamination=0.243 * u / 10.7,
                noise_sd=0.002,
                rng=rng,
            )
            cases.append(dict(tau_s=tau, realization=realization, tone=tone.tolist()))
    output.write_text(
        json.dumps(
            dict(
                generator_repo="christianhbye/eigsep_mock_analysis",
                generator_commit=sha,
                generator_dirty=False,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                edges=edges.tolist(),
                time=time.tolist(),
                u=u.tolist(),
                az=az.tolist(),
                count_start=start,
                duration=duration,
                cases=cases,
            ),
            separators=(",", ":"),
        )
        + "\n"
    )
    print(f"Wrote {output}: {output.stat().st_size} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
