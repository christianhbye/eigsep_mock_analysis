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
    neighbors = []
    pre_u, post_u = np.linspace(-30, 0, 80), np.linspace(0, -30, 80)
    all_u = np.r_[pre_u, u, post_u]
    all_time = (np.arange(len(all_u)) + 0.5) * 0.537
    for curvature in [0.0, 0.0006]:

        def profile(x):
            return 0.243 * x / 10.7 + curvature * x**2

        window = simulate_azimuth_window(
            edges,
            count_start=start,
            duration=duration,
            tau_s=0.2,
            step=0.113,
            contamination=profile(u),
        )
        neighbors.append(
            dict(
                curvature=curvature,
                time=all_time.tolist(),
                el=((all_u + 179.7 + 180) % 360 - 180).tolist(),
                tone=np.r_[profile(pre_u), window, 0.113 + profile(post_u)].tolist(),
                az=np.r_[np.zeros(80), az, np.full(80, 5)].tolist(),
                leg_id=np.r_[np.zeros(86), np.ones(86)].astype(int).tolist(),
                windows=[[80, 92]],
                dwell_i0=84,
                dwell_i1=87,
            )
        )
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
                neighbors=neighbors,
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
