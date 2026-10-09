"""Export small independent pointing references for the analysis test suite.

Run from mock_analysis with uv; the output is an explicit caller-selected
JSON fixture, not a real D5 product. No real data or analysis package is read.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from eigsep_pointing_sim.drive import uniform_window_means
from eigsep_pointing_sim.inject import simulate_windows


def declared_path(phase, n_legs=6):
    dt, rate = 0.537, 5.341
    start = 3 * dt + phase * dt
    moves = []
    for leg in range(n_legs):
        stop = start + 360 / rate
        moves.append((start, stop, rate * (-1) ** leg))
        start = (np.ceil(stop / dt) + 3) * dt + phase * dt
    edges = np.arange(int(np.ceil(moves[-1][1] / dt)) + 5) * dt
    leg_id = np.maximum(
        np.searchsorted(np.asarray(moves)[:, 0], edges[1:], side="left") - 1, 0
    )
    direction = (-1.0) ** leg_id
    return edges, moves, leg_id, direction


def make_fixture():
    root = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True
        ).strip()
    )
    cases = []
    for phase in [0.17, 0.43, 0.81]:
        edges, moves, leg_id, direction = declared_path(phase)
        y = uniform_window_means(edges, moves, y0=-180)
        effects = []
        if phase == 0.43:
            for zeta in [0.0, 0.16, 0.5]:
                for tau in [-0.1, 0.0, 0.1, 0.2]:
                    params = dict(tau_s=tau, zeta_p_deg=zeta)
                    injected = simulate_windows(edges, moves, y0=-180, **params)
                    effects.append(dict(params=params, el=injected["el"].tolist()))
            for params in [
                dict(tau_s=0.1, zeta_p_deg=0.2, T0_deg=-0.6, d_rev_deg=30),
                dict(tau_s=0.1, zeta_p_deg=0.2, windup_deg=0.3),
                dict(tau_s=0.1, zeta_p_deg=0.2, phi=[[0, -0.09 * 180 / np.pi]]),
            ]:
                injected = simulate_windows(edges, moves, y0=-180, **params)
                effects.append(dict(params=params, el=injected["el"].tolist()))
        cases.append(
            dict(
                phase=phase,
                edges=edges.tolist(),
                time=((edges[:-1] + edges[1:]) / 2).tolist(),
                rate_deg_s=5.341,
                counts_per_deg=11300 / 180,
                y0=-180,
                moves=np.asarray(moves).tolist(),
                counts=(y * (11300 / 180)).tolist(),
                leg_id=leg_id.tolist(),
                direction=direction.tolist(),
                target_counts=(direction * 11300).tolist(),
                effects=effects,
            )
        )
    return dict(
        status="synthetic regression reference; not D5 evidence",
        generator_repo="christianhbye/eigsep_mock_analysis",
        generator_commit=sha,
        generator_dirty=dirty,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numpy_version=np.__version__,
        n_sub=64,
        sampling="declared continuous uniform drive; exact drive/play moments",
        cases=cases,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    fixture = make_fixture()
    if fixture["generator_dirty"]:
        raise RuntimeError(
            "Commit the generator before exporting a regression reference"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    print(f"Wrote {args.output}: {args.output.stat().st_size} bytes")
