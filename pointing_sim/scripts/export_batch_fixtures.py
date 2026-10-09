"""Export idealized B8 batch references, independently of the inverse."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from eigsep_pointing_sim.batches import batch_moves
from eigsep_pointing_sim.drive import uniform_window_means
from eigsep_pointing_sim.inject import simulate_windows
from export_analysis_fixtures import declared_path


def main(output):
    root = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("Commit generator before exporting references")
    cases = []
    for phase in [0.17, 0.43, 0.81]:
        edges, macro, _, _ = declared_path(phase)
        batch = batch_moves(macro)
        y = uniform_window_means(edges, batch, y0=-180)
        for endpoint in [-180.0, 180.0]:
            y[np.abs(y - endpoint) < 1e-10] = endpoint
        effects = []
        for n_sub in [32, 64]:
            effect = simulate_windows(
                edges, batch, y0=-180, tau_s=0.1, zeta_p_deg=0.16, n_sub=n_sub
            )
            effects.append(dict(n_sub=n_sub, el=effect["el"].tolist()))
        cases.append(
            dict(
                phase=phase,
                time=((edges[:-1] + edges[1:]) / 2).tolist(),
                edges=edges.tolist(),
                counts=(y * (11300 / 180)).tolist(),
                counts_per_deg=11300 / 180,
                rate_deg_s=5.341,
                macro_moves=np.asarray(macro).tolist(),
                effects=effects,
            )
        )
    output.write_text(
        json.dumps(
            dict(
                generator_commit=sha,
                generator_dirty=False,
                generator_repo="christianhbye/eigsep_mock_analysis",
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                period_s=0.179,
                on_s=0.072,
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
