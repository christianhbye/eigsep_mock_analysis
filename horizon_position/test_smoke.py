"""End-to-end smoke test for run_sims.py (gated behind EIGSEP_SMOKE=1).

Requires output/horizons_position.npz to exist (run make_horizons.py in
the eigsep_terrain env first). Run with:

    EIGSEP_SMOKE=1 uv run pytest horizon_position/test_smoke.py -v
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"

pytestmark = pytest.mark.skipif(
    os.environ.get("EIGSEP_SMOKE") != "1",
    reason="set EIGSEP_SMOKE=1 to run (spawns subprocess, compiles JAX)",
)


def test_run_sims_smoke():
    if not (OUT / "horizons_position.npz").exists():
        pytest.skip("run make_horizons.py (eigsep_terrain env) first")
    tag = "_pytest"
    cmd = [
        sys.executable,
        str(HERE / "run_sims.py"),
        "--freq-stride",
        "40",
        "--n-times",
        "6",
        "--output-tag",
        tag,
    ]
    subprocess.run(cmd, check=True, cwd=HERE.parent)
    out = OUT / f"position_sims{tag}.npz"
    try:
        d = np.load(out, allow_pickle=True)
        assert d["t_sys"].shape[0] == 19
        assert d["fgnd"].shape[0] == 19
        assert np.isfinite(d["t_sys"]).all()
        # nominal (index 0) and +10 m East must differ; t_sys positive
        names = [str(n) for n in d["names"]]
        i10 = names.index("x_p_10")
        assert d["t_sys"].min() > 0
        assert not np.allclose(d["t_sys"][0], d["t_sys"][i10])
    finally:
        out.unlink(missing_ok=True)


def test_run_beam_sims_bowtie_matches_run_sims_nominal():
    """beam_sims.npz's bowtie row is position_sims.npz's nominal row.

    Same beam, sky, horizon mask, times and config, so they must agree to
    float precision. The paper's test_npz_contract requires 1e-5 between
    beam_comparison.npz's bowtie and foreground_svd.npz; before the port the
    two scripts used different masks and frames and differed by kelvins.
    """
    if not (OUT / "horizons_position.npz").exists():
        pytest.skip("run make_horizons.py (eigsep_terrain env) first")
    tag = "_pytest_beams"
    common = ["--freq-stride", "40", "--n-times", "6", "--output-tag", tag]
    sims = OUT / f"position_sims{tag}.npz"
    beams = OUT / f"beam_sims{tag}.npz"
    try:
        subprocess.run(
            [sys.executable, str(HERE / "run_sims.py"), *common],
            check=True,
            cwd=HERE.parent,
        )
        subprocess.run(
            [
                sys.executable,
                str(HERE / "run_beam_sims.py"),
                *common,
                "--beams",
                "bowtie",
                "isotropic",
            ],
            check=True,
            cwd=HERE.parent,
        )
        s = np.load(sims, allow_pickle=True)
        b = np.load(beams, allow_pickle=True)
        assert [str(x) for x in b["beams"]] == ["bowtie", "isotropic"]
        i_nom = [str(n) for n in s["names"]].index("nominal")
        np.testing.assert_array_equal(b["freqs_mhz"], s["freqs_mhz"])
        np.testing.assert_array_equal(b["times_jd"], s["times_jd"])
        np.testing.assert_allclose(b["t_sys"][0], s["t_sys"][i_nom], rtol=1e-6)
        np.testing.assert_allclose(b["fgnd"][0], s["fgnd"][i_nom], rtol=1e-6)
        assert str(b["pos_sha"]) == str(s["pos_sha"])
    finally:
        sims.unlink(missing_ok=True)
        beams.unlink(missing_ok=True)
