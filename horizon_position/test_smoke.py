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
        for b in OUT.glob(f"pos{tag}_batch_*.npz"):
            b.unlink()
