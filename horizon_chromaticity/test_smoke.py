"""Smoke tests for the horizon chromaticity project.

These run the actual scripts on tiny inputs. They need the eigsim
data files and take a few minutes (JAX compilation), so they are
gated behind an env var:

    EIGSEP_SMOKE=1 uv run pytest horizon_chromaticity/test_smoke.py -v
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("EIGSEP_SMOKE") != "1",
    reason="set EIGSEP_SMOKE=1 to run smoke tests",
)

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
CASES = ("nohorizon", "quarry", "eigsep")


@pytest.fixture(scope="module")
def horizons():
    subprocess.run([sys.executable, str(PROJECT_DIR / "make_horizons.py")], check=True)
    return np.load(OUTPUT_DIR / "horizons.npz")


def _quad_weights(lmax):
    import s2fft

    return np.asarray(
        s2fft.utils.quadrature_jax.quad_weights(L=lmax + 1, sampling="mwss")
    )


def test_mask_shapes_and_dtypes(horizons):
    lmax = int(horizons["lmax"])
    shape = (lmax + 2, 2 * (lmax + 1))  # (L + 1, 2L) with L = lmax + 1
    for case in CASES:
        assert horizons[case].shape == shape
        assert horizons[case].dtype == np.bool_


def test_nohorizon_all_open(horizons):
    assert horizons["nohorizon"].all()


def test_eigsep_mask_matches_file(horizons):
    import eigsim

    raw, _ = eigsim.load_horizon()
    assert np.array_equal(horizons["eigsep"], np.isnan(raw))


def test_quarry_is_ring_cut(horizons):
    mask = horizons["quarry"]
    i_cut = int(horizons["i_cut"])
    # whole rings: open above the cut, blocked below, nothing partial
    assert mask[:i_cut].all()
    assert not mask[i_cut:].any()


def test_quarry_solid_angle_matches_eigsep(horizons):
    lmax = int(horizons["lmax"])
    w = _quad_weights(lmax)
    nphi = horizons["quarry"].shape[1]

    omega_quarry = (w[:, None] * ~horizons["quarry"]).sum()
    omega_eigsep = (w[:, None] * ~horizons["eigsep"]).sum()
    assert omega_eigsep == pytest.approx(float(horizons["omega_blocked_target"]))
    # the ring cut can't do better than one ring of solid angle
    max_ring_omega = (nphi * w).max()
    assert abs(omega_quarry - omega_eigsep) <= max_ring_omega
    # and both block a substantial chunk of the sphere (~half)
    assert 0.3 * 4 * np.pi < omega_eigsep < 0.7 * 4 * np.pi


def test_fgnd_per_case(horizons):
    import croissant as cro

    import eigsim

    beam_freqs_hz, beam_data, _ = eigsim.load_beam()
    freqs = beam_freqs_hz[:1] / 1e6
    fgnd = {}
    for case in CASES:
        beam = cro.Beam(
            beam_data[:1],
            freqs,
            sampling="mwss",
            niter=0,
            horizon=horizons[case],
        )
        fgnd[case] = float(np.asarray(beam.compute_fgnd())[0])
    assert fgnd["nohorizon"] == pytest.approx(0.0, abs=1e-12)
    assert fgnd["quarry"] > 0.01
    assert fgnd["eigsep"] > 0.01


@pytest.mark.parametrize("case", CASES)
def test_run_sims_end_to_end(case, horizons):
    """Tiny full run per case: finite, positive, correctly shaped."""
    outfile = OUTPUT_DIR / f"chromaticity_{case}_smoke.npz"
    outfile.unlink(missing_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_DIR / "run_sims.py"),
            "--case",
            case,
            "--n-times",
            "3",
            "--max-orientations",
            "2",
            "--freq-stride",
            "50",
            "--batch-size",
            "2",
            "--output-tag",
            "_smoke",
        ],
        check=True,
    )
    d = np.load(outfile)
    t_sys = d["t_sys"]
    # 2 orientations, 3 times, 5 freqs (201 channels, stride 50)
    assert t_sys.shape == (2, 3, 5)
    assert np.all(np.isfinite(t_sys))
    assert np.all(t_sys > 0)
    assert str(d["case"]) == case
    assert len(d["elevations"]) == 2
    assert len(d["azimuths"]) == 2
    outfile.unlink()
