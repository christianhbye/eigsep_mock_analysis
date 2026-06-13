"""Frame-mapping and anti-aliasing validation (spec "Validation")."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from masks import boolean_weight, mwss_grid, open_sky_weight  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
LMAX = 128


def _open_fraction(W, thetas):
    w = np.sin(thetas)[:, None]
    return (W * w).sum() / (np.ones_like(W) * w).sum()


def test_antialias_matches_boolean_open_fraction():
    # For a generic horizon the solid-angle open fraction from the
    # fractional and boolean masks must agree to ~grid resolution.
    thetas, phis = mwss_grid(LMAX)
    az = np.linspace(0.0, 2 * np.pi, 720, endpoint=False)
    rng = np.random.default_rng(3)
    alpha = np.deg2rad(rng.uniform(-5, 15, 720))
    Wf = open_sky_weight(alpha, az, thetas, phis)
    Wb = boolean_weight(alpha, az, thetas, phis)
    assert abs(_open_fraction(Wf, thetas) - _open_fraction(Wb, thetas)) < 0.01


@pytest.mark.skipif(
    not (OUT / "horizons_position.npz").exists(),
    reason="run make_horizons.py (eigsep_terrain env) first",
)
def test_nominal_frame_matches_existing_horizon_mwss():
    import eigsim

    horizon_raw, lmax = eigsim.load_horizon()  # nominal, croissant frame
    existing_open = np.isnan(horizon_raw)  # NaN = open sky in the raw file
    thetas, phis = mwss_grid(lmax)

    hz = np.load(OUT / "horizons_position.npz", allow_pickle=True)
    names = [str(n) for n in hz["names"]]
    nom = hz["alpha_h"][names.index("nominal")]
    W = open_sky_weight(nom, hz["az_grid"], thetas, phis)
    our_open = W >= 0.5

    # The two nominal masks come from different methods (calc_horizon vs
    # ray-trace+nearest-neighbour) but the same site: they must agree on
    # the large majority of cells. A gross azimuth-frame error would push
    # agreement toward 0.5.
    agree = (our_open == existing_open).mean()
    # A gross azimuth-frame error pushes agreement toward 0.5; method
    # differences near the jagged horizon cost a few percent at most.
    assert agree > 0.85, f"nominal masks agree only {agree:.2f} (frame error?)"
