import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beams import (  # noqa: E402
    band_limited_power_fraction,
    healpix_to_mwss,
    isotropic_beam,
)

LMAX = 128
NSIDE = 64
DATA = Path(__file__).resolve().parents[1] / "eigsim" / "data"


def _bowtie():
    """The bowtie in both samplings, for the round-trip check."""
    hp = np.load(DATA / "eigsep_bowtie_v000.npz")
    mwss = np.load(DATA / "eigsep_bowtie_v000_mwss.npz")
    return hp, mwss


needs_data = pytest.mark.skipif(
    not (DATA / "eigsep_bowtie_v000.npz").exists(), reason="eigsim beam data absent"
)


def test_isotropic_shape_and_constancy():
    bm = isotropic_beam(5, (130, 258))
    assert bm.shape == (5, 130, 258)
    assert np.all(bm == bm.flat[0])


@needs_data
def test_healpix_to_mwss_reproduces_stored_bowtie():
    """The conversion must be the one eigsim's stored MWSS bowtie was made with.

    This is the whole justification for trusting a Vivaldi that only exists in
    HEALPix: the identical recipe reproduces the bowtie exactly.
    """
    hp, mwss = _bowtie()
    n = 3
    got = healpix_to_mwss(hp["bm"][:n], int(hp["nside"]), int(mwss["lmax"]))
    assert got.shape == mwss["bm"][:n].shape
    assert np.allclose(got, mwss["bm"][:n], rtol=0, atol=1e-12)


@needs_data
def test_niter_zero_is_not_good_enough():
    """Guards the NITER constant: without refinement the round trip is wrong.

    If this ever starts passing, s2fft's HEALPix quadrature improved and the
    comment on NITER needs revisiting -- but silently lowering it would have
    put a 12 per cent error into the beam comparison.
    """
    hp, mwss = _bowtie()
    got = healpix_to_mwss(hp["bm"][:1], int(hp["nside"]), int(mwss["lmax"]), niter=0)
    err = np.abs(got - mwss["bm"][:1]).max() / np.abs(mwss["bm"][:1]).max()
    assert err > 1e-3


@needs_data
def test_beams_are_resolved_at_the_band_limit():
    """Both beams must be band-limited, or the comparison measures the grid."""
    hp, _ = _bowtie()
    frac = band_limited_power_fraction(hp["bm"][:2], int(hp["nside"]), LMAX)
    assert np.all(frac > 1 - 1e-6)
