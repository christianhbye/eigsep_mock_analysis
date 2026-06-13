import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis  # noqa: E402

T_GND, T_RCVR = 300.0, 50.0


def _fake(P=4, nt=5, nf=3, seed=0):
    rng = np.random.default_rng(seed)
    t_sys = rng.uniform(1000.0, 5000.0, (P, nt, nf))
    fgnd = rng.uniform(0.02, 0.2, (P, nf))
    return t_sys, fgnd


def test_glc_roundtrip():
    # GLC inverts t_sys = (1-f)*t_sky + f*Tgnd + Trcvr
    rng = np.random.default_rng(2)
    t_sky = rng.uniform(1000.0, 5000.0, (3, 4))
    f = rng.uniform(0.05, 0.3, (3, 4))
    t_sys = (1 - f) * t_sky + f * T_GND + T_RCVR
    out = analysis.glc(t_sys, f, T_GND, T_RCVR)
    assert np.allclose(out, t_sky)


def test_uncorrected_nominal_is_zero():
    t_sys, fgnd = _fake()
    d = analysis.delta_waterfall(t_sys, fgnd, "uncorrected", T_GND, T_RCVR)
    assert np.allclose(d[0], 0.0)
    assert d.shape == t_sys.shape


def test_miscorrected_identity():
    # mode 3 must equal uncorrected delta / (1 - fgnd_nominal)
    t_sys, fgnd = _fake()
    d_unc = analysis.delta_waterfall(t_sys, fgnd, "uncorrected", T_GND, T_RCVR)
    d_mis = analysis.delta_waterfall(t_sys, fgnd, "miscorrected", T_GND, T_RCVR)
    f0 = fgnd[0][None, None, :]
    assert np.allclose(d_mis, d_unc / (1 - f0))


def test_oracle_nominal_is_zero():
    t_sys, fgnd = _fake()
    d = analysis.delta_waterfall(t_sys, fgnd, "oracle", T_GND, T_RCVR)
    assert np.allclose(d[0], 0.0)


def test_summary_and_reductions_shapes():
    t_sys, fgnd = _fake(P=4, nt=5, nf=3)
    d = analysis.delta_waterfall(t_sys, fgnd, "uncorrected", T_GND, T_RCVR)
    s = analysis.summary_stats(d)
    assert s["rms"].shape == (4,) and s["max"].shape == (4,)
    assert analysis.rms_over_time(d).shape == (4, 3)
    assert analysis.rms_over_freq(d).shape == (4, 5)
    assert np.allclose(s["rms"][0], 0.0)
