"""Tests for the S4 T6 diagnostic producer (plan Task 8; spec section 6.4 T6).

Pure helpers only: these tests need neither the frozen analysis source nor
any S1 product.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "s4_t6_diagnostic.py"


@pytest.fixture(scope="module")
def t6():
    spec = importlib.util.spec_from_file_location("s4_t6_diagnostic", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_block_signs_preserve_masks(t6):
    rng = np.random.default_rng(0)
    nf, nb = 5, 23
    blocks = np.array([0] * 3 + [1] * 7 + [2] * 1 + [3] * 8 + [-1] * 4)
    mask = rng.uniform(size=(nf, nb)) > 0.2
    mask[:, blocks < 0] = False
    resid = np.where(mask, rng.normal(size=(nf, nb)), np.nan)
    signs = t6.block_signs(np.random.default_rng(1), blocks)
    assert set(signs) == {0, 1, 2, 3}
    assert set(signs.values()) <= {-1.0, 1.0}
    out = t6.apply_block_signs(resid, mask, blocks, signs)
    # masks and missing cells are unchanged
    assert np.array_equal(np.isnan(out), np.isnan(resid))
    assert np.array_equal(np.isfinite(out), mask)
    # one sign per block at every frequency
    for b, s in signs.items():
        cols = blocks == b
        m = mask[:, cols]
        assert np.allclose(out[:, cols][m], s * resid[:, cols][m])
    # rebuilt data: no missing cell becomes data
    base = np.ones((nf, nb))
    data = t6.rebuild_data(base, np.zeros((nf, nb)), out, mask)
    assert np.array_equal(np.isfinite(data), mask)


def test_confusion(t6):
    records = [
        {"truth": "A", "model": "mA", "passes": True},
        {"truth": "A", "model": "mA", "passes": True},
        {"truth": "A", "model": "mA", "passes": False},
        {"truth": "A", "model": "mB", "passes": False},
        {"truth": "A", "model": "mB", "passes": False},
        {"truth": "A", "model": "mB", "passes": True},
        {"truth": "B", "model": "mA", "passes": False},
        {"truth": "B", "model": "mB", "passes": True},
        {"truth": "null", "model": "mA", "passes": False},
        {"truth": "null", "model": "mB", "passes": False},
    ]
    tab = t6.confusion(records)
    assert tab["A"]["mA"]["successes"] == 2 and tab["A"]["mA"]["n"] == 3
    assert tab["A"]["mB"]["successes"] == 1 and tab["A"]["mB"]["n"] == 3
    assert tab["B"]["mA"]["successes"] == 0 and tab["B"]["mA"]["n"] == 1
    assert tab["null"]["mB"]["rate"] == 0.0
    lo, hi = tab["A"]["mA"]["wilson95"]
    assert lo < 2 / 3 < hi
    # Wilson at 0/50: the upper bound is about 0.07 (spec section 6.4 T6)
    assert np.isclose(t6.wilson(0, 50)[1], 0.0714, atol=2e-3)


def test_seed_sets_disjoint(t6):
    keys = {}
    for set_id in (t6.SET_CAL, t6.SET_EVAL, t6.SET_TRUTH, t6.SET_SMOKE, t6.SET_PROBE):
        for truth in range(44):
            for draw in range(3):
                ss = t6.seed_for(set_id, truth, draw)
                assert ss.spawn_key == (set_id, truth, draw)
                keys[(set_id, truth, draw)] = tuple(ss.generate_state(4))
    assert len(set(keys.values())) == len(keys)
    assert len({t6.SET_CAL, t6.SET_EVAL, t6.SET_TRUTH, t6.SET_SMOKE, t6.SET_PROBE}) == 5
    assert (t6.SET_CAL, t6.SET_EVAL) == (1, 2)
    # the plan's batches: evaluation draws in 5 batches of 10
    assert t6.seed_batch(0) == 0 and t6.seed_batch(49) == 4


def test_degenerate_truth(t6):
    rng = np.random.default_rng(3)
    x = np.linalg.qr(rng.normal(size=(300, 3)))[0]
    planted = np.column_stack([x[:, 0], 3.0 * x[:, 1], x[:, 0] + 1e-5 * x[:, 2]])
    v, rel = t6.weakest_direction(planted)
    # the truth along the weakest direction lies in the planted near-null
    col = planted @ v
    assert np.linalg.norm(col) < 1e-4 * np.linalg.norm(v) * np.linalg.norm(planted)
    assert rel < 1e-4
    assert np.isclose(abs(v[0] / v[2]), 1.0, rtol=1e-3) and abs(v[1]) < 1e-3
    # an exactly null direction (s = 0) is skipped: the weakest above zero
    exact = np.column_stack([x[:, 0], x[:, 0], x[:, 1] + 1e-3 * x[:, 2], x[:, 1]])
    v2, rel2 = t6.weakest_direction(exact)
    assert 1e-12 < rel2 < 1e-2
    assert abs(v2[0]) < 1e-6 and abs(v2[1]) < 1e-6
    assert np.isclose(v2[2], -v2[3], rtol=1e-2)


def test_truth_rebuild(t6):
    rng = np.random.default_rng(4)
    nf, nb = 4, 30
    mask = rng.uniform(size=(nf, nb)) > 0.1
    blocks = np.repeat(np.arange(6), 5)
    y_base = rng.normal(1000, 50, (nf, nb))
    y_gen = y_base + rng.normal(0, 5, (nf, nb))
    y = np.where(mask, y_gen + rng.normal(0, 1, (nf, nb)), np.nan)
    resid = np.where(mask, y - y_gen, np.nan)
    signs = {b: 1.0 for b in range(6)}
    truth = y_gen - y_base  # the fitted generic model as the truth
    data = t6.rebuild_data(
        y_base, truth, t6.apply_block_signs(resid, mask, blocks, signs), mask
    )
    assert np.allclose(data[mask], y[mask], rtol=1e-12)
    assert np.all(np.isnan(data[~mask]))


def test_sealed_amplitudes(t6, tmp_path):
    secret = {"alpha_sw": 0.0123456789, "rho": -1.987654321}

    def fit_truths():
        arrays = {k: np.full((3, 4), v) for k, v in secret.items()}
        return arrays, {k: np.array([v]) for k, v in secret.items()}

    log = tmp_path / "run.log"
    out = tmp_path / "sealed"
    sha = t6.sealed_stage(fit_truths, out, log)
    files = sorted(p.name for p in out.iterdir())
    assert files == ["truths.npz"]
    with np.load(out / "truths.npz") as z:
        assert np.allclose(z["truth__alpha_sw"], secret["alpha_sw"])
        assert np.allclose(z["params__rho"], secret["rho"])
    text = log.read_text()
    assert sha in text and "truths.npz" in text
    for v in secret.values():
        for fmt in (f"{v}", f"{v:.3g}", f"{v:.4f}", f"{v:.2e}"):
            assert fmt not in text
    rec = json.loads(text.strip().splitlines()[-1])
    assert set(rec) == {"time", "event", "file", "sha256", "n_truths"}
