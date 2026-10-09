import numpy as np
import pytest
from eigsep_pointing_sim.batches import batch_moves
from eigsep_pointing_sim.drive import uniform_drive, uniform_window_means


def test_batch_displacement_and_partial_window_areas():
    batch = batch_moves([(0, 2, 1)], period_s=1, on_s=0.25)
    np.testing.assert_allclose(batch, [[0, 0.25, 4], [1, 1.25, 4]])
    np.testing.assert_allclose(
        uniform_drive([0, 0.125, 0.25, 0.5, 1, 1.125, 1.25, 2], batch),
        [0, 0.5, 1, 1, 1, 1.5, 2, 2],
    )
    np.testing.assert_allclose(
        uniform_window_means([0, 0.125, 0.25, 0.5, 1, 1.25], batch),
        [0.25, 0.75, 1, 1, 1.5],
    )


def test_partial_final_batch_keeps_total_motion_and_invalid_timing():
    batch = batch_moves([(1, 2.3, -1)], period_s=1, on_s=0.25)
    np.testing.assert_allclose(batch, [[1, 1.25, -4], [2, 2.075, -4]])
    assert uniform_drive([3], batch)[0] == pytest.approx(-1.3)
    with pytest.raises(ValueError, match="duration"):
        batch_moves([(0, 1, 1)], period_s=0.1, on_s=0.2)


def test_batch_advance_resets_at_macro_reversal_not_each_pulse():
    from eigsep_pointing_sim.inject import simulate_windows

    macro = [(1, 4, 1), (5, 8, -1)]
    pulses = batch_moves(macro, period_s=1, on_s=0.25)
    edges = np.arange(0, 9, 0.125)
    out = simulate_windows(edges, pulses, macro_moves=macro)
    # Just before the next macro command, the full three-degree motion
    # stays completed. At the reversal, advance resets to the new leg.
    assert out["advance"][38] == pytest.approx(3)
    assert out["advance"][40] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="batch transient"):
        simulate_windows(edges, pulses, macro_moves=macro, T0_deg=-0.6)
