import numpy as np
import pytest
from eigsep_pointing_sim.drive import (
    uniform_drive,
    uniform_micro_means,
    uniform_window_means,
)


def test_uniform_drive_and_partial_window_integral():
    moves = [(0.25, 1.75, 4.0), (2.25, 3.75, -4.0)]
    values = uniform_drive([0, 0.25, 1, 1.75, 2, 2.25, 3, 3.75, 4], moves, y0=-3)
    np.testing.assert_allclose(values, [-3, -3, 0, 3, 3, 3, 0, -3, -3])
    means = uniform_window_means(np.arange(5.0), moves, y0=-3)
    # First interval's advance integral is 4 * 0.75**2 / 2.
    np.testing.assert_allclose(means, [-1.875, 1.875, 1.875, -1.875])


def test_exact_micro_means_preserve_window_identity_at_32_and_64():
    edges = np.arange(31) * 0.537
    moves = [(0.381, 6.172, 5.341), (8.081, 13.872, -5.341)]
    target = uniform_window_means(edges, moves, y0=-15)
    for n_sub in [32, 64]:
        fine = uniform_micro_means(edges, moves, y0=-15, n_sub=n_sub)
        np.testing.assert_allclose(
            fine["y_mean"].reshape(-1, n_sub).mean(1), target, rtol=0, atol=1e-11
        )
        assert np.diff(fine["time"]).min() > 0
        # Fixed midpoint sampling is not an exact quadrature next to a break.
        midpoint = uniform_drive(fine["time"], moves, y0=-15).reshape(-1, n_sub).mean(1)
        assert np.max(np.abs(midpoint - target)) > 1e-6


def test_integrals_are_stable_at_unix_epoch_and_validate_moves():
    edges = np.arange(8.0)
    moves = np.array([(0.25, 1.75, 4), (3.25, 4.75, -4)])
    offset = 1_784_320_000.0
    shifted = moves.copy()
    shifted[:, :2] += offset
    np.testing.assert_allclose(
        uniform_window_means(edges + offset, shifted, y0=-3),
        uniform_window_means(edges, moves, y0=-3),
        atol=1e-12,
    )
    for n_sub in [32, 64]:
        fine = uniform_micro_means(edges + offset, shifted, y0=-3, n_sub=n_sub)
        np.testing.assert_allclose(
            fine["y_mean"].reshape(-1, n_sub).mean(1),
            uniform_window_means(edges + offset, shifted, y0=-3),
            atol=1e-11,
        )
    with pytest.raises(ValueError, match="overlap"):
        uniform_drive([0, 1, 2], [(0, 2, 1), (1, 3, -1)])
    with pytest.raises(ValueError, match="increasing"):
        uniform_window_means([0, 2, 1], moves)


def test_play_model_delays_takeup_and_holds_through_dwells():
    from eigsep_pointing_sim.inject import play_model

    moves = [(0.25, 1.75, 4.0), (2.25, 3.75, -4.0)]
    model = play_model(moves, y0=-3, zeta_p_deg=0.5)
    np.testing.assert_allclose(model["moves"], [(0.5, 1.75, 4), (2.5, 3.75, -4)])
    assert model["y0"] == pytest.approx(-2.5)
    np.testing.assert_allclose(
        uniform_drive([0, 0.5, 1, 1.75, 2.5, 3, 3.75], model["moves"], y0=model["y0"]),
        [-2.5, -2.5, -0.5, 2.5, 2.5, 0.5, -2.5],
    )
    short = play_model([(0, 0.1, 4)], y0=0, zeta_p_deg=0.5)
    assert len(short["moves"]) == 0 and short["y0"] == pytest.approx(0.5)
    zero = play_model(moves, y0=-3, zeta_p_deg=0)
    np.testing.assert_allclose(zero["moves"], moves)


def test_zero_effects_injection_preserves_exact_window_identity():
    from eigsep_pointing_sim.inject import simulate_windows

    edges = np.arange(31) * 0.537
    moves = [(0.381, 6.172, 5.341), (8.081, 13.872, -5.341)]
    out = simulate_windows(edges, moves, y0=-15)
    np.testing.assert_allclose(
        out["el"], uniform_window_means(edges, moves, y0=-15), rtol=0, atol=1e-11
    )
    lagged = simulate_windows(edges, moves, y0=-15, tau_s=0.1)
    np.testing.assert_allclose(
        lagged["el"],
        uniform_window_means(edges + 0.1, moves, y0=-15),
        rtol=0,
        atol=1e-11,
    )


def test_stationary_ar1_noise_and_reproducible_pointing_effects():
    from eigsep_pointing_sim.inject import ar1_noise, simulate_windows

    noise = ar1_noise(40_000, 0.3, 0.9, rng=np.random.default_rng(922))
    assert noise.std() == pytest.approx(0.3, abs=0.02)
    assert np.corrcoef(noise[:-1], noise[1:])[0, 1] == pytest.approx(0.9, abs=0.02)
    with pytest.raises(ValueError, match="rho"):
        ar1_noise(20, 0.3, 1.1, rng=np.random.default_rng(922))
    edges = np.arange(31) * 0.537
    moves = [(0.381, 6.172, 5.341), (8.081, 13.872, -5.341)]
    kwargs = dict(
        y0=-15,
        tau_s=0.1,
        zeta_p_deg=0.2,
        T0_deg=-0.6,
        d_rev_deg=30,
        eps=0.003,
        phi=[[0.1, 0.2]],
        ar1=(0.3, 0.9),
    )
    one = simulate_windows(edges, moves, rng=np.random.default_rng(923), **kwargs)
    two = simulate_windows(edges, moves, rng=np.random.default_rng(923), **kwargs)
    np.testing.assert_array_equal(one["el"], two["el"])
    assert np.max(np.abs(one["el"] - one["drive_mean"])) > 0.1
    assert one["advance"].min() >= 0
