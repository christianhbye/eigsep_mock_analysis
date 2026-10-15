import numpy as np
import pytest

from horizon_position_d5.inference import fit_surrogate, local_identifiability


@pytest.mark.parametrize("nuisance", [False, True])
def test_known_linear_injection_recovers_position_and_nuisance(nuisance):
    rng = np.random.default_rng(941)
    a = rng.normal(size=(10, 3))
    baseline = rng.uniform(100, 1000, size=10)
    true_position = np.array([0.7, -1.2, 0.4])
    gain, offset = (1.02, -3.0) if nuisance else (1.0, 0.0)

    def model(position):
        return baseline + a @ position

    def jacobian(position):
        return a

    fit = fit_surrogate(
        model, jacobian, gain * model(true_position) + offset, nuisance=nuisance
    )
    assert fit["success"]
    assert np.allclose(fit["delta_m"], true_position, atol=1e-8, rtol=0)
    assert fit["gain"] == pytest.approx(gain, abs=1e-10)
    assert fit["offset_k"] == pytest.approx(offset, abs=1e-8)
    assert fit["rank"] == 3
    assert fit["linear_width_valid"]
    expected = a
    if nuisance:
        nuisance_columns = np.column_stack([model(true_position), np.ones(10)])
        projection = np.eye(10) - nuisance_columns @ np.linalg.pinv(nuisance_columns)
        expected = projection @ (gain * a)
    covariance = np.linalg.inv(expected.T @ expected)
    assert np.allclose(fit["linear_sigma_m_per_k"], np.sqrt(np.diag(covariance)))


def test_global_gain_offset_degeneracy_is_reported():
    temperature = np.arange(10.0)
    jac = np.column_stack([temperature, np.ones(10), np.sin(temperature)])
    result = local_identifiability(jac, temperature, nuisance=True)
    assert result["rank"] == 1
    assert not result["linear_width_valid"]
    assert result["condition"] is None
    assert np.array_equal(result["linear_sigma_m_per_k"], np.zeros(3))


def test_entirely_degenerate_position_is_not_projection_roundoff_rank():
    temperature = np.arange(10.0)
    jac = np.column_stack([temperature, np.ones(10), 3 * temperature + 2])
    result = local_identifiability(jac, temperature, nuisance=True)
    assert result["rank"] == 0
    assert not result["linear_width_valid"]


def test_reference_width_rejects_kinks_and_step_instability():
    from horizon_position_d5.inference import reference_resolution

    j = np.eye(3)
    temperature = np.arange(3.0)
    stable = reference_resolution(j, j, j, j, temperature)
    assert stable["derivative_resolved"]
    assert not reference_resolution(j, 1.1 * j, j, j, temperature)[
        "derivative_resolved"
    ]
    assert not reference_resolution(j, j, 0.5 * j, 1.5 * j, temperature)[
        "derivative_resolved"
    ]
    ill = np.diag([1.0, 1.0, 1e-7])
    perturbed = ill.copy()
    perturbed[0, 0] += 1e-6
    # Each column passes the loose RMS gate, but width conditioning does not.
    assert not reference_resolution(ill, perturbed, ill, ill, temperature)[
        "derivative_resolved"
    ]


def test_exact_nonlinear_ambiguity_is_preserved_by_restarts():
    from horizon_position_d5.inference import fit_multistart

    def model(p):
        return np.array([np.sin(p[0]), p[1], p[2]])

    def jac(p):
        return np.diag([np.cos(p[0]), 1.0, 1.0])

    injected = np.array([2.5, 0.4, -0.2])
    fit = fit_multistart(model, jac, model(injected), injected)
    zero, truth_start = fit["restarts"][:2]
    assert zero["success"] and truth_start["success"]
    assert zero["sum_squared_residual_k2"] < 1e-16
    assert truth_start["sum_squared_residual_k2"] < 1e-16
    assert abs(zero["delta_m"][0] - truth_start["delta_m"][0]) > 1
    assert fit["restart_position_spread_m"][0] > 1
