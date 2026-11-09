import copy

import pytest
from fine_night_d5.prepare_field_operands import DELETIONS, role_penalties


def ensemble():
    return [
        dict(role="raster_hybrid"),
        dict(role="represented_hfss_control", represented_prior=True),
        dict(role="penalty_0.01"),
    ] + [
        dict(role="delete_stripe_" + str(k), deleted_training_stripe=k)
        for k in DELETIONS
    ]


def test_prior_cannot_replace_a_fitted_deletion_and_mapping_is_explicit():
    fits = ensemble()
    assert role_penalties(fits, 0.001, 0.01)["delete_stripe_9"] == 0.001
    bad = copy.deepcopy(fits)
    bad[3]["represented_prior"] = True
    with pytest.raises(ValueError, match="only represented"):
        role_penalties(bad, 0.001, 0.01)
    bad = copy.deepcopy(fits)
    bad[3]["deleted_training_stripe"] = 10
    with pytest.raises(ValueError, match="role/deletion"):
        role_penalties(bad, 0.001, 0.01)
    with pytest.raises(ValueError, match="exact main"):
        role_penalties(fits[:-1], 0.001, 0.01)
