"""Structural check of a generate.py output. Used on smoke and production runs."""

import json
import sys

import numpy as np

d = np.load(sys.argv[1], allow_pickle=False)
print("keys:", sorted(d.keys()))
header = json.loads(str(d["provenance"]))
print("n_models:", d["T21_mK"].shape, "freqs:", d["freqs_MHz"].shape)
print("header keys:", sorted(header))
print("zeus21 version:", header["packages"]["zeus21"])
print("generator source chars:", len(str(d["generator_source"])))
print("env lock chars:", len(str(d["env_lock"])))
assert d["T21_mK"].shape[1] == 201
assert np.isfinite(d["T21_mK"]).all()
assert len(str(d["generator_source"])) > 1000
assert len(str(d["env_lock"])) > 100
assert header["selection"]["applied"] is False

# Detection half of the checkpoint-identity invariant (generate.py's
# _check_index_complete is the prevention half): if a work directory were
# ever reused across a different --seed/--batch-size, kept_index could
# carry duplicates or drop out of order while every shape check below
# still passes -- rebuild_params(header)[kept_index] == params would even
# still hold, since kept_index is derived from the same corrupted index.
# This is the structural gate that would still catch it on the npz alone.
kept_index = d["kept_index"]
assert np.array_equal(kept_index, np.unique(kept_index)), (
    "kept_index must be strictly increasing with no duplicates"
)
assert (
    d["params"].shape[0] == d["T21_mK"].shape[0] == d["xHI"].shape[0] == kept_index.size
)
assert d["z_native"].size == d["T21_native_mK"].shape[1]
print("OK")
