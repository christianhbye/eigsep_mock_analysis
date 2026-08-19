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
print("OK")
