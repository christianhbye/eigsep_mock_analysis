"""Consume frozen actual-frequency field operands without basis changes."""

import json

import numpy as np

from .assets import checked_file, sha256
from .native_field import harmonic_power


class FieldOperands:
    """Numeric producer-pinned grids and coefficients, with native closure."""

    def __init__(self, pin):
        done_path = checked_file(pin)
        self.record = json.loads(done_path.read_text())
        directory = done_path.parent
        for name, digest in self.record["output_hashes"].items():
            if sha256(directory / name) != digest:
                raise ValueError("frozen field operand changed")
        with np.load(directory / "coefficients.npz", allow_pickle=False) as z:
            self.frequencies = z["frequencies_mhz"]
            self.roles = z["roles"].astype(str).tolist()
            self.coefficients = z["frequency_coefficients"]
            self.field_lmax = int(z["lmax"])
        self.angular = {}
        for L in [129, 180]:
            with np.load(directory / f"angular_L{L}.npz", allow_pickle=False) as z:
                if int(z["L"]) != L or int(z["field_lmax"]) != self.field_lmax:
                    raise ValueError("angular field operand grid/order changed")
                self.angular[L] = {
                    key: z[key] for key in ["Y", "directions", "weights"]
                }
        if (
            self.roles != self.record["roles"]
            or self.coefficients.shape
            != (len(self.roles), len(self.frequencies), 3 * (self.field_lmax + 1) ** 2)
            or np.any(np.diff(self.frequencies) <= 0)
        ):
            raise ValueError("frozen field frequency/role/layout changed")
        self.done_sha256 = pin["sha256"]

    def beam(self, role, frequencies):
        """Evaluate only exact frozen frequencies; retain normalization gauge."""
        f = np.asarray(frequencies)
        idx = np.searchsorted(self.frequencies, f)
        if (
            f.ndim != 1
            or not len(f)
            or np.any(idx >= len(self.frequencies))
            or not np.array_equal(self.frequencies[idx], f)
            or np.any(np.diff(f) <= 0)
        ):
            raise ValueError("actual requested field frequencies absent or ambiguous")
        if role not in self.roles:
            raise ValueError("unknown declared beam role")
        coeff = self.coefficients[self.roles.index(role), idx]
        results = {}
        for L in [129, 180]:
            arrays = self.angular[L]
            results[L] = harmonic_power(
                coeff, arrays["Y"], arrays["directions"], arrays["weights"]
            )
        power, norm = results[129]
        native_norm = results[180][1]
        np.testing.assert_allclose(norm, native_norm, rtol=1e-11, atol=1e-12)
        return power.reshape(len(f), 130, 258), dict(
            role=role,
            frequencies_mhz=f,
            transverse_mean_power=norm,
            native_transverse_mean_power=native_norm,
            native_output_norm_relative_error=abs(norm / native_norm - 1),
        )
