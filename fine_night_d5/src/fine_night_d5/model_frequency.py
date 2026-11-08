"""Explicit model frequencies; measured channels are never relabelled.

Only the C53 nominal-centre control is supported. The default is the exact
native-frequency operator, with no additional output fields or reduction.
"""

import numpy as np

from .streaming import reduce_columns

CENTRES = np.array([288, 608, 848])
OFFSETS = np.array([-5, -4, -3, 3, 4, 5])
CONTROL_CHANNELS = (CENTRES[:, None] + OFFSETS).ravel()
PAYLOAD_FIELDS = {
    "model_evaluation_frequencies_mhz",
    "data_to_model_frequency_index",
    "model_frequency_operator_version",
}


class FrequencyPlan:
    """Resolve an explicit18→3 control or the unchanged native operator.

    Frequencies passed to beam/GSM/convolution are unique evaluation
    coordinates. Expansion returns the original18 data columns before
    masking, so each data channel retains its own count and fitted freedom.
    """

    def __init__(self, channels, declaration=None):
        channels = np.asarray(channels)
        if (
            channels.ndim != 1
            or channels.dtype.kind not in "iu"
            or not len(channels)
            or np.any(np.diff(channels) <= 0)
        ):
            raise ValueError("unique increasing exact data channel IDs required")
        self.channels = channels.copy()
        self.data_frequencies = channels * 250 / 1024
        self.evaluation_frequencies = self.data_frequencies.copy()
        self.mapping = np.arange(len(channels), dtype=np.int64)
        self.control = declaration is not None
        if self.control:
            fields = {
                "version",
                "kind",
                "data_channels",
                "evaluation_frequencies_mhz",
                "data_to_evaluation",
            }
            if (
                not isinstance(declaration, dict)
                or set(declaration) != fields
                or type(declaration["version"]) is not int
                or declaration["version"] != 1
                or declaration["kind"] != "c53-nominal-centres"
            ):
                raise ValueError("explicit version1 C53 model-frequency map required")
            axis = np.asarray(declaration["data_channels"])
            f = np.asarray(declaration["evaluation_frequencies_mhz"], dtype=float)
            mapping = np.asarray(declaration["data_to_evaluation"])
            if (
                axis.dtype.kind not in "iu"
                or not np.array_equal(axis, channels)
                or not np.array_equal(channels, CONTROL_CHANNELS)
                or f.shape != (3,)
                or not np.isfinite(f).all()
                or not np.array_equal(f, CENTRES * 250 / 1024)
                or mapping.dtype.kind not in "iu"
                or not np.array_equal(mapping, np.repeat(np.arange(3), 6))
            ):
                raise ValueError("ambiguous/nonfinite or noncanonical18-to3 map")
            self.evaluation_frequencies = f.copy()
            self.mapping = mapping.copy()
        for value in (
            self.channels,
            self.data_frequencies,
            self.evaluation_frequencies,
            self.mapping,
        ):
            value.flags.writeable = False

    def identity(self):
        """Return control-only provenance, distinct from measured frequency."""
        if not self.control:
            return None
        return {
            "version": 1,
            "kind": "c53-nominal-centres",
            "data_channels": self.channels.tolist(),
            "data_frequencies_mhz": self.data_frequencies.tolist(),
            "evaluation_frequencies_mhz": self.evaluation_frequencies.tolist(),
            "data_to_evaluation": self.mapping.tolist(),
            "expansion": "native rows before actual channel-mask count mean",
        }

    def payload(self):
        """Preserve legacy output keys when the control is absent."""
        if not self.control:
            return {}
        return {
            "model_evaluation_frequencies_mhz": self.evaluation_frequencies,
            "data_to_model_frequency_index": self.mapping,
            "model_frequency_operator_version": np.array(1, dtype=np.int64),
        }

    def expand_and_reduce(self, values, accepted, mapping, n_bins):
        """Expand repeated model columns before the existing0/1 reduction."""
        values = np.asarray(values)
        if (
            values.ndim != 3
            or values.shape[1] != len(self.evaluation_frequencies)
            or not np.isfinite(values).all()
        ):
            raise ValueError("finite evaluation-frequency model columns required")
        columns = values[:, self.mapping] if self.control else values
        means, counts = reduce_columns(columns, accepted, mapping, n_bins)
        return columns, means, counts


def from_config(config):
    """No implicit null/fallback; an opt-in key requires its full map."""
    if (
        "model_frequency_operator" in config
        and config["model_frequency_operator"] is None
    ):
        raise ValueError("model-frequency control cannot be null")
    return FrequencyPlan(config["channels"], config.get("model_frequency_operator"))


def verify_payload(config, record, arrays):
    """Reject cache/context reuse with changed or erased model coordinates."""
    plan = from_config(config)
    present = PAYLOAD_FIELDS & set(arrays)
    if not np.array_equal(arrays["frequencies_mhz"], plan.data_frequencies):
        raise ValueError("payload measured data frequencies differ")
    if record.get("model_frequency_operator") != plan.identity():
        raise ValueError("record model-frequency identity differs from inputs")
    if present != (PAYLOAD_FIELDS if plan.control else set()):
        raise ValueError("missing/unexpected model-frequency payload fields")
    for name, value in plan.payload().items():
        if arrays[name].dtype != value.dtype or not np.array_equal(arrays[name], value):
            raise ValueError("payload model-frequency coordinates differ")
