"""EIGSEP experiment configuration."""

from pathlib import Path

import yaml

_DEFAULT_CONFIG = Path(__file__).parent / "configs" / "eigsep.yaml"


def _expand_range(value):
    """Expand a ``{start, stop, step}`` dict into a list of numbers.

    If *value* is already a list, return it unchanged.
    """
    if isinstance(value, list):
        return value
    start = value["start"]
    stop = value["stop"]
    step = value.get("step", 1)
    # Use integer arithmetic when all inputs are ints.
    if all(isinstance(v, int) for v in (start, stop, step)):
        return list(range(start, stop, step))
    n = int(round((stop - start) / step))
    return [start + i * step for i in range(n)]


def load_config(path=None):
    """Load EIGSEP configuration from a YAML file.

    Parameters
    ----------
    path : str or Path or None
        Path to a YAML config file. If None, loads the built-in
        default configuration.

    Returns
    -------
    config : dict
        Configuration dictionary.

    """
    if path is None:
        path = _DEFAULT_CONFIG
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Expand compact range notation.
    if "frequencies" in cfg and isinstance(cfg["frequencies"], dict):
        cfg["frequencies"] = _expand_range(cfg["frequencies"])
    ori = cfg.get("orientations", {})
    for key in ("elevations", "azimuths"):
        if key in ori and isinstance(ori[key], dict):
            ori[key] = _expand_range(ori[key])

    return cfg
