"""EIGSEP experiment configuration."""

from pathlib import Path

import yaml

_DEFAULT_CONFIG = Path(__file__).parent / "configs" / "eigsep.yaml"


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
        return yaml.safe_load(f)
