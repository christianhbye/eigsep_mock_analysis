"""eigsim: simulation code for the EIGSEP experiment."""

import os
from importlib.metadata import version

os.environ.setdefault("JAX_ENABLE_X64", "1")

__version__ = version("eigsim")

from .config import load_config
from .data import load_beam, load_horizon
from .noise import radiometer_noise
from .rotations import (
    beam_to_alm,
    drive_rotation_matrix,
    rotate_alm_to_beam,
    rotate_beam_data,
)
from .simulate import make_beam, precompute_sky_alm, simulate

__all__ = [
    "beam_to_alm",
    "drive_rotation_matrix",
    "load_beam",
    "load_config",
    "load_horizon",
    "make_beam",
    "precompute_sky_alm",
    "radiometer_noise",
    "rotate_alm_to_beam",
    "rotate_beam_data",
    "simulate",
]
