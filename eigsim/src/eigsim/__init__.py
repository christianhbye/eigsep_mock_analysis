"""eigsim: simulation code for the EIGSEP experiment."""

import os
from importlib.metadata import version

os.environ.setdefault("JAX_ENABLE_X64", "1")

__version__ = version("eigsim")

from .config import load_config
from .data import load_beam, load_horizon, load_horizon_curve
from .horizon import d5_open_sky_weight, mwss_grid, open_sky_weight
from .noise import radiometer_noise
from .position import PositionHorizonGrid
from .rotations import (
    beam_to_alm,
    drive_rotation_matrix,
    rotate_alm_to_beam,
    rotate_beam_data,
)
from .simulate import (
    compute_fgnd,
    correct_ground_loss,
    make_beam,
    precompute_sky_alm,
    simulate,
    simulate_path,
)

__all__ = [
    "PositionHorizonGrid",
    "beam_to_alm",
    "compute_fgnd",
    "correct_ground_loss",
    "d5_open_sky_weight",
    "drive_rotation_matrix",
    "load_beam",
    "load_config",
    "load_horizon",
    "load_horizon_curve",
    "make_beam",
    "mwss_grid",
    "open_sky_weight",
    "precompute_sky_alm",
    "radiometer_noise",
    "rotate_alm_to_beam",
    "rotate_beam_data",
    "simulate",
    "simulate_path",
]
