"""Parameter space for the Zeus21 global 21 cm ensemble.

Pure module: imports numpy/scipy only, never zeus21, so these tests run
in the mock_analysis env rather than the pinned generator env.

Bounds are in *transformed* units. For ``transform == "log10"`` they
bracket log10 of the value handed to Zeus21, so a range of -2.0 .. 2.0
on ``L40_xray`` means four decades of X-ray luminosity.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class Param:
    """One varied Zeus21 ``Astro_Parameters`` keyword."""

    name: str
    transform: str
    lo: float
    hi: float

    def to_native(self, x):
        """Transformed value -> the value Zeus21 expects."""
        if self.transform == "log10":
            return 10.0**x
        if self.transform == "linear":
            return x
        raise ValueError(f"unknown transform {self.transform!r}")


# Broad, deliberately agnostic priors: the figure is a stress test of the
# foreground filter, not a forecast, so ranges span weak-to-extreme signals
# rather than tracking what current observations allow.
PARAMS = (
    # Pop II star formation
    Param("epsstar", "log10", -2.5, -0.5),
    Param("alphastar", "linear", 0.0, 1.0),
    Param("dlog10epsstardz", "linear", -0.5, 0.5),
    # Ionizing escape -> reionization timing -> the 130-250 MHz end
    Param("fesc10", "log10", -2.5, 0.0),
    Param("alphaesc", "linear", -1.0, 1.0),
    # X-ray heating -> trough depth and the emission feature
    Param("L40_xray", "log10", -2.0, 2.0),
    Param("alpha_xray", "linear", -2.0, 0.0),
    Param("E0_xray", "log10", 2.0, 3.2),
    # Pop III
    Param("fstar_III", "log10", -4.0, -1.5),
    Param("Mc_III", "log10", 5.5, 8.0),
    Param("L40_xray_III", "log10", -2.0, 2.0),
    Param("fesc7_III", "log10", -2.5, -0.5),
    # Lyman-Werner feedback (A_LW = 0 switches suppression off)
    Param("A_LW", "linear", 0.0, 4.0),
    Param("beta_LW", "linear", 0.3, 1.0),
)

PARAM_NAMES = tuple(p.name for p in PARAMS)


def sample(m, seed, params=PARAMS):
    """Scrambled Sobol draw of shape ``(2**m, len(params))``.

    Sobol rather than a Latin hypercube because it extends in powers of
    two: 4096 -> 8192 keeps every existing model.
    """
    lo = np.array([p.lo for p in params])
    hi = np.array([p.hi for p in params])
    unit = qmc.Sobol(d=len(params), scramble=True, seed=seed).random_base2(m)
    return lo + unit * (hi - lo)


def to_astro_kwargs(row, params=PARAMS):
    """One sampled row -> ``Astro_Parameters`` keyword arguments."""
    return {p.name: p.to_native(x) for p, x in zip(params, row, strict=True)}
