"""The 19 antenna ENU positions for the position-sensitivity sweep.

Nominal position is eigsep_terrain site ``1P`` (114 m above the quarry
floor). Each of the three axes (x=East, y=North, z=Up) is perturbed one
at a time by +/-{0.1, 1, 10} m, giving 1 + 3*3*2 = 19 positions.
"""

import numpy as np

NOMINAL_ENU = np.array([1648.0, 2024.0, 1796.0])  # eigsep_terrain site 1P
SHIFTS_M = (0.1, 1.0, 10.0)
AXES = ("x", "y", "z")
_AXIS_IDX = {"x": 0, "y": 1, "z": 2}


def build_positions():
    """Return an ordered list of ``(name, enu)`` for the 19 positions.

    Order: nominal first, then for each axis (x, y, z) and magnitude
    (0.1, 1, 10) the minus shift then the plus shift. Names look like
    ``x_p_10`` (+10 m East) or ``z_m_0p1`` (-0.1 m Up).
    """
    out = [("nominal", NOMINAL_ENU.copy())]
    for axis in AXES:
        idx = _AXIS_IDX[axis]
        for mag in SHIFTS_M:
            for sign in (-1.0, +1.0):
                enu = NOMINAL_ENU.copy()
                enu[idx] += sign * mag
                sgn = "p" if sign > 0 else "m"
                mag_s = ("%g" % mag).replace(".", "p")
                out.append((f"{axis}_{sgn}_{mag_s}", enu))
    return out
