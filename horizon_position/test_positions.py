import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from positions import NOMINAL_ENU, build_positions  # noqa: E402


def test_count_and_nominal_first():
    pos = build_positions()
    assert len(pos) == 19
    assert pos[0][0] == "nominal"
    assert np.allclose(pos[0][1], NOMINAL_ENU)


def test_names_unique():
    names = [n for n, _ in build_positions()]
    assert len(set(names)) == 19


def test_shifts_apply_to_correct_axis():
    d = dict(build_positions())
    # x = East (index 0), y = North (1), z = Up (2)
    assert np.allclose(d["x_p_10"] - NOMINAL_ENU, [10.0, 0.0, 0.0])
    assert np.allclose(d["y_m_1"] - NOMINAL_ENU, [0.0, -1.0, 0.0])
    assert np.allclose(d["z_p_0p1"] - NOMINAL_ENU, [0.0, 0.0, 0.1])


def test_nominal_unmodified():
    # build_positions must not mutate the module constant
    build_positions()
    assert np.allclose(NOMINAL_ENU, [1648.0, 2024.0, 1796.0])
