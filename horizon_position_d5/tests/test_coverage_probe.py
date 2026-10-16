"""GeoTIFF conventions must fail closed before a mosaic is built."""

import numpy as np
import pytest
from PIL import Image, TiffImagePlugin

from horizon_position_d5.coverage_probe import read_tile


def test_tile_orientation_coordinates_and_nodata(tmp_path):
    a = np.zeros((2000, 2000), dtype=np.float32)
    a[0, 0], a[-1, 0] = 123.0, 456.0
    tags = TiffImagePlugin.ImageFileDirectory_v2()
    tags[33550] = (0.5, 0.5, 0.0)
    tags[33922] = (0.0, 0.0, 0.0, 290000.0, 4344000.0, 0.0)
    tags[34735] = (1, 1, 0, 2, 3072, 0, 1, 6341, 1025, 0, 1, 1)
    path = tmp_path / "tile.tif"
    Image.fromarray(a).save(path, tiffinfo=tags)
    result = read_tile(path, 90, 43)
    assert result[0, 0] == 456.0
    assert result[-1, 0] == 123.0
    with pytest.raises(ValueError, match="georeferencing"):
        read_tile(path, 91, 43)
    tags[42113] = "123"
    Image.fromarray(a).save(path, tiffinfo=tags)
    with pytest.raises(ValueError, match="nodata"):
        read_tile(path, 90, 43)


@pytest.mark.parametrize("levels", [(), (0,), (2, 2), (21,), (2.5,)])
def test_invalid_extensions_do_not_write(tmp_path, levels):
    from horizon_position_d5.coverage_probe import build

    with pytest.raises(ValueError, match="extensions"):
        build(tmp_path / "missing.npz", tmp_path / "output", levels)
    assert not (tmp_path / "output").exists()
