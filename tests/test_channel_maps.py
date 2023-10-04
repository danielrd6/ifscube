import pytest
import importlib.resources

from ifscube import channel_maps
from ifscube import Cube


def test_deprecation_warning():
    cube_file = str(importlib.resources.files("ifscube") / "examples/ngc3081_cube.fits")
    c = Cube(fname=cube_file)
    with pytest.warns(DeprecationWarning):
        channel_maps.channelmaps(cube=c, lambda0=6000.0, vel_min=-100.0, vel_max=100.0, screen=False)
    assert True
