from importlib import resources
import ifscube


def test_plotspec():
    cube_file = str(resources.files("ifscube") / "../../examples/ngc3081_cube.fits")
    cube = ifscube.Cube(cube_file)
    cube.plotspec(x=(2, 3), y=(4, 5), show=False)