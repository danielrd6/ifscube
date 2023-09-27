import importlib.resources
import subprocess


def test_specfit():
    config = importlib.resources.files("ifscube") / "examples/halpha_gauss.cfg"
    input_data = importlib.resources.files("ifscube") / "examples/manga_onedspec.fits"
    subprocess.run(["specfit", "-o", "-c", config, input_data])
    assert 1


def test_cubefit():
    config = importlib.resources.files("ifscube") / "examples/halpha_cube.cfg"
    input_data = importlib.resources.files("ifscube") / "examples/ngc3081_cube.fits"
    subprocess.run(["cubefit", "-o", "-c", config, input_data])
    assert 1
