import importlib.resources
import os
import subprocess


def test_specfit():
    config = importlib.resources.files("ifscube") / "examples/halpha_gauss.cfg"
    input_data = importlib.resources.files("ifscube") / "examples/manga_onedspec.fits"
    script = os.path.abspath(os.path.join(str(importlib.resources.files("ifscube")), "..", "bin", "specfit"))
    subprocess.run(["python", script, "-o", "-c", config, input_data])
    assert 1


def test_cubefit():
    config = importlib.resources.files("ifscube") / "examples/halpha_cube.cfg"
    input_data = importlib.resources.files("ifscube") / "examples/ngc3081_cube.fits"
    script = os.path.abspath(os.path.join(str(importlib.resources.files("ifscube")), "..", "bin", "cubefit"))
    subprocess.run(["python", script, "-o", "-c", config, input_data])
    assert 1
