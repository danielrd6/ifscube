import importlib.resources

from ifscube import Cube
from ifscube import ppxf_wrapper


def test_mask():
    file_name = importlib.resources.files('ifscube') / 'examples/ngc3081_cube.fits'
    cube = Cube(file_name, redshift=0.00798)

    mask = [[5860, 5920], [6290, 6320], [6360, 6390], [6510, 6610], [6700, 6745]]

    idx = (3, 3)
    p_no_mask = ppxf_wrapper.cube_kinematics(cube, fitting_window=(5800, 6800), individual_spec=idx, deg=-1)
    p_mask = ppxf_wrapper.cube_kinematics(cube, fitting_window=(5800, 6800), individual_spec=idx, deg=-1, mask=mask)

    assert (p_no_mask.stellar - p_mask.stellar)[:, idx].mean() > 0
