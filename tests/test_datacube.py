import pkg_resources
from ifscube import Cube
import numpy as np


def fit_setup():
    lines_wl = np.array([
        6548.04,  # [N II] 6548
        6562.80,  # H alpha
        6583.46,  # [N II] 6583
    ])

    # Defining the initial guess for the parameters.
    # five parameters for the gauss hermite polynomials
    # amplitude, velocity, sigma, h3, h4

    n_components = 3
    n_pars = 3
    p0 = np.zeros(n_components * n_pars)

    p0[0::n_pars] = 1e-14  # flux
    p0[1::n_pars] = 0  # velocity
    p0[2::n_pars] = 100  # sigma

    # Setting bounds

    b = []
    for i in range(n_components):
        # amplitude
        b += [[0.0, 1e-12]]
        # velocity
        b += [[-300.0, +300.0]]
        # sigma
        b += [[40.0, 500.0]]

    # Setting the constraints

    c = [
        # Keeping the same doppler shift on all lines
        {'type': 'eq', 'fun': lambda x: x[1] - x[4]},
        {'type': 'eq', 'fun': lambda x: x[4] - x[7]},

        # And the same goes for the sigmas
        {'type': 'eq', 'fun': lambda x: x[2] - x[5]},
        {'type': 'eq', 'fun': lambda x: x[5] - x[8]},
    ]

    return lines_wl, p0, b, c


def test_cube_w80():
    file_name = pkg_resources.resource_filename('ifscube', 'examples/ngc3081_cube.fits')
    lines_wl, p0, b, c = fit_setup()

    # Creating a fake variance spectrum with signal-to-noise = 20.
    my_cube = Cube(file_name)
    my_cube.variance = (my_cube.data / 10) ** 2

    my_cube.linefit(
        p0, fitting_window=(6500.0, 6700.0), feature_wl=lines_wl, function='gaussian', constraints=c, bounds=b,
        fit_continuum=True, write_fits=True, overwrite=True, continuum_line_weight=0.0)

    my_cube.w80(1)

    assert 1
