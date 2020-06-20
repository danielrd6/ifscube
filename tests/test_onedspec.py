import numpy as np
import pkg_resources
import pytest

from ifscube import onedspec


def test_read_spec():
    file_name = pkg_resources.resource_filename('ifscube', 'examples/ngc6300_nuc.fits')
    onedspec.Spectrum(file_name)
    assert 1


def line_fitting(f_bounds: bool = False, f_constraints: bool = False, f_monte_carlo: bool = False):
    file_name = pkg_resources.resource_filename('ifscube', 'examples/ngc6300_nuc.fits')
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
    if f_bounds:
        for i in range(n_components):
            # amplitude
            b += [[0.0, 1e-12]]
            # velocity
            b += [[-300.0, +300.0]]
            # sigma
            b += [[40.0, 500.0]]

    # Setting the constraints

    c = []
    if f_constraints:
        c = [
            # Keeping the same doppler shift on all lines
            {'type': 'eq', 'fun': lambda x: x[1] - x[4]},
            {'type': 'eq', 'fun': lambda x: x[4] - x[7]},

            # And the same goes for the sigmas
            {'type': 'eq', 'fun': lambda x: x[2] - x[5]},
            {'type': 'eq', 'fun': lambda x: x[5] - x[8]},
        ]

    # Creating a fake variance spectrum with signal-to-noise = 20.
    my_spec = onedspec.Spectrum(file_name)
    my_spec.variance = (my_spec.data / 10) ** 2

    monte_carlo = 10 if f_monte_carlo else 0
    my_spec.linefit(
        p0, fitting_window=(6500.0, 6700.0), feature_wl=lines_wl, function='gaussian', constraints=c, bounds=b,
        fit_continuum=True, write_fits=True, overwrite=True, monte_carlo=monte_carlo,
        continuum_line_weight=0.0)

    return my_spec


def test_simple_fit():
    line_fitting()
    assert 1


def test_monte_carlo_fit():
    line_fitting(f_monte_carlo=True)
    assert 1


def test_fit_with_bounds():
    line_fitting(f_bounds=True)
    assert 1


def test_fit_with_constraints():
    line_fitting(f_constraints=True)
    assert 1


def test_fit_with_constraints_and_bounds():
    line_fitting(f_bounds=True, f_constraints=True)
    assert 1
