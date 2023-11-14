import numpy as np
from astropy import units
from ifscube.elprofile import gauss_vel

from ifscube import spectools


def test_find_intermediary_value():
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([10, 20, 30, 40, 50])
    xv = spectools.find_intermediary_value(x, y, 31)
    assert xv == 2.1


def test_velocity_width_oversample():
    rest_wavelength = units.Quantity(6500, 'angstrom')
    p = np.array([1, -5000, 2000])

    wavelength = np.linspace(5000, 8000, 15000)
    g = gauss_vel(wavelength, rest_wavelength.value, p)
    obs = np.random.normal(g, .05)

    vw_natural = spectools.velocity_width(
        wavelength=wavelength, model=g, data=obs, rest_wavelength=rest_wavelength, oversample=1,
        fractional_pixels=True)

    wavelength = np.linspace(5000, 8000, 100)
    g = gauss_vel(wavelength, rest_wavelength.value, p)
    obs = np.random.normal(g, .05)
    vw_oversampled = spectools.velocity_width(
        wavelength=wavelength, model=g, data=obs, rest_wavelength=rest_wavelength, oversample=15,
        fractional_pixels=True)

    assert ((vw_oversampled['model_velocity_width'] / vw_natural['model_velocity_width']) - 1.0) < 1e-3
