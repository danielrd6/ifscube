import numpy as np
from astropy import units

from ifscube import spectools
from ifscube.profiles import gauss_vel


def test_find_intermediary_value():
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([10, 20, 30, 40, 50])
    xv = spectools.find_intermediary_value(x, y, 31)
    assert xv == 2.1


def test_velocity_width_oversample():
    rest_wavelength = units.Quantity(value=6500, unit='angstrom')
    rwl = np.array([rest_wavelength.value])
    p = np.array([1.0, -5000.0, 2000.0])

    wavelength = np.linspace(start=5000, stop=8000, num=15000)
    g = gauss_vel(x=wavelength, rest_wl=rwl, p=p)
    obs = np.random.normal(g, scale=.05)

    vw_natural = spectools.velocity_width(
        wavelength=wavelength, model=g, data=obs, rest_wavelength=rest_wavelength, oversample=1,
        fractional_pixels=True)

    wavelength = np.linspace(start=5000, stop=8000, num=100)
    g = gauss_vel(x=wavelength, rest_wl=rwl, p=p)
    obs = np.random.normal(g, scale=.05)
    vw_oversampled = spectools.velocity_width(
        wavelength=wavelength, model=g, data=obs, rest_wavelength=rest_wavelength, oversample=15,
        fractional_pixels=True)

    assert ((vw_oversampled['model_velocity_width'] / vw_natural['model_velocity_width']) - 1.0) < 1e-3
