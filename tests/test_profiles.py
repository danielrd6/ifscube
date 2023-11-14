import numpy as np
from scipy.integrate import trapz

from ifscube import elprofile


def flux_conservation(fun):
    x = np.linspace(4000, 10000, 6000)
    rest_wl = 7000
    v = 10000
    p_blue = [1, -v, 3000]
    p_red = [1, v, 3000]
    if fun == elprofile.gausshermitevel:
        p_blue += [.2, .2]
        p_red += [.2, .2]

    flux_density_blue = fun(x, rest_wl, p_blue)
    flux_density_red = fun(x, rest_wl, p_red)

    flux_blue = trapz(flux_density_blue, x)
    flux_red = trapz(flux_density_red, x)

    assert ((flux_blue / flux_red) - 1.0) < 1e-8


def test_flux_conservation_gaussvel():
    flux_conservation(fun=elprofile.gaussvel)


def test_flux_conservation_gausshermitevel():
    flux_conservation(fun=elprofile.gausshermitevel)


def test_gauss():
    x = np.linspace(start=-10, stop=10, num=11)
    p = np.array([1.0, 0.0, 3.0, 2.0, 2.0, 1.0])
    y = elprofile.gauss(x=x, p=p)
    expected = 1.0 + (2.0 * np.exp(-(2.0 ** 2) / 2.0))
    assert y[5] == expected


def test_gauss_hermite_symmetry():
    x = np.linspace(start=-10, stop=10, num=11)
    p = np.array([1.0, 0.0, 3.0, 0.0, 0.2])
    y = elprofile.gauss_hermite(x=x, p=p)
    assert y[4] == y[6]


def test_gauss_hermite_asymmetry():
    x = np.linspace(start=-10, stop=10, num=11)
    p = np.array([1.0, 0.0, 3.0, 0.2, 0.0])
    y = elprofile.gauss_hermite(x=x, p=p)
    assert y[4] > y[6]
