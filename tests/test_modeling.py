import pkg_resources
import pytest

from ifscube import onedspec, modeling

file_name = pkg_resources.resource_filename('ifscube', 'examples/ngc6300_nuc.fits')
spec = onedspec.Spectrum(file_name)


def test_simple_fit():
    fit = modeling.LineFit(spec, fitting_window=(6400.0, 6700.0), fit_continuum=True)
    fit.add_feature(name='n2_6548', rest_wavelength=6548.04, amplitude=1.0e-14, velocity=0.0, sigma=100.0)
    fit.add_feature(name='ha', rest_wavelength=6562.8, amplitude=1.0e-14, velocity=0.0, sigma=100.0)
    fit.add_feature(name='n2_6583', rest_wavelength=6583.46, amplitude=1.0e-14, velocity=0.0, sigma=100.0)
    fit.fit(verbose=True)
    assert 1


def test_skip_feature():
    fit = modeling.LineFit(spec, fitting_window=(6400.0, 6700.0), fit_continuum=True)
    with pytest.warns(UserWarning):
        fit.add_feature(name='hb', rest_wavelength=4861.0, amplitude=0, velocity=0, sigma=10)
    fit.add_feature(name='ha', rest_wavelength=6562.8, amplitude=1.0e-14, velocity=0.0, sigma=100.0)
    fit.fit()
    assert 1


def test_bounds():
    fit = modeling.LineFit(spec, fitting_window=(6400.0, 6700.0), fit_continuum=True)
    fit.add_feature(name='ha', rest_wavelength=6562.8, amplitude=1.0e-14, velocity=0.0, sigma=100.0)
    fit.set_bounds('ha', 'velocity', [-50, 50])
    fit.set_bounds('ha', 'amplitude', [None, 0.4e-14])
    fit.fit()
    assert 1


def test_constraints():
    fit = modeling.LineFit(spec, function='gauss_hermite', fitting_window=(6400.0, 6700.0), fit_continuum=True)
    fit.add_feature(name='n2_6548', rest_wavelength=6548.04, amplitude=1.0e-14, velocity=0.0, sigma=100.0)
    fit.add_feature(name='ha', rest_wavelength=6562.8, amplitude=1.0e-14, velocity=0.0, sigma=100.0)
    fit.add_feature(name='n2_6583', rest_wavelength=6583.46, amplitude=1.0e-14, velocity=0.0, sigma=100.0)
    fit.add_minimize_constraint('n2_6548.sigma', 'n2_6583')
    fit.add_minimize_constraint('n2_6548.velocity', 'n2_6583')
    fit.add_minimize_constraint('n2_6548.amplitude', 'n2_6583 / 3.06')
    fit.add_minimize_constraint('n2_6548.amplitude', '< ha')
    fit.fit()
    assert 1


def test_gauss_hermite():
    fit = modeling.LineFit(spec, function='gauss_hermite', fitting_window=(6400.0, 6700.0), fit_continuum=True)
    fit.add_feature(name='n2_6548', rest_wavelength=6548.04, amplitude=1.0e-14, velocity=0.0, sigma=100.0)
    fit.add_feature(name='ha', rest_wavelength=6562.8, amplitude=1.0e-14, velocity=0.0, sigma=100.0)
    fit.add_feature(name='n2_6583', rest_wavelength=6583.46, amplitude=1.0e-14, velocity=0.0, sigma=100.0)
    fit.add_minimize_constraint(parameter='n2_6548.h_3', expression='n2_6583')
    fit.add_minimize_constraint(parameter='n2_6548.h_4', expression='n2_6583')
    fit.fit()
    assert 1
