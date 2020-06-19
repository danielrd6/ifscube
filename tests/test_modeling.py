import pkg_resources
import pytest

from ifscube import onedspec, modeling

file_name = pkg_resources.resource_filename('ifscube', 'examples/ngc6300_nuc.fits')
spec = onedspec.Spectrum(file_name)


@pytest.mark.filterwarnings("ignore:RADECSYS", "ignore:'datfix'")
def test_simple_fit():
    fit = modeling.LineFit(spec, fitting_window=(6400.0, 6700.0), fit_continuum=True)
    fit.add_feature(name='n2_6548', rest_wavelength=6548.04, amplitude=1.0e-14, velocity=0.0, sigma=100.0)
    fit.add_feature(name='ha', rest_wavelength=6562.8, amplitude=1.0e-14, velocity=0.0, sigma=100.0)
    fit.add_feature(name='n2_6583', rest_wavelength=6583.46, amplitude=1.0e-14, velocity=0.0, sigma=100.0)
    fit.fit()
    fit.plot()
    assert 1


@pytest.mark.filterwarnings("ignore:RADECSYS", "ignore:'datfix'")
def test_skip_feature():
    fit = modeling.LineFit(spec, fitting_window=(6400.0, 6700.0), fit_continuum=True)
    fit.add_feature(name='hb', rest_wavelength=4861.0, amplitude=0, velocity=0, sigma=10)
    fit.add_feature(name='ha', rest_wavelength=6562.8, amplitude=1.0e-14, velocity=0.0, sigma=100.0)
    fit.fit()
    fit.plot()
    assert 1
