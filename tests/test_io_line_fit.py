import pkg_resources
import warnings
from astropy import table

from ifscube.io import line_fit
from .test_modeling import simple_fit


def test_parse_config_table():
    file_name = pkg_resources.resource_filename('ifscube', 'examples/example_onedspec_linefit.fits')
    t = table.Table.read(file_name, 'FITCONFIG')
    line_fit.table_to_config(t)
    assert True


def test_write_fit_1d():
    fit = simple_fit()
    fit.fit_pseudo_continuum()
    fit.fit(verbose=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="tostring", category=DeprecationWarning)
        line_fit.write_spectrum_fit(fit, out_image='tests/test_write_fit_1d.fits', function='gaussian', overwrite=True)
    assert 1


def test_load_fit_1d():
    line_fit.load_fit('tests/test_write_fit_1d.fits')
    assert 1


def test_write_fit_3d():
    fit = simple_fit(function='gaussian', fit_type='cube', spiral_loop=True, spiral_center=(3, 4))
    fit.fit(verbose=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="tostring", category=DeprecationWarning)
        line_fit.write_spectrum_fit(fit, out_image='tests/test_write_fit_3d.fits', function='gaussian', overwrite=True)
    assert 1


def test_load_fit_3d():
    line_fit.load_fit('tests/test_write_fit_3d.fits')
    assert 1
