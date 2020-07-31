import pkg_resources

from astropy import table
from ifscube.io import line_fit

from .test_modeling import simple_fit


def test_parse_config_table():
    file_name = pkg_resources.resource_filename('ifscube', 'examples/example_onedspec_linefit.fits')
    t = table.Table.read(file_name, 'FITCONFIG')
    line_fit.table_to_config(t)
    assert True


def test_load_fit():
    file_name = pkg_resources.resource_filename('ifscube', 'examples/example_onedspec_linefit.fits')
    line_fit.load_fit(file_name)
    assert 1


def test_write_fit_1d():
    fit = simple_fit()
    fit.fit(verbose=True, fit_continuum=True)
    line_fit.write_spectrum_fit(fit, out_image='tests/test_write_fit_1d.fits', function='gaussian', overwrite=True)
    assert 1
