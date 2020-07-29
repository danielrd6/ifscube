import pkg_resources

from ifscube import onedspec


def test_read_spec():
    file_name = pkg_resources.resource_filename('ifscube', 'examples/ngc6300_nuc.fits')
    onedspec.Spectrum(file_name)
    assert 1
