import numpy as np
import pytest
from astropy.io import fits

from ifscube import onedspec


@pytest.fixture()
def one_dimensional_spectrum(tmp_path):
    rng = np.random.default_rng(0)
    flux = rng.normal(size=1000, scale=1.0).astype(np.float32)
    var = np.abs(flux / 10) ** 2
    flag = (flux > 2.0).astype(np.int16)

    header = fits.Header()
    header['NAXIS'] = 1
    header['CRPIX1'] = 0
    header['CRVAL1'] = 5000.0
    header['CD1_1'] = 1.0
    header['CTYPE1'] = "WAVE"
    header['CUNIT1'] = "Angstrom"

    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(flux, header, name="SCI"),
        fits.ImageHDU(var, header, name="VAR"),
        fits.ImageHDU(flag, header, name="FLAG"),
    ])
    path = tmp_path / "test.fits"
    hdul.writeto(path)

    return path


def test_read_spec(one_dimensional_spectrum):
    onedspec.Spectrum(one_dimensional_spectrum, scidata='SCI', variance='VAR', flags='FLAG')
    assert 1


def test_accessory_data_error(one_dimensional_spectrum):
    with pytest.raises(Exception) as e_info:
        onedspec.Spectrum(one_dimensional_spectrum, variance=3.14)
    assert "Error reading Variance data" in e_info.value.args[0]
