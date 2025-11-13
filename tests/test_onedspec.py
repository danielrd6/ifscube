import importlib.resources
import pytest

from src.ifscube import onedspec


def test_read_spec():
    file_name = importlib.resources.files('ifscube') / 'examples/ngc6300_nuc.fits'
    onedspec.Spectrum(file_name)
    assert 1


def test_accessory_data_error():
    file_name = importlib.resources.files('ifscube') / 'examples/ngc6300_nuc.fits'
    with pytest.raises(Exception) as e_info:
        # noinspection PyTypeChecker
        onedspec.Spectrum(file_name, variance=3.14)
    assert "Error reading Variance data" in e_info.value.args[0]
