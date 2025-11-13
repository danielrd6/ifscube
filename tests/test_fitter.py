import importlib.resources

import numpy as np
import pytest

from src.ifscube import parser
from src.ifscube.fitter import spectrum_fit
from src.ifscube.onedspec import Spectrum


def basic_1d_fit():
    config = importlib.resources.files("ifscube") / "examples/halpha_gauss.cfg"
    input_data = importlib.resources.files("ifscube") / "examples/manga_onedspec.fits"
    data = Spectrum(fname=str(input_data), scidata="F_OBS", primary="PRIMARY")

    c = parser.LineFitParser(str(config))
    line_fit_args = c.get_vars()

    return data, line_fit_args


@pytest.mark.filterwarnings("ignore:Spectral feature", "ignore:Parameter")
def test_wavelength_window_low():
    data, line_fit_args = basic_1d_fit()
    max_wl = np.max(data.rest_wavelength)

    line_fit_args["fitting_window"] = (max_wl * 1.5, max_wl * 2.0)
    line_fit_args["optimize_fit"] = False

    with pytest.raises(AssertionError, match="Lower limit of fitting window above maximum"):
        spectrum_fit(data, **line_fit_args)


@pytest.mark.filterwarnings("ignore:Spectral feature", "ignore:Parameter")
def test_wavelength_window_high():
    data, line_fit_args = basic_1d_fit()
    min_wl = np.min(data.rest_wavelength)

    line_fit_args["fitting_window"] = (min_wl * 0.5, min_wl * 0.8)
    line_fit_args["optimize_fit"] = False

    with pytest.raises(AssertionError, match="Upper limit of fitting window below minimum"):
        spectrum_fit(data, **line_fit_args)
