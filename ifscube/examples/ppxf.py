# This module is a wrapper for the ppxf package available in PyPI
from pkg_resources import resource_filename
from astropy import wcs
from astropy.io import fits
import numpy as np
import glob
import ppxf


class Fit(object):

    def __init__(self, fitting_window=None, mask=None):

        self.base = np.array([])
        self.base_wavelength = np.array([])
        self.base_delta = 1.0

        self.load_miles_models()

    def load_miles_models(self):

        path = resource_filename('ppxf', 'miles_models')
        base_files = glob.glob(path + '*.fits')
        base_files.sort()

        w = wcs.WCS(base_files[0], naxis=1)
        spectrum = fits.getdata(base_files[0])
        wavelength = w.wcs_pix2world(np.arange(spectrum.size), 0)

        base = spectrum.reshape((1, spectrum.size))

        for file in base_files:
            base = np.row_stack([base, fits.getdata(file)])

        self.base = base
        self.base_wavelength = wavelength

        if np.unique(np.diff(self.base_wavelength)).size != 1:
            raise UserWarning('Base is not evenly sampled in wavelength.')

        self.base_delta = np.mean(np.diff(self.base_wavelength))
