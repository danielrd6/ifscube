import warnings
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from astropy import wcs
from astropy.io import fits


class Spectrum:
    def __init__(self, fname: str = None, scidata: Union[str, int] = 'SCI', variance: Union[str, int] = None,
                 flags: Union[str, int] = None, stellar: Union[str, int] = None, primary: Union[str, int] = 'PRIMARY',
                 redshift: float = None, wcs_axis: int = None, wavelength: Union[str, int] = None) -> None:
        """
        Base class for 1D spectra.

        Parameters
        ----------
        fname : str
            Name of the input FITS file.
        scidata : st
            Science data extension name.
        variance : str
            Variance extension name.
        flags : str
            Flag extension name.
        stellar : str
            Stellar (or synthetic) spectra extension name.
        wavelength : str
            Wavelength extension name. Overrides WCS from header.
        primary : str
            Primary extension name, normally containing only a header.
        redshift : float
            Redshift of the source given as z.
        wcs_axis : list
            Number of the WCS axis which represents the wavelength.
        """

        self.header = None
        self.ppxf_sol = np.ndarray([])

        if fname is not None:
            arg_names = ['fname', 'scidata', 'variance', 'flags', 'stellar',
                         'primary', 'redshift', 'wcs_axis', 'wavelength']

            locale = locals()
            load_args = {i: locale[i] for i in arg_names}
            self._load(**load_args)

    def _accessory_data(self, hdu, variance, flags, stellar):

        def shape_error(name):
            s = '{:s} spectrum must have the same shape of the spectrum itself'
            return s.format(name)

        self.variance = np.ones_like(self.data)
        self.flags = np.zeros_like(self.data)
        self.stellar = np.zeros_like(self.data)

        acc_data = [self.variance, self.flags, self.stellar]
        ext_names = [variance, flags, stellar]
        labels = ['Variance', 'Flags', 'Synthetic']

        for i, j, lab in zip(acc_data, ext_names, labels):

            if j is not None:
                if isinstance(j, str):
                    if j in hdu:
                        assert hdu[j].data.shape == self.data.shape, shape_error(lab)
                        i[:] = hdu[j].data
                elif isinstance(j, np.ndarray):
                    i[:] = j

        self.flags = self.flags.astype(bool)
        self._flags()

    def _wavelength(self, hdu, wave):

        if isinstance(wave, str) and (wave in hdu):
            assert hdu[wave].data.shape == self.data.shape[:1],\
                   'wavelength  must have the same shape of the data'
            self.wl = hdu[wave].data
        else:
            self.wl = self.wcs.wcs_pix2world(np.arange(len(self.data)), 0)[0]

    def _flags(self):

        # Flag nan and inf values
        self.flags += (np.isnan(self.data) + np.isinf(self.data)
                       + np.isnan(self.variance) + np.isinf(self.variance)
                       + np.isnan(self.stellar) + np.isinf(self.stellar))

    def _load(self, fname: str, scidata: str = 'SCI', variance: str = None, flags: str = None, stellar: str = None,
              primary: str = 'PRIMARY', redshift: float = None, wcs_axis: int = None, wavelength: str = None) -> None:
        self.fitsfile = fname

        with fits.open(fname) as hdu:
            self.data = hdu[scidata].data
            self.header = hdu[primary].header
            self.header_data = hdu[scidata].header
            if wcs_axis is not None:
                wcs_axis = [wcs_axis]
            self.wcs = wcs.WCS(self.header_data, naxis=wcs_axis)

            self._accessory_data(hdu, variance, flags, stellar)

            self._wavelength(hdu, wavelength)

        try:
            if self.header_data['cunit1'] == 'm':
                self.wl *= 1.e+10
        except KeyError:
            pass

        # Redshift from arguments takes precedence over redshift
        # from the image header.
        if redshift is not None:
            self.redshift = redshift
        elif 'redshift' in self.header:
            self.redshift = self.header['REDSHIFT']
        else:
            self.redshift = 0
        self.data, self.rest_wavelength = self.doppler_correction(self.redshift, self.data, self.wl)

    @staticmethod
    def doppler_correction(z, flux, wl):
        rest_wavelength = wl / (1. + z)
        flux = flux * (1 + z)
        return flux, rest_wavelength

    def dn4000(self):
        """
        Notes
        -----
        Dn4000 index based on Balogh et al. 1999 (ApJ, 527, 54).

        The index is defined as the ratio between continuum fluxes
        at 3850A-3950A and 4000A-4100A.

        red / blue

        """

        warn_message = 'Dn4000 could not be evaluated because the spectrum does not include wavelengths shorter than ' \
                       '3850.'

        if self.rest_wavelength[0] > 3850:
            warnings.warn(RuntimeWarning(warn_message))
            dn = np.nan

        else:
            # Mask for the blue part
            bm = (self.rest_wavelength >= 3850) & (self.rest_wavelength <= 3950)
            # Mask for the red part
            rm = (self.rest_wavelength >= 4000) & (self.rest_wavelength <= 4100)
            # Dn4000
            dn = np.sum(self.data[rm]) / np.sum(self.data[bm])

        return dn

    def plot(self, over_plot=True):

        if over_plot:
            ax = plt.gca()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax.plot(self.rest_wavelength, self.data)
        plt.show()
