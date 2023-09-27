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
            arg_names = ['fname', 'scidata', 'variance', 'flags', 'stellar', 'primary', 'redshift', 'wcs_axis',
                         'wavelength']

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
                if isinstance(j, str) or isinstance(j, int):
                    if j in hdu:
                        assert hdu[j].data.shape == self.data.shape, shape_error(lab)
                        i[:] = hdu[j].data
                elif isinstance(j, np.ndarray):
                    i[:] = j
                else:
                    raise RuntimeError(f"Error reading {lab} data."
                                       f" Parameter {j} is not a valid FITS extension nor an array object.")

        self.flags = self.flags.astype(bool)
        self._flags()

    def _wavelength(self, hdu: fits.HDUList, wave: Union[str, int] = None):
        """
        Reads and sets the wavelength vector.

        Parameters
        ----------
        hdu : astropy.fits.HDUList
            The HDUList object containing the data being read.
        wave : None (default), str or int
            If None, the wavelength will be read from the WCS information on the image header.
            The wavelength can also be stored as an array in an ImageHDU extension, in which
            case 'wave' can be either a string or an integer specifying that extension.

        Notes
        -----
        This method relies on the astropy.wcs module to calculate the wavelength vector from
        the image header. Therefore, it is possible that the wavelength units will be converted
        to SI units during the transformation from pixel coordinates to world coordinates. If this
        conversion occurs, and the transformation results in wavelength units of meters, a scaling
        factor will be applied to return the wavelength back to Angstroms, and a warning will be
        issued. Wavelengths read directly as an array from an image extension will not undergo this
        transformation.
        """
        if isinstance(wave, str) or isinstance(wave, int):
            assert hdu[wave].data.shape[0] == self.data.shape[0], 'Wavelength must have the same shape of the data.'
            self.wl = hdu[wave].data
        else:
            self.wl = self.wcs.all_pix2world(np.arange(len(self.data)), 0)[0]
            if self.wcs.world_axis_units == ["m"]:
                warnings.warn(message="Wavelength read in meters. Changing it to Angstroms.", category=RuntimeWarning)
                self.wl *= 1.0e+10

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
            assert (self.wcs.naxis == 1) or (wcs_axis is not None), \
                f'WCS read from extension {scidata} has {self.wcs.naxis} dimensions but no wcs_axis was specified. ' \
                'Please specify the correct axis (usually 1) in the loading section of the configuration file.'

            self._accessory_data(hdu, variance, flags, stellar)

            self._wavelength(hdu, wavelength)

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
