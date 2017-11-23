from . import cubetools, spectools
from . import elprofile as lprof
from astropy.io import fits
from astropy import wcs
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapz


class cube(cubetools.gmosdc):

    def __init__(self, fitsfile, redshift=None):

        self.filename = fitsfile
        self.redshift = redshift

        # Opens the FITS file.
        hdu = fits.open(fitsfile)
        self.header = hdu[0].header

        self.data = hdu['F_OBS'].data
        self.header_data = hdu['F_OBS'].header

        self.wcs = wcs.WCS(self.header_data)

        self.binned = False

        self.spec_indices = np.column_stack([
            np.ravel(np.indices(np.shape(self.data)[1:])[0]),
            np.ravel(np.indices(np.shape(self.data)[1:])[1])
        ])

        self.__getwl__()
        self.cont = hdu['F_SYN'].data
        self.syn = hdu['F_SYN'].data

        hdu.close()

    def __getwl__(self):
        """
        Builds a wavelength vector based on the wcs object created in
        the __init__ method.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """

        x = np.arange(self.data.shape[0])
        z = np.zeros_like(x)

        self.wl = self.wcs.wcs_pix2world(z, z, x, 0)[2] * 1e+10

        if self.redshift is not None:
            self.restwl = self.wl / (1. + self.redshift)
        else:
            self.restwl = self.wl

    def eqw(self, component, sigma_factor=5, windows=None):
        """
        Evaluates the equivalent width of a previous linefit.

        Parameters
        ----------
        component : number
          Component of emission model.
        sigma_factor : number
          Radius of integration as a number of line sigmas.
        windows : iterable
          Continuum fitting windows in the form
          [blue0, blue1, red0, red1].

        Returns
        -------
        eqw : numpy.ndarray 
          Equivalent widths measured on the emission line model and
          directly on the observed spectrum, respectively.
        """

        xy = self.spec_indices
        eqw_model = np.zeros(np.shape(self.em_model)[1:], dtype='float32')
        eqw_direct = np.zeros(np.shape(self.em_model)[1:], dtype='float32')

        if self.fit_func == lprof.gauss:
            npars = 3
        if self.fit_func == lprof.gausshermite:
            npars = 5

        par_indexes = np.arange(npars) + npars * component

        center_index = 1 + npars * component
        sigma_index = 2 + npars * component

        for i, j in xy:

            # Wavelength vector of the line fit
            fwl = self.fitwl
            # Rest wavelength vector of the whole data cube
            rwl = self.restwl
            # Center wavelength coordinate of the fit
            cwl = self.em_model[center_index, i, j]
            # Sigma of the fit
            sig = self.em_model[sigma_index, i, j]
            # Just a short alias for the sigma_factor parameter
            sf = sigma_factor

            nandata_flag = np.any(np.isnan(self.em_model[par_indexes, i, j]))
            nullcwl_flag = cwl == 0

            if nandata_flag or nullcwl_flag:

                eqw_model[i, j] = np.nan
                eqw_direct[i, j] = np.nan

            else:

                cond = (fwl > cwl - sf * sig) & (fwl < cwl + sf * sig)
                cond_data = (rwl > cwl - sf * sig) & (rwl < cwl + sf * sig)

                fit = self.fit_func(
                        fwl[cond], self.em_model[par_indexes, i, j])

                # cont = self.fitcont[cond, i, j]

                # If the continuum fitting windos are set, use that
                # to define the weights vector.
                if windows is not None:
                    assert len(windows) == 4, 'Windows must be an '\
                        'iterable of the form (blue0, blue1, red0, red1)'
                    weights = np.zeros_like(self.fitwl)
                    windows_cond = (
                        ((fwl > windows[0]) & (fwl < windows[1])) |
                        ((fwl > windows[2]) & (fwl < windows[3]))
                    )
                    weights[windows_cond] = 1
                else:
                    weights = np.ones_like(self.fitwl)

                cont = spectools.continuum(
                    fwl, self.syn[cond, i, j], weights=weights,
                    degr=1, niterate=3, lower_threshold=1,
                    upper_threshold=1, returns='function')

                cont_data = interp1d(
                    fwl, self.fitcont[:, i, j])(rwl[cond_data])

                eqw_model[i, j] = trapz(
                    1. - (fit + cont) / cont, x=fwl[cond])

                eqw_direct[i, j] = trapz(
                    1. - self.data[cond_data, i, j] / cont_data,
                    x=rwl[cond_data])

        return np.array([eqw_model, eqw_direct])
