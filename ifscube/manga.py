from . import cubetools, spectools
from . import elprofile as lprof
from astropy.io import fits
from astropy import wcs
import numpy as np
from scipy.integrate import trapz


class cube(cubetools.gmosdc):

    def __init__(self, fitsfile, redshift=None):

        self.filename = fitsfile
        self.redshift = redshift

        # Opens the FITS file.
        hdu = fits.open(fitsfile)
        self.header = hdu[0].header

        self.data = hdu['F_OBS'].data
        self.flags = hdu['F_FLAG'].data
        self.header_data = hdu['F_OBS'].header
        self.fobs_norm = hdu['FOBS_NORM'].data

        self.wcs = wcs.WCS(self.header_data)

        self.binned = False

        self.__getwl__()
        self.__spec_indices__()

        self.cont = hdu['F_SYN'].data
        self.syn = hdu['F_SYN'].data

        hdu.close()

    def __spec_indices__(self):

        # _unused = 0x0001
        no_data = 0x0002
        bad_pix = 0x0004
        ccd_gap = 0x0008
        # telluric = 0x0010
        # seg_has_badpixels = 0x0020
        # low_sn = 0x0040

        # starlight_masked = 0x0100
        starlight_failed_run = 0x0200
        starlight_no_data = 0x0400
        # starlight_clipped = 0x0800

        # Compound flags
        no_obs = no_data | bad_pix | ccd_gap
        # before_starlight = no_obs | telluric | low_sn
        no_starlight = starlight_no_data | starlight_failed_run

        flags = no_obs | no_starlight
        bad_lyx = (self.flags & flags) > 0
        spatial_mask = (
            np.sum(bad_lyx, 0) > len(self.restwl) * 0.8
        )

        self.spatial_mask = spatial_mask

        self.spec_indices = np.column_stack([
            np.ravel(
                np.indices(np.shape(self.data)[1:])[0][~spatial_mask]),
            np.ravel(
                np.indices(np.shape(self.data)[1:])[1][~spatial_mask]),
        ])

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

        eqw_model[self.spatial_mask] = np.nan
        eqw_direct[self.spatial_mask] = np.nan

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
                cond_syn = (rwl >= fwl[0]) & (rwl <= fwl[-1])

                fit = self.fit_func(
                        fwl[cond], self.em_model[par_indexes, i, j])

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
                    fwl, self.syn[cond_syn, i, j], weights=weights,
                    degr=1, niterate=3, lower_threshold=3,
                    upper_threshold=3, returns='function')[1][cond]

                eqw_model[i, j] = trapz(
                    1. - (fit + cont) / cont, x=fwl[cond])

                eqw_direct[i, j] = trapz(
                    1. - self.data[cond_data, i, j] / cont, x=fwl[cond])

        return np.array([eqw_model, eqw_direct])
