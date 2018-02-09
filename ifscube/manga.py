# STDLIB

# THIRD PARTY
import numpy as np
from scipy.integrate import trapz
from astropy import wcs
from astropy.io import fits

# LOCAL
from . import datacube, spectools
from . import elprofile as lprof


class cube(datacube.Cube):

    def __init__(self, *args, **kwargs):

        if len(args) > 0:
            self.__load__(*args, **kwargs)

    def __load__(self, fitsfile, redshift=0):

        self.fitsfile = fitsfile
        self.redshift = redshift

        # Opens the FITS file.
        hdu = fits.open(fitsfile)
        self.header = hdu[0].header

        self.data = hdu['F_OBS'].data
        self.header_data = hdu['F_OBS'].header
        self.fobs_norm = hdu['FOBS_NORM'].data
        self.noise_cube = hdu['F_ERR'].data
        self.flag_cube = hdu['F_FLAG'].data
        self.flags = self.flag_cube
        # self.weights = hdu['F_WEI'].data

        self.data *= self.fobs_norm
        self.noise_cube *= self.fobs_norm

        self.wcs = wcs.WCS(self.header_data)

        self.binned = False

        self.__getwl__()
        self.__flags__()
        self.__spec_indices__()

        self.cont = hdu['F_SYN'].data * self.fobs_norm
        self.syn = hdu['F_SYN'].data * self.fobs_norm

        self.variance = np.square(self.noise_cube)
        self.stellar = self.syn

        self.weights = np.ones_like(self.data)

        hdu.close()

    def __flags__(self):

        _unused = 0x0001
        no_data = 0x0002
        bad_pix = 0x0004
        ccd_gap = 0x0008
        telluric = 0x0010
        seg_has_badpixels = 0x0020
        low_sn = 0x0040

        no_obs = no_data | bad_pix | ccd_gap | _unused | telluric\
            | seg_has_badpixels | low_sn

        self.flags = self.flag_cube & no_obs

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
        bad_lyx = (self.flag_cube & flags) > 0
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
