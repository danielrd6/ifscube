from . import cubetools, spectools
from astropy.io import fits
from astropy import wcs
import numpy as np


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
            self.restwl = self.wl / (1. + redshift)
        else:
            self.restwl = self.wl

    def eqw(self, component):


