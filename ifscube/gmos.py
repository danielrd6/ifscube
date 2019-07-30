import numpy as np
from astropy import wcs
from astropy.io import fits

from . import datacube


class Cube(datacube.Cube):
    """
    A class for dealing with data cubes, originally written to work
    with GMOS IFU.
    """

    def __init__(self, *args, **kwargs):

        datacube.Cube.__init__(self, *args, **kwargs)
        if len(args) > 0:
            self._load(*args, **kwargs)

    def _load(self, fitsfile, redshift=None, vortab=None, dataext='SCI', hdrext='PRIMARY', var_ext='ERR',
              ncubes_ext='NCUBE', nan_spaxels='all', spatial_mask=None):
        """
        Initializes the class and loads basic information onto the
        object.

        Parameters
        ----------
        fitsfile: string
            Name of the FITS file containing the GMOS datacube. This
            should be the standard output from the GFCUBE task of the
            GEMINI-GMOS IRAF package.
        redshift : float
            Value of redshift (z) of the source, if no Doppler
            correction has been applied to the spectra yet.
        vortab : string
            Name of the file containing the Voronoi binning table
        dataext: integer
            Extension of the FITS file containing the scientific data.
        hdrext: integer
            Extension of the FITS file containing the basic header.
        var_ext: integer
            Extension of the FITS file containing the variance cube.
        nan_spaxels: None, 'any', 'all'
            Mark spaxels as NaN if any or all pixels are equal to
            zero.


        Returns
        -------
        Nothing.
        """

        self.dataext = dataext
        self.var_ext = var_ext
        self.ncubes_ext = ncubes_ext

        hdu = fits.open(fitsfile)

        self.data = hdu[dataext].data
        self.header_data = hdu[dataext].header
        self.header = hdu[hdrext].header
        self.hdrext = hdrext

        if spatial_mask is not None:
            assert hdu[spatial_mask].data.shape == self.data.shape[1:], \
                'Spatial mask must match the last two dimensions of the data cube.'
            self.spatial_mask = hdu[spatial_mask].data.astype('bool')
        else:
            self.spatial_mask = np.zeros(self.data.shape[1:]).astype('bool')

        if nan_spaxels == 'all':
            self.nan_mask = np.isnan(self.data).all(axis=0)
        elif nan_spaxels == 'any':
            self.nan_mask = np.isnan(self.data).any(axis=0)
        else:
            self.nan_mask = np.zeros(self.data.shape[1:]).astype('bool')
        self.spatial_mask |= self.nan_mask

        self.wcs = wcs.WCS(self.header_data)
        self.wl = self.wcs.sub(axes=(3,)).wcs_pix2world(np.arange(self.data.shape[0]), 0)[0]

        if redshift is None:
            try:
                redshift = self.header['REDSHIFT']
            except KeyError:
                print('WARNING! Redshift not given and not found in the image header. Using redshift = 0.')
                redshift = 0.0
        self.rest_wavelength = self.wl / (1. + redshift)

        if var_ext is not None:
            # The noise for each pixel in the cube
            self.noise_cube = hdu[var_ext].data
            self.variance = np.square(self.noise_cube)

            # An image of the mean noise, collapsed over the
            # wavelength dimension.
            self.noise = np.nanmean(hdu[var_ext].data, 0)

            # Image of the mean signal
            self.signal = np.nanmean(self.data, 0)

            # Maybe this step is redundant, I have to check it later.
            # Guarantees that both noise and signal images have
            # the appropriate spaxels set to nan.
            self.noise[self.nan_mask] = np.nan
            self.signal[self.nan_mask] = np.nan

            self.noise[np.isinf(self.noise)] = self.signal[np.isinf(self.noise)]
        else:
            self.variance = np.ones_like(self.data)

        if ncubes_ext is not None:
            # The self.ncubes variable describes how many different
            # pixels contributed to the final combined pixel. This can
            # also serve as a flag, when zero cubes contributed to the
            # pixel. Additionally, it may be useful to mask regions that
            # are present in only one observation, for greater
            # confidence.
            self.ncubes = hdu[ncubes_ext].data
        else:
            self.ncubes = np.ones_like(self.data)

        self.flags = np.zeros_like(self.data, dtype=int)
        self.ncubes[np.isnan(self.ncubes)] = 0
        self.flags[self.ncubes <= 0] = 1
        self.flags = self.flags.astype(bool)

        try:
            if self.header['VORBIN']:
                vortab = fits.open(fitsfile)['VOR'].data
                self.voronoi_tab = vortab
                self.binned = True
        except KeyError:
            self.binned = False

        self.fitsfile = fitsfile
        self.redshift = redshift

        self.stellar = np.zeros_like(self.data)
        self.weights = np.zeros_like(self.data)

        self._set_spec_indices()
