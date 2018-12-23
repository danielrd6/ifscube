# stdlib
import copy

# Third party
import numpy as np
from astropy.io import fits
from astropy import wcs

# Local
from . import datacube


class cube(datacube.Cube):

    """
    A class for dealing with data cubes, originally written to work
    with GMOS IFU.
    """

    def __init__(self, *args, **kwargs):

        if len(args) > 0:
            self._load(*args, **kwargs)

    def _load(self, fitsfile, redshift=None, vortab=None,
              dataext='SCI', hdrext='PRIMARY', var_ext='ERR',
              ncubes_ext='NCUBE', nan_spaxels='all',
              spatial_mask=None):
        """
        Initializes the class and loads basic information onto the
        object.

        Parameters
        ----------
        fitstile : string
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
        self.spatial_mask = spatial_mask

        hdulist = fits.open(fitsfile)

        self.data = hdulist[dataext].data
        self.header_data = hdulist[dataext].header
        self.header = hdulist[hdrext].header
        self.hdrext = hdrext

        if nan_spaxels is not None:
            if nan_spaxels == 'all':
                self.nanSpaxels = np.all(self.data == 0, 0)
            if nan_spaxels == 'any':
                self.nanSapxels = np.any(self.data == 0, 0)
            self.data[:, self.nanSpaxels] = np.nan

        self.wcs = wcs.WCS(self.header_data)
        self.wl = self.wcs.sub(axes=(3,)).wcs_pix2world(
            np.arange(self.data.shape[0]), 0)[0]

        if redshift is None:
            try:
                redshift = self.header['REDSHIFT']
            except KeyError:
                print(
                    'WARNING! Redshift not given and not found in the image' +
                    ' header. Using redshift = 0.')
                redshift = 0.0
        self.restwl = self.wl / (1. + redshift)

        if var_ext is not None:
            # The noise for each pixel in the cube
            self.noise_cube = hdulist[var_ext].data
            self.variance = np.square(self.noise_cube)

            # An image of the mean noise, collapsed over the
            # wavelength dimension.
            self.noise = np.nanmean(hdulist[var_ext].data, 0)

            # Image of the mean signal
            self.signal = np.nanmean(self.data, 0)

            # Maybe this step is redundant, I have to check it later.
            # Guarantees that both noise and signal images have
            # the appropriate spaxels set to nan.
            self.noise[self.nanSpaxels] = np.nan
            self.signal[self.nanSpaxels] = np.nan

            self.noise[np.isinf(self.noise)] =\
                self.signal[np.isinf(self.noise)]
        else:
            self.variance = np.ones_like(self.data)

        if ncubes_ext is not None:
            # The self.ncubes variable describes how many different
            # pixels contributed to the final combined pixel. This can
            # also serve as a flag, when zero cubes contributed to the
            # pixel. Additionaly, it may be useful to mask regions that
            # are present in only one observation, for greater
            # confidence.
            self.ncubes = hdulist[ncubes_ext].data
        else:
            self.ncubes = np.ones_like(self.data)

        self.flags = np.zeros_like(self.data)
        self.ncubes[np.isnan(self.ncubes)] = 0
        self.flags[self.ncubes <= 0] = 1

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

    def voronoi_binning(self, targetsnr=10.0, writefits=False,
                        outfile=None, clobber=False, writevortab=True,
                        dataext=1):
        """
        Applies Voronoi binning to the data cube, using Cappellari's
        Python implementation.

        Parameters
        ----------
        targetsnr : float
            Desired signal to noise ratio of the binned pixels
        writefits : boolean
            Writes a FITS image with the output of the binning.
        outfile : string
            Name of the output FITS file. If 'None' then the name of
            the original FITS file containing the data cube will be used
            as a root name, with '.bin' appended to it.
        clobber : boolean
            Overwrites files with the same name given in 'outfile'.
        writevortab : boolean
            Saves an ASCII table with the binning recipe.

        Returns
        -------
        Nothing.
        """

        try:
            from voronoi_2d_binning import voronoi_2d_binning
        except ImportError:
            raise ImportError(
                'Could not find the voronoi_2d_binning module. '
                'Please add it to your PYTHONPATH.')
        try:
            x = np.shape(self.noise)
        except AttributeError:
            print(
                'This function requires prior execution of the snr_eval' +
                'method.')
            return

        # Initializing the binned arrays as zeros.
        try:
            b_data = np.zeros(np.shape(self.data), dtype='float32')
        except AttributeError as err:
            err.args += (
                'Could not access the data attribute of the gmosdc object.',)
            raise err

        try:
            b_ncubes = np.zeros(np.shape(self.ncubes), dtype='float32')
        except AttributeError as err:
            err.args += (
                'Could not access the ncubes attribute of the gmosdc object.',)
            raise err

        try:
            b_noise = np.zeros(np.shape(self.noise_cube), dtype='float32')
        except AttributeError as err:
            err.args += (
                'Could not access the noise_cube attribute of the gmosdc '
                'object.',)

        valid_spaxels = np.ravel(~np.isnan(self.signal))

        x = np.ravel(np.indices(np.shape(self.signal))[1])[valid_spaxels]
        y = np.ravel(np.indices(np.shape(self.signal))[0])[valid_spaxels]

        xnan = np.ravel(np.indices(np.shape(self.signal))[1])[~valid_spaxels]
        ynan = np.ravel(np.indices(np.shape(self.signal))[0])[~valid_spaxels]

        s, n = copy.deepcopy(self.signal), copy.deepcopy(self.noise)

        s[s <= 0] = np.average(self.signal[self.signal > 0])
        n[n <= 0] = np.average(self.signal[self.signal > 0]) * .5

        signal, noise = np.ravel(s)[valid_spaxels], np.ravel(n)[valid_spaxels]

        binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = \
            voronoi_2d_binning(x, y, signal, noise, targetsnr, plot=1, quiet=0)
        v = np.column_stack([y, x, binNum])

        # For every nan in the original cube, fill with nan the
        # binned cubes.
        for i in [b_data, b_ncubes, b_noise]:
            i[:, ynan, xnan] = np.nan

        for i in np.arange(binNum.max() + 1):
            samebin = v[:, 2] == i
            samebin_coords = v[samebin, :2]

            for k in samebin_coords:

                # Storing the indexes in a variable to avoid typos in
                # subsequent references to the same indexes.
                #
                # binned_idx represents the indexes of the new binned
                # arrays, which are being created here.
                #
                # unbinned_idx represents the original cube indexes.

                binned_idx = (Ellipsis, k[0], k[1])
                unbinned_idx = (
                    Ellipsis, samebin_coords[:, 0], samebin_coords[:, 1]
                )

                # The binned spectra should be the average of the
                # flux densities.
                b_data[binned_idx] = np.average(
                    self.data[unbinned_idx], axis=1)

                # Ncubes must be the sum, since they represent how many
                # original pixels have contributed to each pixel in the
                # binned cube. In the unbinned data this is identical
                # to the number of individual exposures that contribute
                # to a given pixel.
                b_ncubes[binned_idx] = np.sum(
                    self.ncubes[unbinned_idx], axis=1)

                # The resulting noise is defined as the quadratic sum
                # of the original noise.
                b_noise[binned_idx] = np.sqrt(np.sum(np.square(
                    self.noise_cube[unbinned_idx]), axis=1))

        if writefits:

            # Starting with the original data cube
            hdulist = fits.open(self.fitsfile)
            hdr = self.header

            # Add a few new keywords to the header
            try:
                hdr['REDSHIFT'] = self.redshift
            except KeyError:
                hdr['REDSHIFT'] = (self.redshift,
                                   'Redshift used in GMOSDC')
            hdr['VORBIN'] = (True, 'Processed by Voronoi binning?')
            hdr['VORTSNR'] = (targetsnr, 'Target SNR for Voronoi binning.')

            hdulist[self.hdrext].header = hdr

            # Storing the binned data in the HDUList
            hdulist[self.dataext].data = b_data
            hdulist[self.var_ext].data = b_noise
            hdulist[self.ncubes_ext].data = b_ncubes

            # Write a FITS table with the description of the
            # tesselation process.
            tbhdu = fits.BinTableHDU.from_columns(
                [
                    fits.Column(name='xcoords', format='i8', array=x),
                    fits.Column(name='ycoords', format='i8', array=y),
                    fits.Column(name='binNum', format='i8', array=binNum),
                ], name='VOR')

            tbhdu_plus = fits.BinTableHDU.from_columns(
                [
                    fits.Column(name='ubin', format='i8',
                                array=np.unique(binNum)),
                    fits.Column(name='xNode', format='F16.8', array=xNode),
                    fits.Column(name='yNode', format='F16.8', array=yNode),
                    fits.Column(name='xBar', format='F16.8', array=xBar),
                    fits.Column(name='yBar', format='F16.8', array=yBar),
                    fits.Column(name='sn', format='F16.8', array=sn),
                    fits.Column(name='nPixels', format='i8', array=nPixels),
                ], name='VORPLUS')

            hdulist.append(tbhdu)
            hdulist.append(tbhdu_plus)

            if outfile is None:
                outfile = '{:s}bin.fits'.format(self.fitsfile[:-4])

            hdulist.writeto(outfile, clobber=clobber)

        self.binned_cube = b_data
