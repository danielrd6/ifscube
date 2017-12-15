"""
Functions for the analysis of integral field spectroscopy.

Author: Daniel Ruschel Dutra
Website: https://github.com/danielrd6/ifscube
"""
from copy import deepcopy

import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from astropy import constants, units, table
from astropy.io import fits
import progressbar

from . import datacube, spectools, onedspec
from . import plots as ifsplots
from . import elprofile as lprof


def peak_spaxel(cube):
    im = cube.sum(axis=0)
    y, x = [int(i) for i in np.where(im == im.max())]
    return x, y


def nan_to_nearest(d):
    """
    Replaces nan values with the valeu of the nearest pixel.

    Parameters
    ----------
    d : numpy.ndarray or numpy.ma.core.MaskedArray
      The array to have the nan values replaced. It d is a masked
      array, the masked values will also be replaced.

    Returns
    -------
    g : same type as d
      Output array.

    Description
    -----------
    ...
    """
    x, y = [np.ravel(i) for i in np.indices(d.shape)]

    if type(d) != np.ma.core.MaskedArray:
        m1 = np.full_like(d, False, dtype='bool')
        m2 = np.isnan(d)
        d = ma.array(data=d, mask=(m1 | m2))

    # Flatten arrays
    dflat = np.ravel(d)

    idx = np.where(dflat.mask)[0]
    g = deepcopy(dflat)

    for i in idx:
        r = np.sqrt((x - x[i]) ** 2 + (y - y[i]) ** 2)[~dflat.mask]
        g[i] = dflat[~dflat.mask][np.argsort(r)[0]]

    g.mask = dflat.mask

    return g.reshape(d.shape)


class nanSolution:

    def ppxf(self, sol, galaxy, bestfit):
        self.sol = np.array([np.nan for i in range(len(sol))])
        self.galaxy = np.array([np.nan for i in range(len(galaxy))])
        self.bestfit = np.array([np.nan for i in range(len(galaxy))])


def wlprojection(arr, wl, wl0, fwhm=10, filtertype='box'):
        """
        Writes a projection of the data cube along the wavelength
        coordinate, with the flux given by a given type of filter.

        Parameters:
        -----------
        arr : np.ndarray
          Array to projected.
        wl : np.ndarray
          Wavelength coordinates of the arr pixels.
        wl0 : float
          Central wavelength at the rest frame.
        fwhm : float
          Full width at half maximum. See 'filtertype'.
        filtertype : string
          Type of function to be multiplied by the spectrum to return
          the argument for the integral.
          'box'      = Box function that is zero everywhere and 1
                       between wl0-fwhm/2 and wl0+fwhm/2.
          'gaussian' = Normalized gaussian function with center at
                       wl0 and sigma = fwhm/(2*sqrt(2*log(2)))

        Returns:
        --------
        outim : numpy.ndarray
          The integrated flux of the cube times the filter.
        """

        if filtertype == 'box':
            arrfilt = np.array(
                (wl >= wl0 - fwhm / 2.) & (wl <= wl0 + fwhm / 2.),
                dtype='float')
            arrfilt /= trapz(arrfilt, wl)
        elif filtertype == 'gaussian':
            s = fwhm / (2. * np.sqrt(2. * np.log(2.)))
            arrfilt = 1. / np.sqrt(2 * np.pi) *\
                np.exp(-(wl - wl0)**2 / 2. / s**2)
        else:
            raise ValueError(
                'ERROR! Parameter filtertype "{:s}" not understood.'
                .format(filtertype))

        outim = np.zeros(arr.shape[1:], dtype='float32')
        idx = np.column_stack([np.ravel(i) for i in np.indices(outim.shape)])

        for i, j in idx:
            outim[i, j] = trapz(arr[:, i, j] * arrfilt, wl)

        return outim


def bound_updater(p0, bound_range, bounds=None):

    newbound = []
    if np.shape(bound_range) != ():
        npars = len(bound_range)

        if type(bound_range[0]) == dict:
            for i in range(0, len(p0), npars):
                for j in range(npars):
                    if bound_range[j]['type'] == 'factor':
                        newbound += [
                            [p0[i + j] * (1. - bound_range[j]['value']),
                             p0[i + j] * (1. + bound_range[j]['value'])]]
                        if newbound[-1][1] < newbound[-1][0]:
                            newbound[-1].sort()
                    if bound_range[j]['type'] == 'add':
                        newbound += [
                            [p0[i + j] - bound_range[j]['value'],
                             p0[i + j] + bound_range[j]['value']]]
                    if bound_range[j]['type'] == 'hard':
                        newbound += [
                            [- bound_range[j]['value'],
                             + bound_range[j]['value']]]

        else:
            for i in range(0, len(p0), npars):
                for j in range(npars):
                    newbound += [
                        [p0[i + j] - bound_range[j],
                         p0[i + j] + bound_range[j]]]
    else:

        for i in p0:
            if i == 0:
                newbound += [[i - bound_range, i + bound_range]]
            else:
                newbound += [[i * (1. - bound_range), i * (1. + bound_range)]]
                if newbound[-1][1] < newbound[-1][0]:
                    newbound[-1].sort()

    # If bounds is set, then newbounds must not exceed bounds.
    if bounds is not None:
        for i, j in enumerate(newbound):

            low_new, high_new = j
            low, high = bounds[i]
            d = abs(high_new - low_new)

            if low is not None:
                if low_new < low:
                    low_new = deepcopy(low)
                    high_new = low_new + d
            else:
                low_new = None

            if high is not None:
                if high_new > high:
                    high_new = deepcopy(high)
                    low_new = high_new - d
            else:
                high_new = None

            newbound[i] = [low_new, high_new]

    return newbound


def scale_bounds(bounds, flux_sf):

    b = deepcopy(bounds)
    for i, j in enumerate(b):
        for k in (0, 1):
            if j[k] is not None:
                j[k] /= flux_sf[i]

    return b


class gmosdc(datacube.Cube):

    """
    A class for dealing with data cubes, originally written to work
    with GMOS IFU.
    """

    def __init__(self, *args, **kwargs):

        if len(args) > 0:
            self.__load__(*args, **kwargs)

    def __load__(self, fitsfile, redshift=None, vortab=None,
                 dataext=1, hdrext=0, var_ext=None, ncubes_ext=None,
                 nan_spaxels='all', spatial_mask=None):
        """
        Initializes the class and loads basic information onto the
        object.

        Parameters:
        -----------
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


        Returns:
        --------
        Nothing.
        """

        self.dataext = dataext
        self.var_ext = var_ext
        self.ncubes_ext = ncubes_ext
        self.spatial_mask = spatial_mask

        hdulist = fits.open(fitsfile)

        self.data = hdulist[dataext].data

        if nan_spaxels is not None:
            if nan_spaxels == 'all':
                self.nanSpaxels = np.all(self.data == 0, 0)
            if nan_spaxels == 'any':
                self.nanSapxels = np.any(self.data == 0, 0)
            self.data[:, self.nanSpaxels] = np.nan

        self.header_data = hdulist[dataext].header
        self.header = hdulist[hdrext].header
        self.hdrext = hdrext

        self.wl = spectools.get_wl(
            fitsfile, hdrext=dataext, dimension=0, dwlkey='CD3_3',
            wl0key='CRVAL3', pix0key='CRPIX3')

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

        self.__set_spec_indices__()
