"""
Functions for the analysis of integral field spectroscopy.

Author: Daniel Ruschel Dutra
Website: https://github.com/danielrd6/ifscube
"""
import configparser
import copy

import numpy as np
from astropy import table
from astropy.io import fits
from numpy import interp, ma
from scipy.integrate import trapz


def peak_spaxel(cube):
    im = cube.sum(axis=0)
    y_max, x_max = np.where(im == im[~np.isnan(im)].max())

    assert y_max.size == 1, \
        'More than one spaxel with flux equal to the maximum. ' \
        'Consider using center of mass for the center spaxel. '

    y, x = [int(i) for i in (y_max, x_max)]
    return x, y


def nan_to_nearest(d):
    """
    Replaces nan values with the value of the nearest pixel.

    Parameters
    ----------
    d: numpy.ndarray, numpy.ma.core.MaskedArray
      The array to have the nan values replaced. It d is a masked
      array, the masked values will also be replaced.

    Returns
    -------
    g: numpy.ma.core.MaskedArray
      Output array.

    Description
    -----------
    """
    x, y = [np.ravel(i) for i in np.indices(d.shape)]

    if type(d) != np.ma.core.MaskedArray:
        m1 = np.full_like(d, False, dtype='bool')
        m2 = np.isnan(d)
        d = ma.array(data=d, mask=(m1 | m2))

    # Flatten arrays
    dflat = d.flatten()

    idx = np.where(dflat.mask)[0]
    g = copy.deepcopy(dflat)

    for i in idx:
        r = np.sqrt((x - x[i]) ** 2 + (y - y[i]) ** 2)[~dflat.mask]
        g[i] = dflat[~dflat.mask][np.argsort(r)[0]]

    g.mask = dflat.mask

    return g.reshape(d.shape)


class NanSolution:

    def __init__(self):
        self.sol = None
        self.galaxy = None
        self.bestfit = None

    def ppxf(self, sol, galaxy):
        self.sol = np.array([np.nan for _ in range(len(sol))])
        self.galaxy = np.array([np.nan for _ in range(len(galaxy))])
        self.bestfit = np.array([np.nan for _ in range(len(galaxy))])


def wlprojection(arr, wl, wl0, fwhm=10, filtertype='box'):
    """
    Writes a projection of the data cube along the wavelength
    coordinate, with the flux given by a given type of filter.

    Parameters
    ----------
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

    Returns
    -------
    outim : numpy.ndarray
      The integrated flux of the cube times the filter.
    """

    assert (wl[0] < wl0 - fwhm / 2) and (wl[-1] > wl0 + fwhm / 2), \
        'Wavelength limits outside wavelength vector.'

    if filtertype == 'box':
        lower_wl = wl0 - fwhm / 2.
        upper_wl = wl0 + fwhm / 2.
        # Find non zero filter indices.
        ind_non_zero = np.flatnonzero(np.array((wl >= lower_wl) & (wl <= upper_wl), dtype='float'))
        ind_non_zero = np.pad(ind_non_zero, (1, 1), 'constant',
                              constant_values=(ind_non_zero[0] - 1, ind_non_zero[-1] + 1))
        # Keep only the relevant slices of data array.
        arr = arr[ind_non_zero, :, :]
        # Create filter and wavelenght arrays with array shape.
        arrfilt = np.ones(np.shape(arr))
        wl = np.tile(wl[ind_non_zero], (len(arr[0, 0, :]), len(arr[0, :, 0]), 1)).T
        # Adjust limits to account for integration of fractional pixel slice
        lower_frac = (lower_wl - wl[0]) / (wl[1] - wl[0])
        upper_frac = (wl[-1] - upper_wl) / (wl[-1] - wl[-2])
        wl[0] = lower_wl
        wl[-1] = upper_wl
        arrfilt[0, :, :] += lower_frac * (arr[1, :, :] - arr[0, :, :]) / arr[0, :, :]
        arrfilt[-1, :, :] -= upper_frac * (arr[-1, :, :] - arr[-2, :, :]) / arr[-1, :, :]
        arrfilt /= trapz(arrfilt, wl, axis=0)
        # FIXME: Is this still needed? I can't remember why I put it here.
    elif filtertype == 'gaussian':
        s = fwhm / (2. * np.sqrt(2. * np.log(2.)))
        arrfilt = 1. / np.sqrt(2 * np.pi) * np.exp(-(wl - wl0) ** 2 / 2. / s ** 2)
        arrfilt = np.tile(arrfilt, (len(arr[0, 0, :]), len(arr[0, :, 0]), 1)).T
        wl = np.tile(wl, (len(arr[0, 0, :]), len(arr[0, :, 0]), 1)).T
    else:
        raise ValueError('ERROR! Parameter filtertype "{:s}" not understood.'.format(filtertype))

    outim = trapz(arr * arrfilt, wl, axis=0)

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
                    low_new = copy.deepcopy(low)
                    high_new = low_new + d
            else:
                low_new = None

            if high is not None:
                if high_new > high:
                    high_new = copy.deepcopy(high)
                    low_new = high_new - d
            else:
                high_new = None

            newbound[i] = [low_new, high_new]

    return newbound


def scale_bounds(bounds, flux_sf):
    b = copy.deepcopy(bounds)
    for i, j in enumerate(b):
        for k in (0, 1):
            if j[k] is not None:
                j[k] /= flux_sf[i]

    return b


def rebin(arr, xbin, ybin, combine='sum', mask=None):
    assert combine in ['sum', 'mean'], \
        'The combine parameter must be "sum" or "mean".'

    old_shape = copy.copy(arr.shape)
    new_shape = (
        old_shape[0],
        int(np.ceil(old_shape[1] / ybin)),
        int(np.ceil(old_shape[2] / xbin)),
    )

    new = np.zeros(new_shape)

    if mask is not None:
        data = ma.array(data=arr, mask=mask)
    else:
        data = ma.array(data=arr)

    def comb_sum(x, m, n):
        return x[:, m * ybin:(m + 1) * ybin, n * xbin:(n + 1) * xbin].sum(axis=(1, 2))

    def comb_avg(x, m, n):
        return x[:, m * ybin:(m + 1) * ybin, n * xbin:(n + 1) * xbin].mean(axis=(1, 2))

    comb_fun = dict(sum=comb_sum, mean=comb_avg)

    for i in range(new_shape[1]):
        for j in range(new_shape[2]):
            new[:, i, j] = comb_fun[combine](data, i, j)

    return new


def aperture_spectrum(arr, x0=None, y0=None, radius=3, combine='sum'):
    y, x = np.indices(arr.shape[1:])

    if x0 is None:
        x0 = x.mean()
    if y0 is None:
        y0 = x.mean()

    x -= x0
    y -= y0
    r = np.sqrt(x ** 2 + y ** 2)

    new_arr = ma.masked_invalid(arr)
    new_arr.mask |= (r > radius)

    npix = np.sum(r < radius)

    if combine == 'sum':
        s = new_arr.sum(axis=(1, 2))
    elif combine == 'mean':
        s = new_arr.mean(axis=(1, 2))
    elif combine == 'sqrt_sum':
        s = np.sqrt((np.square(new_arr)).sum(axis=(1, 2)))
    else:
        raise Exception('Combination type not understood.')

    return s, npix


# This is only here for backwards compatibility.
class gmosdc:

    def __init__(self, *args, **kwargs):
        raise DeprecationWarning(
            'The gmosdc class has been moved to the ifscube.gmos '
            'module. Please change the instance initialization line from '
            'cubetools.gmosdc() to gmos.cube()')


def append_config(config_file: str, fit_file: str) -> None:
    """
    Appends the configuration parameters as FITS table to the fit output file.
    
    Parameters
    ----------
    config_file : str
        Name of the configuration file.
    fit_file : str
        Name of the file onto which the configuration table will be
        appended.

    Returns
    -------
    None

    """
    c = configparser.ConfigParser()
    c.read(config_file)
    t = table.Table(names=['parameters', 'values'], dtype=('S64', 'S64'))

    for i in c.sections():
        for j in c[i]:
            t.add_row(('{:s}.{:s}'.format(i, j), c[i][j]))

    with fits.open(fit_file) as outfits:
        outfits.append(fits.BinTableHDU(data=t))
        outfits[-1].name = 'FITCONFIG'
        outfits.writeto(fit_file, overwrite=True)

    return


def resample_stellar(binned, original, normalization_wavelength=5500, span=30):
    """
    Resamples binned stellar spectra to each individual spaxel.

    Parameters
    ----------
    binned
        Binned cube with stellar spectra.
    original
        Original data cube
    normalization_wavelength
        Wavelength coordinate to perform the normalization.
    span
        Width in wavelength of the normalization window.

    Returns
    -------
    unbinned
        A data cube with the stellar spectra resampled for each spaxel.
    """

    binned_level = wlprojection(binned.stellar, binned.rest_wavelength, normalization_wavelength, span)
    original_level = wlprojection(original.data, original.rest_wavelength, normalization_wavelength, span)

    original.stellar = np.zeros_like(original.data)

    new_stellar = binned.stellar * original_level / binned_level
    for h in original.spec_indices:
        i, j = h
        original.stellar[:, i, j] = interp(original.rest_wavelength, binned.rest_wavelength, new_stellar[:, i, j],
                                           left=np.nan, right=np.nan)

    original.flags |= np.isnan(original.stellar)
    return original
