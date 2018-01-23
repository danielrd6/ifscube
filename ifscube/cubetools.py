"""
Functions for the analysis of integral field spectroscopy.

Author: Daniel Ruschel Dutra
Website: https://github.com/danielrd6/ifscube
"""
# STDLIB
import copy

# THIRD PARTY
import numpy as np
from numpy import ma
from scipy.integrate import trapz


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
    g = copy.deepcopy(dflat)

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

    assert combine in ['sum', 'mean'],\
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

    def combSum(x, i, j):
        return x[:, i * ybin:(i + 1) * ybin, j * xbin:(j + 1) * xbin].sum(
            axis=(1, 2))

    def combAvg(x, i, j):
        return x[:, i * ybin:(i + 1) * ybin, j * xbin:(j + 1) * xbin].mean(
            axis=(1, 2))

    comb_fun = dict(sum=combSum, mean=combAvg)

    for i in range(new_shape[1]):
        for j in range(new_shape[2]):
            new[:, i, j] = comb_fun[combine](data, i, j)

    return new


# This is only here for backwards compatibility.
class gmosdc:

    def __init__(self, *args, **kwargs):

        raise DeprecationWarning(
            'The gmosdc class has been moved to the ifscube.gmos '
            'module. Please change the instance initialization line from '
            'cubetools.gmosdc() to gmos.cube()')
