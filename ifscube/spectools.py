#!/usr/bin/env python

"""
Spectools provide a number of small routines to work with 1D spectra in
the form of numpy arrays or FITS files, depending on the function.

If the spectrum is specified as an array, the default is to assume that
arr[:,0] are the wavelength coordinates and arr[:,1] are the flux
points.
"""
import copy
import re
import warnings
from typing import Callable, Iterable

import astropy.io.fits as pf
import numpy as np
from astropy import units
from scipy.integrate import trapz, quad, cumtrapz
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit, minimize


def remove_bg(x: np.ndarray, y: np.ndarray, sampling_limits: list, order: int = 1) -> tuple:
    """
    Removes a background function from one dimensional data.

    Parameters
    ----------
    x : numpy.array
    y : numpy.array
    sampling_limits : iterable
      A list of the background sampling limits
    order : integer
      Polyfit order

    Returns
    -------
    x : numpy.ndarray
    y_new : numpy.array
    """
    xs = np.array([])
    ys = np.array([])

    for i in range(0, len(sampling_limits), 2):
        xs = np.append(xs, x[(x >= sampling_limits[i]) & (x < sampling_limits[i + 1])])
        ys = np.append(ys, y[(x >= sampling_limits[i]) & (x < sampling_limits[i + 1])])

    p = np.polyfit(xs, ys, deg=order)

    y_new = y - np.polyval(p, x)

    return x, y_new


def fit_gauss(x, y, p0=None, fit_center=True, fit_background=True):
    """
    Returns a gaussian function that fits the data. This function
    requires a background subtraction, meaning that the data should
    fall to zero within the fitting limits.

    Parameters
    ----------
    x, y : numpy.ndarray
        Independent variable and flux value vectors.
    p0 : iterable
        Initial guesses for m, mu, sigma and bg respectively
        m = maximum value
        mu = center
        sigma = FWHM/(2*sqrt(2*log(2)))
        bg = background level
    fit_center : boolean
        Chooses whether to fit center or accept the value given
        in p0[1].
    fit_background : boolean
        Chooses whether to fit a horizontal background level or accept
        the value given in p0[3].

    Returns
    -------
    ans : tuple
        ans[0] = Lambda function with the fitted parameters
        ans[1] = Fit parameters

    """

    if p0 is None:
        p0 = np.zeros(4)
        p0[0] = y.max()
        p0[1] = np.where(y == y.max())[0][0]
        p0[2] = 3
        p0[3] = 1

    p0 = np.array(p0)
    p = copy.deepcopy(p0)

    def gauss(t, m, mu, sigma, bg):
        return m * np.exp(-(t - mu) ** 2 / (2 * sigma ** 2)) + bg

    fit_pars = np.array([True, fit_center, True, fit_background])

    if not fit_center and fit_background:
        p[fit_pars] = curve_fit(lambda a, b, c, d: gauss(a, b, p0[1], c, d), x, y, p0[fit_pars])[0]
    elif fit_center and not fit_background:
        p[fit_pars] = curve_fit(lambda a, b, c, d: gauss(a, b, c, d, p0[3]), x, y, p0[fit_pars])[0]
    elif not fit_center and not fit_background:
        p[fit_pars] = curve_fit(lambda a, b, c: gauss(a, b, p0[1], c, p0[3]), x, y, p0[fit_pars])[0]
    else:
        # noinspection PyTypeChecker
        p = curve_fit(gauss, x, y, p0)[0]

    def fit(t):
        return p[0] * np.exp(-(t - p[1]) ** 2 / (2 * p[2] ** 2)) + p[3]

    p[2] = np.abs(p[2])
    return fit, p


def full_width_half_maximum(x: np.ndarray, y: np.ndarray, bg: list) -> float:
    """
    Evaluates the full width half maximum of y in units of x.

    Parameters
    ----------
    x : numpy.array
    y : numpy.array
    bg : list
      Background sampling limits

    Returns
    -------
    full_width : number
      Full width half maximum
    """

    x_new, y_new = remove_bg(x, y, bg)

    f = UnivariateSpline(x_new, y_new / max(y_new) - .5, s=0)
    full_width = f.roots()[1] - f.roots()[0]

    return full_width


def blackbody(x, t, coordinate='wavelength'):
    """
    Evaluates the blackbody spectrum for a given temperature.

    Parameters
    -----------
    x : numpy.array
      Wavelength or frequency coordinates (CGS).
    t : number
      Blackbody temperature in Kelvin
    coordinate : string
      Specify the coordinates of x as 'wavelength' or 'frequency'.

    Returns
    --------
    b(x, T) : numpy.array
      Flux density in cgs units.
    """

    h = 6.6261e-27  # cm**2*g/s
    c = 2.998e+10  # cm/s
    kb = 1.3806488e-16  # erg/K

    if coordinate == 'wavelength':
        def b(z, u):
            return 2. * h * c ** 2. / z ** 5 * (1. / (np.exp(h * c / (z * kb * u)) - 1.))
    elif coordinate == 'frequency':
        def b(z, u):
            return 2 * h * c ** 2 / z ** 5 * (np.exp((h * c) / (z * kb * u)) - 1) ** (-1)
    else:
        raise RuntimeError('Coordinate type "{:s}" not recognized.'.format(coordinate))

    return b(x, t)


def natural_sort(l):
    # This code is not mine! I copied it from
    # http://stackoverflow.com/questions/4836710/does-python-have-a-built-in-
    # function-for-string-natural-sort

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(l, key=alphanum_key)


def closest(arr, value):
    """
    Returns the index of the array element that is numerically closest
    to the given value. The returned variable is an integer. This
    function expects a one-dimensional array.

    >>> closest(np.arange(5), 3.7)
    4
    """

    idx = np.abs(arr - value).argmin()
    return idx


def get_wl(image, dimension=0, hdrext=0, dataext=0, dwlkey='CD1_1', wl0key='CRVAL1', pix0key='CRPIX1'):
    """
    Obtains the wavelenght coordinates from the header keywords of the
    FITS file image. The default keywords are CD1_1 for the delta
    lambda, CRVAL for the value of the first pixel and CRPIX1 for the
    number of the first pixel. These keywords are the standard for
    GEMINI images.

    The function is prepared to work with Multi-Extesion FITS (MEF)
    files.

    Parameters
    ----------
    image : string
        Name of the FITS file containing the spectrum
    dimension : integer
        Dimension of the dispersion direction
    hdrext : number
        Extension that contains the header
    dataext : number
        Extension that contains the actual spectral data
    dwlkey : string
        Header keyword for the interval between two data points
    wl0key : string
        Header keyword for the first pixel value
    pix0key : string
        Header keyword for the first pixel coordinate

    Returns
    -------
    wl : numpy.array
        Wavelength coordinates for each data point
    """

    h = pf.getheader(image, ext=hdrext)
    dwl, wl1, pix0 = [float(h[i]) for i in [dwlkey, wl0key, pix0key]]
    npoints = np.shape(pf.getdata(image, dataext))[dimension]
    wl = wl1 + (np.arange(1, npoints + 1, dtype='float64') - pix0) * dwl

    try:
        if h['dc-flag'] == 1:
            wl = 10. ** wl
    except KeyError:
        pass

    return wl


def normspec(x, y, wl, span):
    """
    Normalizes a copy of a 1D spectrum given as arr_in, with the
    wavelength as arr_in[:,0] and the flux as arr_in[:,1]. The
    normalization value is the average flux between wl-span/2. and
    wl+span/2..

    Parameters
    -----------
    x : numpy.array
      Input wavelength coordinates
    y : numpy.array
      Input flux coordinates
    wl : number
      Central wavelength for normalization
    span : number
      Width of normalization window
    """

    # arr = copy(column_stack([x,y]))

    y2 = copy.deepcopy(y)

    # if closest(x,wl-span/2.) == closest(x,wl+span/2.):
    if wl - span / 2. < x[0] or wl + span / 2. > x[-1]:
        print('ERROR: Normalization wavelength outside data limits.')
        return

    f = interp1d(x, y2)
    y2 = y2 / np.average(f(np.linspace(wl - span / 2., wl + span / 2., 1000)))

    return y2


def flags_to_mask(wavelength: np.ndarray, flags: np.ndarray) -> list:
    """
    Converts an array of flags into a list of masked intervals.

    Parameters
    ----------
    wavelength : numpy.ndarray
        Wavelength coordinates of the flags. These values will be the ones
        used in the mask list.
    flags : numpy.ndarray
        Flags vector.

    Returns
    -------
    new_mask : list
        A list of wavelength pairs defining masked regions.

    """
    new_mask = []
    c = 0
    d = len(wavelength)
    while c < d:
        region = []
        if flags[c] == 1:
            region.append(wavelength[c])
            while c < d:
                if flags[c] == 1:
                    c += 1
                else:
                    break
            region.append(wavelength[c - 1])
            new_mask.append(region)
        else:
            c += 1

    return new_mask


def continuum(x, y, output='ratio', degree=6, n_iterate=5, lower_threshold=2, upper_threshold=3, verbose=False,
              weights=None):
    """
    Builds a polynomial continuum from segments of a spectrum,
    given in the form of wl and flux arrays.

    Parameters
    ----------
    x : array-like
        Independent variable
    y : array-like
        y = f(x)
    output: string
        Specifies what will be returned by the function

        'ratio'      = ratio between fitted continuum and the spectrum
        'difference' = difference between fitted continuum and the spectrum
        'function'   = continuum function evaluated at x

    degree : integer
        Degree of polynomial for the fit
    n_iterate : integer
        Number of rejection iterations
    lower_threshold : float
        Lower threshold for point rejection in units of standard
        deviation of the residuals
    upper_threshold : float
        Upper threshold for point rejection in units of standard
        deviation of the residuals
    verbose : boolean
        Prints information about the fitting
    weights : array-like
        Weights for continuum fitting. Must be the shape of x and y.

    Returns
    -------
    c : tuple

        c[0]: numpy.ndarray
            Input x coordinates
        c[1]: numpy.ndarray
            See parameter "output".

    """

    assert not np.isnan(x).all(), 'All x values are NaN.'
    assert not np.isnan(y).all(), 'All y values are NaN.'

    x_full = copy.deepcopy(x)
    # NOTE: For now, interp1d skips interpolation of NaNs.
    s = interp1d(x, y)

    if weights is None:
        weights = np.ones_like(x)

    if np.isnan(y).any():
        nan_mask = np.isnan(s(x))
        x = x[~nan_mask]
        weights = copy.deepcopy(weights)[~nan_mask]
        warnings.warn(
            'NaN values found in data! Removed {:d} out of {:d} data points.'.format(
                np.count_nonzero(nan_mask), len(x_full)),
            category=RuntimeWarning,
        )

    def f(z):
        return np.polyval(np.polyfit(z, s(z), deg=degree, w=weights), z)

    for i in range(n_iterate):

        res = s(x) - f(x)
        sig = np.std(res)
        rej_cond = ((res < upper_threshold * sig) & (res > -lower_threshold * sig))

        if np.sum(rej_cond) <= degree:
            if verbose:
                warnings.warn('Not enough fitting points. Stopped at iteration {:d}. sig={:.2e}'.format(i, sig))
            break

        if np.sum(weights == 0.0) >= degree:
            if verbose:
                warnings.warn(
                    'Number of non-zero values in weights vector is lower than the polynomial degree. '
                    'Stopped at iteration {:d}. sig={:.2e}'.format(i, sig))
            break

        x = x[rej_cond]
        weights = weights[rej_cond]

    if verbose:
        print('Final number of points used in the fit: {:d}'
              .format(len(x)))
        print('Rejection ratio: {:.2f}'
              .format(1. - float(len(x)) / float(len(x_full))))

    p = np.polyfit(x, s(x), deg=degree, w=weights)

    out = dict(
        ratio=(x_full, s(x_full) / np.polyval(p, x_full)),
        difference=(x_full, s(x_full) - np.polyval(p, x_full)),
        function=(x_full, np.polyval(p, x_full)),
        polynomial=p,
    )

    return out[output]


def eqw(wl, flux, limits, continuum_iterate=5):
    """
    Measure the equivalent width of a feature in `arr`, defined by `lims`:


    """

    f_spectrum = interp1d(wl, flux)

    f_continuum, continuum_wavelength = continuum(wl, flux, limits[2:], n_iterate=continuum_iterate)

    # area under continuum
    continuum_integral = trapz(np.polyval(f_continuum, np.linspace(limits[0], limits[1])),
                               x=np.linspace(limits[0], limits[1]))

    # area under spectrum
    spectrum_integral = trapz(f_spectrum(np.linspace(limits[0], limits[1])), np.linspace(limits[0], limits[1]))

    w = ((continuum_integral - spectrum_integral) / continuum_integral) * (limits[1] - limits[0])

    # Calculation of the error in the equivalent width, following the
    # definition of Vollmann & Eversberg 2006 (doi: 10.1002/asna.200610645)

    sn = np.average(f_spectrum(continuum_wavelength)) / np.std(f_spectrum(continuum_wavelength))
    sigma_eqw = np.sqrt(1 + np.average(np.polyval(f_continuum, continuum_wavelength)) / np.average(flux)) * (
            limits[1] - limits[0] - w) / sn

    return w, sigma_eqw


def joinspec(x1, y1, x2, y2):
    """
    Joins two spectra

    Parameters
    -----------
    x1 : array
        Wavelength coordinates of the first spectrum
    y1 : array
        Flux coordinates of the first spectrum
    x2 : array
        Wavelength coordinates of the second spectrum
    y2 : array
        Flux coordinates of the second spectrum

    Returns
    --------
    x : array
        Joined spectrum wavelength coordinates
    f : array
        Flux coordinates
    """

    f1 = interp1d(x1, y1, bounds_error='false', fill_value=0)
    f2 = interp1d(x2, y2, bounds_error='false', fill_value=0)

    if np.average(np.diff(x1)) <= np.average(np.diff(x2)):
        dx = np.average(np.diff(x1))
    else:
        dx = np.average(np.diff(x2))

    x = np.arange(x1[0], x2[-1], dx)

    f = f1(x[x < x2[0]])

    f = np.append(f, np.average(np.array([
        f1(x[(x >= x2[0]) & (x < x1[-1])]),
        f2(x[(x >= x2[0]) & (x < x1[-1])])]), axis=0))

    f = np.append(f, f2(x[x >= x1[-1]]))

    return x, f


def fnu2flambda(fnu, l):
    """
    Converts between flux units

    Parameters
    -----------
    fnu : number
        Flux density in W/m^2/Hz
    l : number
        Wavelength in microns

    Returns
    --------
    flambda: number
        Flux density in W/m^2/um

    Where does it come from?
      d nu            c
    -------- = - ----------
    d lambda      lambda^2
    """

    flambda = 2.99792458 * fnu * 10 ** (-12) / l ** 2

    return flambda


def flambda2fnu(wl, fl):
    """
    Converts a flambda to fnu.

    Parameters
    -----------
    wl : 1d-array
        Wavelength in angstroms
    fl : 1d-array
        Flux in ergs/s/cm^2/A

    Returns
    --------
    fnu : 1d-array
        Flux in Jy (10^23 erg/s/cm^2/Hz)
    """

    fnu = 3.33564095e+4 * fl * wl ** 2

    return fnu


def mask(arr, maskfile):
    """
    Eliminates masked points from a spectrum.

    Parameters
    -----------
    arr : ndarray
        Input spectrum for masking. The array can have
        any number of columns, provided that the wavelength
        is in the first one.
    maskfile : string
        Name of the ASCII file containing the mask definitions,
        with one region per line defined by a lower and an
        upper limit in this order.

    Returns
    --------
    b : ndarray
        An exact copy of arr without the points that lie
        within the regions defined by maskfile.
    """
    m = np.loadtxt(maskfile)
    a = copy.deepcopy(arr)

    for i in range(len(m)):
        a[(a[:, 0] >= m[i, 0]) & (a[:, 0] <= m[i, 1])] = 0

    b = a[a[:, 0] != 0]

    return b


def spectrophotometry(spec: Callable, transmission: Callable, limits: Iterable, verbose: bool = False,
                      get_filter_center: bool = False):
    """
    Evaluates the integrated flux for a given filter
    over a spectrum.

    Parameters
    -----------
    spec : function
        A function that describes the spectrum with only
        the wavelength or frequency as parameter.
    transmission : function
        The same as the above but for the filter.
    limits: iterable
        Lower and upper limits for the integration.
    verbose : bool
        Verbosity.
    get_filter_center : bool
        Returns the pivot wavelength.

    Returns
    --------
    photometry : float
        Photometric data point that results from the
        integration and scaling of the filter over the
        spectrum (CGS).

    Notes:
    ------
    This method employs the scipy.integrate.quad function
    to perform all the integrations, with a limit of 1000
    and epsrel of 1.e-3.
    """

    l, er = 100, 1.e-2
    x0, x1 = limits

    def y2(x):
        return transmission(x) * spec(x)

    if get_filter_center:
        def median_function(x):
            return np.abs(quad(transmission, -np.inf, x, epsrel=er, limit=l)[0] -
                          quad(transmission, x, +np.inf, epsrel=er, limit=l)[0])

        f_center = minimize(median_function, np.array([11.5]), bounds=[[11.1, 11.9]], method='slsqp')
    else:
        f_center = np.nan

    cal = quad(transmission, x0, x1, limit=l, epsrel=er)[0]
    photometry = quad(y2, x0, x1, limit=l, epsrel=er)[0] / cal

    if verbose:
        print('Central wavelength: {:.2f}; Flux: {:.2f}'.format(float(f_center), photometry))

    if get_filter_center:
        return photometry, float(f_center['x'])
    else:
        return photometry


def w80eval(wl: np.ndarray, spec: np.ndarray, wl0: float, smooth: float = None) -> tuple:
    """
    Evaluates the W80 parameter of a given emission fature.

    Parameters
    ----------
    wl : array-like
      Wavelength vector.
    spec : array-like
      Flux vector.
    wl0 : number
      Central wavelength of the emission feature.
    smooth : float
        Smoothing sigma to apply after the cumulative sum.

    Returns
    -------
    w80 : float
      The resulting w80 parameter.
    r0 : float
    r1 : float
    velocity : numpy.ndarray
        Velocity vector.
    velocity_spectrum : numpy.ndarray
        Spectrum in velocity coordinates.

    Notes
    -----
    W80 is the width in velocity space which encompasses 80% of the
    light emitted in a given spectral feature. It is widely used as
    a proxy for identifying outflows of ionized gas in active galaxies.
    For instance, see Zakamska+2014 MNRAS.
    """

    velocity = (wl * units.angstrom).to(units.km / units.s,
                                        equivalencies=units.doppler_relativistic(wl0 * units.angstrom))

    if smooth is not None:
        cumulative = cumtrapz(gaussian_filter(spec, smooth), velocity, initial=0)
    else:
        cumulative = cumtrapz(spec, velocity, initial=0)
    cumulative /= cumulative.max()

    r0 = velocity[(np.abs(cumulative - 0.1)).argsort()[0]].value
    r1 = velocity[(np.abs(cumulative - 0.9)).argsort()[0]].value
    w80 = r1 - r0
    return w80, r0, r1, velocity, spec


class Constraints:

    def __init__(self):
        pass

    @staticmethod
    def redshift(wla, wlb, rest0, rest1):
        def func(x):
            return (x[wla] / rest0) - (x[wlb] / rest1)

        d = dict(type='eq', fun=func)

        return d

    @staticmethod
    def sigma(sa, sb, wla, wlb):
        def func(x):
            return x[sa] / x[wla] - x[sb] / x[wlb]

        d = dict(type='eq', fun=func)

        return d

    @staticmethod
    def same(ha, hb):
        def func(x):
            return x[ha] - x[hb]

        d = dict(type='eq', fun=func)

        return d
