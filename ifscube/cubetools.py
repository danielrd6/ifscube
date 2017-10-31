"""
Functions for the analysis of integral field spectroscopy.

Author: Daniel Ruschel Dutra
Website: https://github.com/danielrd6/ifscube
"""
import numpy as np
import astropy.io.fits as pf
import ifscube.spectools as st
import matplotlib.pyplot as plt
from scipy.integrate import trapz, fixed_quad
from copy import deepcopy
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy import ndimage
import ifscube.elprofile as lprof
from numpy import ma
from astropy import constants
from astropy import units
from scipy.ndimage import gaussian_filter
import ifscube.plots as ifsplots


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


def w80eval(wl, spec, wl0, **min_args):
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
    **min_args : dictionary
      Options passed directly to the scipy.optimize.minimize.

    Returns
    -------
    w80 : number
      The resulting w80 parameter.

    Description
    -----------
    W80 is the width in velocity space which encompasses 80% of the
    light emitted in a given spectral feature. It is widely used as
    a proxy for identifying outflows of ionized gas in active galaxies.
    """

    # First we begin by transforming from wavelength space to velocity
    # space.

    velocity = (wl * units.angstrom).to(
        units.km / units.s,
        equivalencies=units.doppler_relativistic(wl0 * units.angstrom),
    )

    f_norm = np.mean(spec)

    new_spec = deepcopy(spec) / f_norm
    # new_spec[new_spec < 0] = 0

    s = interp1d(velocity, new_spec, fill_value=0, bounds_error=False)
    total_flux = trapz(new_spec, velocity.value)
    tf80 = 0.8 * total_flux

    def res(p):
        return np.square(fixed_quad(s, p[0], p[1], n=7)[0] - tf80)

    # In order to have a good initial guess, the code will find the
    # the Half-Width at Half Maximum (hwhm) of the specified feature.

    for i in np.linspace(0, velocity[-1]):
        if s(i) <= new_spec.max() / 2:
            hwhm = i
            break

    # In some cases, when the spectral feature has a very low
    # amplitude, the algorithm might have trouble finding the half
    # width at half maximum (hwhm), which is used as an initial guess
    # for the w80. In such cases the loop above will reach the end of
    # the spectrum without finding a value below half the amplitude of
    # the spectral feature, and consequently the *hwhm* variable will
    # not be set.  This next conditional expression sets w80 to
    # numpy.nan when such events occur.

    if 'hwhm' in locals():
        r = minimize(res, x0=[-hwhm.value, hwhm.value], **min_args)
        w80 = r.x[1] - r.x[0]
    else:
        w80 = np.nan

    return w80, r.x[0], r.x[1], velocity, s(velocity)


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


def progress(x, xmax, steps=10):
    try:
        if x % (xmax / steps) == 0:
            print('{:2.0f}%\r'.format(np.float(x) / np.float(xmax) * 100))
    except ZeroDivisionError:
        pass


def bound_updater(p0, bound_range):

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

    return newbound


class gmosdc:
    """
    A class for dealing with data cubes, originally written to work
    with GMOS IFU.
    """

    def __init__(self, fitsfile, redshift=None, vortab=None,
                 dataext=1, hdrext=0, var_ext=None, ncubes_ext=None,
                 nan_spaxels='all'):
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

        hdulist = pf.open(fitsfile)

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

        self.wl = st.get_wl(
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

        if ncubes_ext is not None:
            # The self.ncubes variable describes how many different
            # pixels contributed to the final combined pixel. This can
            # also serve as a flag, when zero cubes contributed to the
            # pixel. Additionaly, it may be useful to mask regions that
            # are present in only one observation, for greater
            # confidence.
            self.ncubes = hdulist[ncubes_ext].data

        try:
            if self.header['VORBIN']:
                vortab = pf.open(fitsfile)['VOR'].data
                self.voronoi_tab = vortab
                self.binned = True
        except KeyError:
            self.binned = False

        self.fitsfile = fitsfile
        self.redshift = redshift
        self.spec_indices = np.column_stack([
            np.ravel(np.indices(np.shape(self.data)[1:])[0]),
            np.ravel(np.indices(np.shape(self.data)[1:])[1])
            ])

    def continuum(self, writefits=False, outimage=None,
                  fitting_window=None, copts=None):
        """
        Evaluates a polynomial continuum for the whole cube and stores
        it in self.cont.
        """

        if self.binned:
            v = self.voronoi_tab
            xy = np.column_stack([
                v[np.unique(v['binNum'], return_index=True)[1]][coords]
                for coords in ['xcoords', 'ycoords']])
        else:
            xy = self.spec_indices

        fw = fitting_window
        fwidx = (self.restwl > fw[0]) & (self.restwl < fw[1])

        wl = deepcopy(self.restwl[fwidx])
        data = deepcopy(self.data[fwidx])

        c = np.zeros(np.shape(data), dtype='float32')

        # nspec = len(xy)

        if copts is None:
            copts = {'degr': 3, 'upper_threshold': 2,
                     'lower_threshold': 2, 'niterate': 5}

        try:
            copts['returns']
        except KeyError:
            copts['returns'] = 'function'

        for k, h in enumerate(xy):
            i, j = h
            s = deepcopy(data[:, i, j])
            if (any(s[:20]) and any(s[-20:])) or \
                    (any(np.isnan(s[:20])) and any(np.isnan(s[-20:]))):
                try:
                    cont = st.continuum(wl, s, **copts)
                    if self.binned:
                        for l, m in v[v[:, 2] == k, :2]:
                            c[:, l, m] = cont[1]
                    else:
                        c[:, i, j] = cont[1]
                except TypeError:
                    print(
                        'Could not find a solution for {:d},{:d}.'
                        .format(i, j))
                    c[:, i, j] = np.nan
                except ValueError:
                    c[:, i, j] = np.nan
            else:
                c[:, i, j] = np.nan

        self.cont = c

        if writefits:
            if outimage is None:
                outimage = self.fitsfile.replace('.fits', '_continuum.fits')

            hdr = deepcopy(self.header_data)

            try:
                hdr['REDSHIFT'] = self.redshift
            except KeyError:
                hdr['REDSHIFT'] = (self.redshift, 'Redshift used in GMOSDC')

            hdr['CRVAL3'] = wl[0]
            hdr['CONTDEGR'] = (copts['degr'],
                               'Degree of continuum polynomial')
            hdr['CONTNITE'] = (copts['niterate'],
                               'Continuum rejection iterations')
            hdr['CONTLTR'] = (copts['lower_threshold'],
                              'Continuum lower threshold')
            hdr['CONTHTR'] = (copts['upper_threshold'],
                              'Continuum upper threshold')

            pf.writeto(outimage, data=c, header=hdr)

        return c

    def snr_eval(self, wl_range=[6050, 6200], copts=None):
        """
        Measures the signal to noise ratio (SNR) for each spectrum in a
        data cube, returning an image of the SNR.

        Parameters:
        -----------
        self : gmosdc instance
            gmosdc object
        wl_range : array like
            An array like object containing two wavelength coordinates
            that define the SNR window at the rest frame.
        copts : dictionary
            Options for the continuum fitting function.

        Returns:
        --------
        snr : numpy.ndarray
            Image of the SNR for each spectrum.

        Description:
        ------------
            This method evaluates the SNR for each spectrum in a data
            cube by measuring the residuals of a polynomial continuum
            fit. The function CONTINUUM of the SPECTOOLS package is used
            to provide the continuum, with zero rejection iterations
            and a 3 order polynomial.
        """

        noise = np.zeros(np.shape(self.data)[1:], dtype='float32')
        signal = np.zeros(np.shape(self.data)[1:], dtype='float32')
        snrwindow = (self.restwl >= wl_range[0]) &\
            (self.restwl <= wl_range[1])
        data = deepcopy(self.data)

        wl = self.restwl[snrwindow]

        if copts is None:
            copts = {'niterate': 0, 'degr': 3, 'upper_threshold': 3,
                     'lower_threshold': 3, 'returns': 'function'}
        else:
            copts['returns'] = 'function'

        for i, j in self.spec_indices:
            if any(data[snrwindow, i, j]) and\
                    all(~np.isnan(data[snrwindow, i, j])):
                s = data[snrwindow, i, j]
                cont = st.continuum(wl, s, **copts)[1]
                noise[i, j] = np.nanstd(s - cont)
                signal[i, j] = np.nanmean(cont)
            else:
                noise[i, j], signal[i, j] = np.nan, np.nan

        self.noise = noise
        self.signal = signal

        return np.array([signal, noise])

    def wlprojection(self, wl0, fwhm, filtertype='box', writefits=False,
                     outimage='outimage.fits'):
        """
        Writes a projection of the data cube along the wavelength
        coordinate, with the flux given by a given type of filter.

        Parameters:
        -----------
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
        outimage : string
            Name of the output image

        Returns:
        --------
        Nothing.
        """

        outim = wlprojection(
            arr=self.data, wl=self.restwl, wl0=wl0, fwhm=fwhm,
            filtertype=filtertype)

        if writefits:

            hdr = deepcopy(self.header)

            try:
                hdr['REDSHIFT'] = self.redshift
            except KeyError:
                hdr['REDSHIFT'] = (self.redshift, 'Redshift used in GMOSDC')

            hdr['WLPROJ'] = (True, 'Processed by WLPROJECTION?')
            hdr['WLPRTYPE'] = (filtertype,
                               'Type of filter used in projection.')
            hdr['WLPRWL0'] = (wl0, 'Central wavelength of the filter.')
            hdr['WLPRFWHM'] = (fwhm, 'FWHM of the projection filter.')

            pf.writeto(outimage, data=outim, header=hdr)

        return outim

    def plotspec(self, x, y, noise_smooth=30, ax=None):
        """
        Plots the spectrum at coordinates x,y.

        Parameters
        ----------
        x,y : numbers or iterables
            If x and y are numbers plots the spectrum at the specific
            spaxel. If x and y are two element tuples plots the average
            between x[0],y[0] and x[1],y[1]

        Returns
        -------
        Nothing.
        """

        # fig = plt.figure(1)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if hasattr(x, '__iter__') and hasattr(y, '__iter__'):
            s = np.average(
                    np.average(self.data[:, y[0]:y[1], x[0]:x[1]], 1), 1)
        elif hasattr(x, '__iter__') and not hasattr(y, '__iter__'):
            s = np.average(self.data[:, y, x[0]:x[1]], 1)
        elif not hasattr(x, '__iter__') and hasattr(y, '__iter__'):
            s = np.average(self.data[:, y[0]:y[1], x], 1)
        else:
            s = self.data[:, y, x]

        ax.plot(self.restwl, s)

        if hasattr(self, 'noise_cube'):

            if hasattr(x, '__iter__') and hasattr(y, '__iter__'):
                n = np.average(
                    np.average(
                        self.noise_cube[:, y[0]:y[1], x[0]:x[1]], 1
                    ), 1
                )
            elif hasattr(x, '__iter__') and not hasattr(y, '__iter__'):
                n = np.average(self.noise_cube[:, y, x[0]:x[1]], 1)
            elif not hasattr(x, '__iter__') and hasattr(y, '__iter__'):
                n = np.average(self.noise_cube[:, y[0]:y[1], x], 1)
            else:
                n = self.noise_cube[:, y, x]

            n = ndimage.gaussian_filter1d(n, noise_smooth)
            sg = ndimage.gaussian_filter1d(s, noise_smooth)

            ax.fill_between(self.restwl, sg - n, sg + n, edgecolor='',
                            alpha=0.2, color='green')

        plt.show()

    def linefit(self, p0, function='gaussian', fitting_window=None,
                writefits=False, outimage=None, variance=None,
                constraints=(), bounds=None, inst_disp=1.0,
                individual_spec=False, min_method='SLSQP',
                minopts={'eps': 1e-3}, copts=None, refit=False,
                update_bounds=False, bound_range=.1, spiral_loop=False,
                spiral_center=None, fit_continuum=True, refit_radius=3,
                sig_threshold=0):

        """
        Fits a spectral feature with a gaussian function and returns a
        map of measured properties. This is a wrapper for the scipy
        minimize function that basically iterates over the cube,
        has a formula for the reduced chi squared, and applies
        an internal scale factor to the flux.

        Parameters
        ----------
        p0 : iterable
            Initial guess for the fitting funcion, consisting of a list
            of 3N parameters for N components of **function**. In the
            case of a gaussian fucntion, these parameters must be given
            as [amplitude0, center0, sigma0, amplitude1, center1, ...].
        function : string
            The function to be fitted to the spectral features.
            Available options and respective parameters are:
                'gaussian' : amplitude, central wavelength in angstroms,
                    sigma in angstroms
                'gauss_hermite' : amplitude, central wavelength in
                    angstroms, sigma in angstroms, h3 and h4
        fitting_window : iterable
            Lower and upper wavelength limits for the fitting
            algorithm. These limits should allow for a considerable
            portion of continuum besides the desired spectral features.
        writefits : boolean
            Writes the results in a FITS file.
        outimage : string
            Name of the FITS file in which to write the results.
        variance : float, 1D, 2D or 3D array
            The variance of the flux measurments. It can be given
            in one of four formats. If variance is a float it is
            applied as a contant to the whole spectrum. If given as 1D
            array it assumed to be a spectrum that will be applied to
            the whole cube. As 2D array, each spaxel will be applied
            equally to all wavelenths. Finally the 3D array must
            represent the variance for each elemente of the data cube.
            It defaults to None, in which case it does not affect the
            minimization algorithm, and the returned Chi2 will be in
            fact just the fit residuals.
        inst_disp : number
            Instrumental dispersion in pixel units. This argument is
            used to evaluate the reduced chi squared. If let to default
            it is assumed that each wavelength coordinate is a degree
            of freedom. The physically sound way to do it is to use the
            number of dispersion elements in a spectrum as the degrees
            of freedom.
        bounds : sequence
            Bounds for the fitting algorithm, given as a list of
            [xmin, xmax] pairs for each x parameter.
        constraints : dict or sequence of dicts
            See scipy.optimize.minimize
        min_method : string
            Minimization method. See scipy.optimize.minimize.
        minopts : dict
            Dictionary of options to be passed to the minimization
            routine. See scipy.optimize.minimize.
        individual_spec : False or x,y pair
            Pixel coordinates for the spectrum you wish to fit
            individually.
        copts : dict
            Arguments to be passed to the spectools.continuum function.
        refit : boolean
            Use parameters from nearby sucessful fits as the initial
            guess for the next fit.
        update_bounds : boolean
            If using refit, update the bounds for the next fit.
        bound_range : number
            Fractional difference for updating the bounds when using refit.
        spiral_loop : boolean
            Begins the fitting with the central spaxel and continues
            spiraling outwards.
        spiral_center : iterable
            Central coordinates for the beginning of the spiral given
            as a list of two coordinates [x0, y0]
        fit_continuum : boolean
            If True fits the continuum just before attempting to fit
            the emission lines. Setting this option to False will
            cause the algorithm to look for self.cont, which should
            contain a data cube of continua.

        Returns
        -------
        sol : numpy.ndarray
            A data cube with the solution for each spectrum occupying
            the respective position in the image, and each position in
            the first axis giving the different parameters of the fit.

        See also
        --------
        scipy.optimize.curve_fit, scipy.optimize.leastsq
        """

        if function == 'gaussian':
            fit_func = lprof.gauss
            self.fit_func = lprof.gauss
            npars_pc = 3
        elif function == 'gauss_hermite':
            fit_func = lprof.gausshermite
            self.fit_func = lprof.gausshermite
            npars_pc = 5
        else:
            raise NameError('Unknown function "{:s}".'.format(function))

        if fitting_window is not None:
            fw = (self.restwl > fitting_window[0]) &\
                 (self.restwl < fitting_window[1])
        else:
            fw = Ellipsis

        if copts is None:
            copts = {
                'niterate': 5, 'degr': 4, 'upper_threshold': 2,
                'lower_threshold': 2}

        copts['returns'] = 'function'

        wl = deepcopy(self.restwl[fw])
        scale_factor = np.nanmean(self.data[fw, :, :])
        data = deepcopy(self.data[fw, :, :]) / scale_factor
        fit_status = np.ones(np.shape(data)[1:], dtype='float32') * -1

        try:
            vcube = (self.noise_cube[fw, :, :] ** 2) / (scale_factor ** 2)
        except AttributeError:
            vcube = np.ones(np.shape(data), dtype='float32')

        if variance is not None:
            if len(np.shape(variance)) == 0:
                vcube *= variance
            elif len(np.shape(variance)) == 1:
                for i, j in self.spec_indices:
                    vcube[:, i, j] = variance
            elif len(np.shape(variance)) == 2:
                for i, j in enumerate(vcube):
                    vcube[i] = variance
            elif len(np.shape(variance)) == 3:
                vcube = variance[fw, :, :]

            vcube /= scale_factor ** 2

        npars = len(p0)
        nan_solution = np.array([np.nan for i in range(npars+1)])
        sol = np.zeros(
            (npars+1, np.shape(self.data)[1], np.shape(self.data)[2]),
            dtype='float32')
        self.fitcont = np.zeros(np.shape(data), dtype='float32')
        self.fitwl = wl
        self.fitspec = np.zeros(np.shape(data), dtype='float32')
        self.resultspec = np.zeros(np.shape(data), dtype='float32')

        if self.binned:
            v = self.voronoi_tab
            xy = np.column_stack([
                v[np.unique(v['binNum'], return_index=True)[1]][coords]
                for coords in ['ycoords', 'xcoords']])
            vor = np.column_stack([
                v[coords] for coords in ['ycoords', 'xcoords', 'binNum']])
        else:
            xy = self.spec_indices

        # Scale factor for the flux. Needed to avoid problems with
        # the minimization algorithm.
        flux_sf = np.ones(npars, dtype='float32')
        flux_sf[np.arange(0, npars, npars_pc)] *= scale_factor
        p0 /= flux_sf
        if bounds is not None:
            bounds = np.array(bounds)
            for i, j in enumerate(bounds):
                j /= flux_sf[i]

        Y, X = np.indices(np.shape(data)[1:])

        if individual_spec:
            xy = [individual_spec[::-1]]
        elif spiral_loop:
            if self.binned:
                y, x = xy[:, 0], xy[:, 1]
            else:
                y, x = self.spec_indices[:, 0], self.spec_indices[:, 1]
            if spiral_center is None:
                r = np.sqrt((x - x.max() / 2.) ** 2 + (y - y.max() / 2.) ** 2)
            else:
                r = np.sqrt(
                    (x - spiral_center[0]) ** 2 + (y - spiral_center[1]) ** 2)
            t = np.arctan2(y - y.max()/2., x - x.max()/2.)
            t[t < 0] += 2 * np.pi

            b = np.array([
                (np.ravel(r)[i], np.ravel(t)[i]) for i in
                range(len(np.ravel(r)))], dtype=[
                    ('radius', 'f8'), ('angle', 'f8')])

            s = np.argsort(b, axis=0, order=['radius', 'angle'])
            xy = np.column_stack([np.ravel(y)[s], np.ravel(x)[s]])

        nspec = len(xy)
        for k, h in enumerate(xy):
            progress(k, nspec, 10)
            i, j = h
            if self.binned:
                binNum = vor[(vor[:, 0] == i) & (vor[:, 1] == j), 2]
            if (~np.any(data[:20, i, j])) or\
                    ~np.any(data[-20:, i, j]) or\
                    np.any(np.isnan(data[:, i, j])):
                sol[:, i, j] = nan_solution
                continue
            v = vcube[:, i, j]
            if fit_continuum:
                cont = st.continuum(wl, data[:, i, j], **copts)[1]
            else:
                cont = self.cont[:, i, j]/scale_factor
            s = data[:, i, j] - cont

            # Avoids fitting if the spectrum is null.
            try:
                def res(x):
                    return np.sum((s - fit_func(self.fitwl, x)) ** 2 / v)

                if refit and k != 0:
                    radsol = np.sqrt((Y - i)**2 + (X - j)**2)
                    nearsol = sol[:-1, (radsol < refit_radius) &
                                  (fit_status == 0)]
                    if np.shape(nearsol) == (5, 1):
                        p0 = deepcopy(nearsol.transpose()/flux_sf)
                    elif np.any(nearsol):
                        p0 = deepcopy(
                            np.average(nearsol.transpose(), 0) / flux_sf)

                        if update_bounds:
                            bounds = bound_updater(p0, bound_range)

                r = minimize(res, x0=p0, method=min_method, bounds=bounds,
                             constraints=constraints, options=minopts)
                if r.status != 0:
                    print(h, r.message, r.status)
                # Reduced chi squared of the fit.
                chi2 = res(r['x'])
                nu = len(s)/inst_disp - npars - len(constraints) - 1
                red_chi2 = chi2 / nu
                p = np.append(r['x']*flux_sf, red_chi2)
                fit_status[i, j] = r.status
            except RuntimeError:
                print(
                    'Optimal parameters not found for spectrum {:d},{:d}'
                    .format(int(i), int(j)))
                p = nan_solution

            # Sets p to nan if the flux is smaller than the average
            # noise level.

            # mean noise level
            mnl = np.sqrt(np.sum(np.square(v * scale_factor)) / v.size)

            # The first argument of the gauss_hermite function is the
            # integrated flux, and not the amplitude. Therefore it is necessary
            # to make some sort of approximation for the flux of the noise,
            # were it to have a quasi-gaussian shape. This is not needed for
            # the gaussian function.
            if function == 'gauss_hermite':
                mnl_flux = mnl * p[2] * np.sqrt(2. * np.pi)
            elif function == 'gaussian':
                mnl_flux = mnl

            if p[0] < mnl_flux * sig_threshold:
                p = nan_solution
                fit_status[i, j] = 99

            if self.binned:
                for l, m in vor[vor[:, 2] == binNum, :2]:
                    sol[:, l, m] = p
                    self.fitcont[:, l, m] = cont * scale_factor
                    self.fitspec[:, l, m] = (s + cont) * scale_factor
                    self.resultspec[:, l, m] = (
                        cont+fit_func(self.fitwl, r['x'])) * scale_factor
            else:
                sol[:, i, j] = p
                self.fitcont[:, i, j] = cont*scale_factor
                self.fitspec[:, i, j] = (s+cont)*scale_factor
                self.resultspec[:, i, j] = (
                    cont + fit_func(self.fitwl, r['x'])) * scale_factor

        self.em_model = sol
        self.fit_status = fit_status
        p0 *= flux_sf

        if writefits:

            # Basic tests and first header
            if outimage is None:
                outimage = self.fitsfile.replace('.fits',
                                                 '_linefit.fits')
            hdr = deepcopy(self.header_data)
            try:
                hdr['REDSHIFT'] = self.redshift
            except KeyError:
                hdr['REDSHIFT'] = (self.redshift,
                                   'Redshift used in GMOSDC')

            # Creates MEF output.
            h = pf.HDUList()
            h.append(pf.PrimaryHDU(header=hdr))

            # Creates the fitted spectrum extension
            hdr = pf.Header()
            hdr['object'] = ('spectrum', 'Data in this extension')
            hdr['CRPIX3'] = (1, 'Reference pixel for wavelength')
            hdr['CRVAL3'] = (wl[0], 'Reference value for wavelength')
            hdr['CD3_3'] = (np.average(np.diff(wl)), 'CD3_3')
            h.append(pf.ImageHDU(data=self.fitspec, header=hdr))

            # Creates the fitted continuum extension.
            hdr['object'] = 'continuum'
            h.append(pf.ImageHDU(data=self.fitcont, header=hdr))

            # Creates the fitted function extension.
            hdr['object'] = 'fit'
            h.append(pf.ImageHDU(data=self.resultspec, header=hdr))

            # Creates the solution extension.
            hdr['object'] = 'parameters'
            hdr['function'] = (function, 'Fitted function')
            hdr['nfunc'] = (len(p)/3, 'Number of functions')
            h.append(pf.ImageHDU(data=sol, header=hdr))

            # Creates the minimize's exit status extension
            hdr['object'] = 'status'
            h.append(pf.ImageHDU(data=fit_status, header=hdr))

            h.writeto(outimage)

        if individual_spec:
            return wl, s * scale_factor, cont * scale_factor,\
                fit_func(wl, p[:-1]), r
        else:
            return sol

    def linefit_model(
            self, model, fitter, fitting_window=None, writefits=False,
            outimage=None, variance=None, inst_disp=1.0,
            individual_spec=False, copts={}, refit=False,
            update_bounds=False, bound_range=.1, spiral_loop=False,
            spiral_center=None, fit_continuum=True, refit_radius=3,
            goodfit_flags=(1, 2, 3, 4), minopts={}):

        """
        Fits a spectral feature with a gaussian function and returns a
        map of measured properties. This is a wrapper for the scipy
        minimize function that basically iterates over the cube,
        has a formula for the reduced chi squared, and applies
        an internal scale factor to the flux.

        Parameters
        ----------
        model : astropy.modeling.FittableModel 
            Fittable model for the spectral features.
        fitter : astropy.modeling.fitting object
            Fitter to apply to the model and the data.
        fitting_window : iterable
            Lower and upper wavelength limits for the fitting
            algorithm. These limits should allow for a considerable
            portion of continuum besides the desired spectral features.
        writefits : boolean
            Writes the results in a FITS file.
        outimage : string
            Name of the FITS file in which to write the results.
        variance : float, 1D, 2D or 3D array
            The variance of the flux measurments. It can be given
            in one of four formats. If variance is a float it is
            applied as a contant to the whole spectrum. If given as 1D
            array it assumed to be a spectrum that will be applied to
            the whole cube. As 2D array, each spaxel will be applied
            equally to all wavelenths. Finally the 3D array must
            represent the variance for each elemente of the data cube.
            It defaults to None, in which case it does not affect the
            minimization algorithm, and the returned Chi2 will be in
            fact just the fit residuals.
        inst_disp : number
            Instrumental dispersion in pixel units. This argument is
            used to evaluate the reduced chi squared. If let to default
            it is assumed that each wavelength coordinate is a degree
            of freedom. The physically sound way to do it is to use the
            number of dispersion elements in a spectrum as the degrees
            of freedom.
        bounds : sequence
            Bounds for the fitting algorithm, given as a list of
            [xmin, xmax] pairs for each x parameter.
        constraints : dict or sequence of dicts
            See scipy.optimize.minimize
        min_method : string
            Minimization method. See scipy.optimize.minimize.
        minopts : dict
            Dictionary of options to be passed to the minimization
            routine. See scipy.optimize.minimize.
        individual_spec : False or x,y pair
            Pixel coordinates for the spectrum you wish to fit
            individually.
        copts : dict
            Arguments to be passed to the spectools.continuum function.
        refit : boolean
            Use parameters from nearby sucessful fits as the initial
            guess for the next fit.
        update_bounds : boolean
            If using refit, update the bounds for the next fit.
        bound_range : number
            Fractional difference for updating the bounds when using refit.
        spiral_loop : boolean
            Begins the fitting with the central spaxel and continues
            spiraling outwards.
        spiral_center : iterable
            Central coordinates for the beginning of the spiral given
            as a list of two coordinates [x0, y0]
        fit_continuum : boolean
            If True fits the continuum just before attempting to fit
            the emission lines. Setting this option to False will
            cause the algorithm to look for self.cont, which should
            contain a data cube of continua.
        goodfit_flags : tuple
            Ierr flags from the fitter output that are to be accepted
            as good fits.

        Returns
        -------
        sol : numpy.ndarray
            A data cube with the solution for each spectrum occupying
            the respective position in the image, and each position in
            the first axis giving the different parameters of the fit.

        See also
        --------
        scipy.optimize.curve_fit, scipy.optimize.leastsq
        """

        if fitting_window is not None:
            fw = (self.restwl > fitting_window[0]) &\
                 (self.restwl < fitting_window[1])
        else:
            fw = Ellipsis

        default_copts = dict(
            niterate=5, degr=4, upper_threshold=2, lower_threshold=2)

        default_copts.update(copts)
        copts.update(returns='function')

        wl = deepcopy(self.restwl[fw])
        data = deepcopy(self.data[fw, :, :])
        fit_status = np.ones(np.shape(data)[1:], dtype='float32') * -1

        try:
            vcube = self.noise_cube[fw, :, :] ** 2
        except AttributeError:
            vcube = np.ones(np.shape(data), dtype='float32')

        if variance is not None:
            if len(np.shape(variance)) == 0:
                vcube *= variance
            elif len(np.shape(variance)) == 1:
                for i, j in self.spec_indices:
                    vcube[:, i, j] = variance
            elif len(np.shape(variance)) == 2:
                for i, j in enumerate(vcube):
                    vcube[i] = variance
            elif len(np.shape(variance)) == 3:
                vcube = variance[fw, :, :]

        npars = len(model.parameters)
        nan_solution = np.array([np.nan for i in range(npars+1)])
        sol = np.zeros(
            (npars+1, np.shape(self.data)[1], np.shape(self.data)[2]),
            dtype='float32')
        self.fitcont = np.zeros(np.shape(data), dtype='float32')
        self.fitwl = wl
        self.fitspec = np.zeros(np.shape(data), dtype='float32')
        self.resultspec = np.zeros(np.shape(data), dtype='float32')

        if self.binned:
            v = self.voronoi_tab
            xy = np.column_stack([
                v[np.unique(v['binNum'], return_index=True)[1]][coords]
                for coords in ['ycoords', 'xcoords']])
            vor = np.column_stack([
                v[coords] for coords in ['ycoords', 'xcoords', 'binNum']])
        else:
            xy = self.spec_indices

        Y, X = np.indices(np.shape(data)[1:])

        if individual_spec:
            xy = [individual_spec[::-1]]
        elif spiral_loop:
            if self.binned:
                y, x = xy[:, 0], xy[:, 1]
            else:
                y, x = self.spec_indices[:, 0], self.spec_indices[:, 1]
            if spiral_center is None:
                r = np.sqrt((x - x.max() / 2.) ** 2 + (y - y.max() / 2.) ** 2)
            else:
                r = np.sqrt(
                    (x - spiral_center[0]) ** 2 + (y - spiral_center[1]) ** 2)
            t = np.arctan2(y - y.max()/2., x - x.max()/2.)
            t[t < 0] += 2 * np.pi

            b = np.array([
                (np.ravel(r)[i], np.ravel(t)[i]) for i in
                range(len(np.ravel(r)))], dtype=[
                    ('radius', 'f8'), ('angle', 'f8')])

            s = np.argsort(b, axis=0, order=['radius', 'angle'])
            xy = np.column_stack([np.ravel(y)[s], np.ravel(x)[s]])

        nspec = len(xy)
        for k, h in enumerate(xy):
            progress(k, nspec, 10)
            i, j = h
            if self.binned:
                binNum = vor[(vor[:, 0] == i) & (vor[:, 1] == j), 2]
            if (~np.any(data[:20, i, j])) or\
                    ~np.any(data[-20:, i, j]) or\
                    np.any(np.isnan(data[:, i, j])):
                sol[:, i, j] = nan_solution
                continue
            v = vcube[:, i, j]
            if fit_continuum:
                cont = st.continuum(wl, data[:, i, j], **copts)[1]
            else:
                cont = self.cont[:, i, j]
            s = data[:, i, j] - cont

            # Avoids fitting if the spectrum is null.
            try:
                
                if refit and k != 0:

                    # Create radius mask
                    radsol = np.sqrt((Y - i)**2 + (X - j)**2)
                    radius_mask = (radsol < refit_radius)
                    
                    # Creates a good fit mask.
                    goodfit_mask = np.zeros(fit_status.shape, dtype='bool')
                    for flag in goodfit_flags:
                        # Checks for each flag in goodfit_flags
                        goodfit_mask |= fit_status == flag

                    # nearsol cotains all good solutions within a given
                    # radius from the current spaxel.
                    nearsol = sol[:-1, radius_mask & goodfit_mask]
                    
                    if np.shape(nearsol) == (5, 1):
                        model.parameters = deepcopy(nearsol.transpose())
                    elif np.any(nearsol):
                        model.parameters = deepcopy(
                                np.average(nearsol.transpose(), 0))
                        # NOT IMPLEMENTED YET
                        # if update_bounds:
                        #     bounds = bound_updater(p0, bound_range)

                r = fitter(model, wl, s, **minopts)
                
                if fitter.fit_info['ierr'] not in goodfit_flags:
                    print(
                        h, fitter.fit_info['message'], fitter.fit_info['ierr'])
                # Reduced chi squared of the fit.
                chi2 = np.sum(np.square(r(wl) - s))
                n_constraints = len(model.eqcons) + len(model.ineqcons)
                nu = len(s)/inst_disp - npars - n_constraints - 1
                red_chi2 = chi2 / nu
                p = np.append(r.parameters, red_chi2)
                fit_status[i, j] = fitter.fit_info['ierr']
            
            except RuntimeError:
                print(
                    'Optimal parameters not found for spectrum {:d},{:d}'
                    .format(int(i), int(j)))
                p = nan_solution
            if self.binned:
                for l, m in vor[vor[:, 2] == binNum, :2]:
                    sol[:, l, m] = p
                    self.fitcont[:, l, m] = cont
                    self.fitspec[:, l, m] = (s + cont)
                    self.resultspec[:, l, m] = (cont + r(wl))
            else:
                sol[:, i, j] = p
                self.fitcont[:, i, j] = cont
                self.fitspec[:, i, j] = s + cont
                self.resultspec[:, i, j] = cont + r(wl)

        self.em_model = sol
        self.fit_status = fit_status

        if writefits:

            # Basic tests and first header
            if outimage is None:
                outimage = self.fitsfile.replace('.fits',
                                                 '_linefit.fits')
            hdr = deepcopy(self.header_data)
            try:
                hdr['REDSHIFT'] = self.redshift
            except KeyError:
                hdr['REDSHIFT'] = (self.redshift,
                                   'Redshift used in GMOSDC')

            # Creates MEF output.
            h = pf.HDUList()
            h.append(pf.PrimaryHDU(header=hdr))

            # Creates the fitted spectrum extension
            hdr = pf.Header()
            hdr['object'] = ('spectrum', 'Data in this extension')
            hdr['CRPIX3'] = (1, 'Reference pixel for wavelength')
            hdr['CRVAL3'] = (wl[0], 'Reference value for wavelength')
            hdr['CD3_3'] = (np.average(np.diff(wl)), 'CD3_3')
            h.append(pf.ImageHDU(data=self.fitspec, header=hdr))

            # Creates the fitted continuum extension.
            hdr['object'] = 'continuum'
            h.append(pf.ImageHDU(data=self.fitcont, header=hdr))

            # Creates the fitted function extension.
            hdr['object'] = 'fit'
            h.append(pf.ImageHDU(data=self.resultspec, header=hdr))

            # Creates the solution extension.
            hdr['object'] = 'parameters'
            hdr['function'] = (function, 'Fitted function')
            hdr['nfunc'] = (len(p)/3, 'Number of functions')
            h.append(pf.ImageHDU(data=sol, header=hdr))

            # Creates the minimize's exit status extension
            hdr['object'] = 'status'
            h.append(pf.ImageHDU(data=fit_status, header=hdr))

            h.writeto(outimage)

        if individual_spec:
            return wl, s, cont, model(wl), r
        else:
            return sol

    def loadfit(self, fname):
        """
        Loads the result of a previous fit, and put it in the
        appropriate variables for the plotfit function.

        Parameters
        ----------
        fname : string
            Name of the FITS file generated by gmosdc.linefit.

        Returns
        -------
        Nothing.
        """

        self.fitwl = st.get_wl(fname, pix0key='crpix3', wl0key='crval3',
                               dwlkey='cd3_3', hdrext=1, dataext=1)
        self.fitspec = pf.getdata(fname, ext=1)
        self.fitcont = pf.getdata(fname, ext=2)
        self.resultspec = pf.getdata(fname, ext=3)

        self.em_model = pf.getdata(fname, ext=4)
        self.fit_status = pf.getdata(fname, ext=5)

        fit_info = {}
        func_name = pf.getheader(fname, ext=4)['function']
        fit_info['function'] = func_name

        if func_name == 'gaussian':
            self.fit_func = lprof.gauss
            npars = 3

        if func_name == 'gauss_hermite':
            self.fit_func = lprof.gausshermite
            npars = 5

        fit_info['parameters'] = npars
        fit_info['components'] = (self.em_model.shape[0] - 1) / npars

        self.fit_info = fit_info

    def eqw(self, component=0, sigma_factor=3):
        """
        Evaluates the equivalent width of a previous linefit.

        Parameters
        ----------
        component : number
            Component of emission model
        sigma_factor : number
            Radius of integration as a number of line sigmas.
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

                cont = self.fitcont[cond, i, j]
                cont_data = interp1d(
                    fwl, self.fitcont[:, i, j])(rwl[cond_data])

                eqw_model[i, j] = trapz(
                    1. - (fit + cont) / cont, x=fwl[cond])

                eqw_direct[i, j] = trapz(
                    1. - self.data[cond_data, i, j] / cont_data,
                    x=rwl[cond_data])

        return np.array([eqw_model, eqw_direct])

    def w80(self, component, sigma_factor=5, individual_spec=False,
            verbose=False):

        if individual_spec:
            # The reflaction of the *individual_spec* iterable puts the
            # horizontal coordinate first, and the vertical coordinate
            # second.
            xy = [individual_spec[::-1]]
        else:
            xy = self.spec_indices

        w80_model = np.zeros(np.shape(self.em_model)[1:], dtype='float32')
        w80_direct = np.zeros(np.shape(self.em_model)[1:], dtype='float32')

        if self.fit_func == lprof.gauss:
            npars = 3
        if self.fit_func == lprof.gausshermite:
            npars = 5

        par_indexes = np.arange(npars) + npars * component

        center_index = 1 + npars * component
        sigma_index = 2 + npars * component

        for i, j in xy:

            if verbose:
                print(i, j)

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

                w80_model[i, j] = np.nan
                w80_direct[i, j] = np.nan

            else:

                cond = (fwl > cwl - sf * sig) & (fwl < cwl + sf * sig)
                cond_data = (rwl > cwl - sf * sig) & (rwl < cwl + sf * sig)

                fit = self.fit_func(
                    fwl[cond], self.em_model[par_indexes, i, j])

                # cont = self.fitcont[cond, i, j]
                cont_data = interp1d(
                    fwl, self.fitcont[:, i, j])(rwl[cond_data])

                w80_model[i, j], m0, m1, mv, ms = w80eval(fwl[cond], fit, cwl)

                w80_direct[i, j], d0, d1, dv, ds = w80eval(
                    rwl[cond_data], self.data[cond_data, i, j] - cont_data, cwl
                )

                # Plots the fit when evaluating only one spectrum.
                if len(xy) == 1:

                    print('W80 model: {:.2f} km/s'.format(w80_model[i, j]))
                    print('W80 direct: {:.2f} km/s'.format(w80_direct[i, j]))

                    p = [
                            [m0, m1, mv, ms],
                            [d0, d1, dv, ds],
                    ]

                    ifsplots.w80(p)

        return np.array([w80_model, w80_direct])

    def plotfit(self, x, y, show=True, axis=None, output='stdout'):
        """
        Plots the spectrum and features just fitted.

        Parameters
        ----------
        x : number
            Horizontal coordinate of the desired spaxel.
        y : number
            Vertical coordinate of the desired spaxel.

        Returns
        -------
        Nothing.
        """

        if axis is None:
            fig = plt.figure(1)
            plt.clf()
            ax = fig.add_subplot(111)
        else:
            ax = axis

        p = self.em_model[:-1, y, x]
        c = self.fitcont[:, y, x]
        wl = self.fitwl
        f = self.fit_func
        s = self.fitspec[:, y, x]

        norm_factor = np.int(np.log10(np.median(s)))

        ax.plot(wl, (c + f(wl, p)) / 10. ** norm_factor)
        ax.plot(wl, c / 10. ** norm_factor)
        ax.plot(wl, s / 10. ** norm_factor)

        ax.set_xlabel(r'Wavelength (${\rm \AA}$)')
        ax.set_ylabel(
            'Flux density ($10^{{{:d}}}\, {{\\rm erg\,s^{{-1}}\,cm^{{-2}}'
            '\,\AA^{{-1}}}}$)'.format(norm_factor))

        if self.fit_func == lprof.gauss:
            npars = 3
            parnames = ('A', 'wl', 's')
        elif self.fit_func == lprof.gausshermite:
            npars = 5
            parnames = ('A', 'wl', 's', 'h3', 'h4')
        else:
            raise NameError('Unkown fit function.')

        if len(p) > npars:
            for i in np.arange(0, len(p), npars):
                ax.plot(
                    wl, (c + f(wl, p[i: i+npars])) / 10. ** norm_factor, 'k--')

        pars = (npars * '{:10s}' + '\n').format(*parnames)
        for i in np.arange(0, len(p), npars):
            pars += (
                ('{:10.2e}' + (npars-1) * '{:10.2f}' + '\n')
                .format(*p[i:i+npars]))

        if show:
            plt.show()

        if output == 'stdout':
            print(pars)
        if output == 'return':
            return pars

        return ax
    
    def plotfit_model(self, x, y, model, show=True, axis=None,
                      output='stdout'):
        """
        Plots the spectrum and features just fitted.

        Parameters
        ----------
        x : number
            Horizontal coordinate of the desired spaxel.
        y : number
            Vertical coordinate of the desired spaxel.

        Returns
        -------
        Nothing.
        """

        if axis is None:
            fig = plt.figure(1)
            plt.clf()
            ax = fig.add_subplot(111)
        else:
            ax = axis

        p = self.em_model[:-1, y, x]
        c = self.fitcont[:, y, x]
        wl = self.fitwl
        f = model 
        s = self.fitspec[:, y, x]

        norm_factor = np.int(np.log10(np.median(s)))

        ax.plot(wl, (c + f(wl)) / 10. ** norm_factor)
        ax.plot(wl, c / 10. ** norm_factor)
        ax.plot(wl, s / 10. ** norm_factor)

        ax.set_xlabel(r'Wavelength (${\rm \AA}$)')
        ax.set_ylabel(
            'Flux density ($10^{{{:d}}}\, {{\\rm erg\,s^{{-1}}\,cm^{{-2}}'
            '\,\AA^{{-1}}}}$)'.format(norm_factor))

        if model.n_submodels() > 1:
            for submodel in model:
                ax.plot(
                    wl, (c + submodel(wl)) / 10. ** norm_factor, 'k--')
        if show:
            plt.show()

        if output == 'stdout':
            for i, j in enumerate(model.param_names):
                print(j, model.parameters[i])

        if output == 'return':
            return pars

        return ax

    def channelmaps(
            self, lambda0, velmin, velmax, channels=6,
            continuum_width=300, logFlux=False, continuum_opts=None,
            lowerThreshold=1e-16, plot_opts={}, fig_opts={},
            wspace=None, hspace=None, text_color='black'):
        """
        Creates velocity channel maps from a data cube.

        Parameters
        ----------
            lambda0 : number
                Central wavelength of the desired spectral feature.
            vmin : number
                Mininum velocity in kilometers per second.
            vmax : number
                Maximum velocity in kilometers per second.
            channels : integer
                Number of channel maps to build.
            continuum_width : number
                Width in wavelength units for the continuum evaluation
                window.
            continuum_opts : dictionary
                Dicitionary of options to be passed to the
                spectools.continuum function.
            lowerThreshold: number
                Minimum emission flux for plotting, after subtraction
                of the continuum level. Spaxels with flux values below
                lowerThreshold will be masked in the channel maps.
            plot_opts : dict
                Dictionary of options to be passed to **pcolormesh**.
            fig_opts : dict
                Options passed to **pyplot.figure**.
            wspace : number
                Horizontal gap between channel maps.
            hspace : number
                Vertical gap between channel maps.
            text_color : matplotlib color
                The color of the annotated text specifying the velocity
                bin.

        Returns
        -------
        """

        sigma = lowerThreshold

        # Converting from velocities to wavelength
        wlmin, wlmax = lambda0 * (
            np.array([velmin, velmax]) /
            constants.c.to(units.km / units.s).value + 1.
        )

        wlstep = (wlmax - wlmin)/channels
        wl_limits = np.arange(wlmin, wlmax + wlstep, wlstep)

        side = int(np.ceil(np.sqrt(channels)))  # columns
        otherside = int(np.ceil(channels / side))  # lines
        fig = plt.figure(**fig_opts)
        plt.clf()

        if continuum_opts is None:
            continuum_opts = dict(
                niterate=3, degr=5, upper_threshold=3, lower_threshold=3,
                returns='function')

        cp = continuum_opts
        cw = continuum_width
        fw = lambda0 + np.array([-cw / 2., cw / 2.])

        self.cont = self.continuum(
            writefits=False, outimage=None, fitting_window=fw, copts=cp)

        contwl = self.wl[(self.wl > fw[0]) & (self.wl < fw[1])]
        # cont_wl2pix = interp1d(contwl, np.arange(len(contwl)))
        channelMaps = []
        axes = []
        pmaps = []

        for i in np.arange(channels):
            ax = fig.add_subplot(otherside, side, i+1)
            axes += [ax]
            wl = self.restwl
            wl0, wl1 = wl_limits[i], wl_limits[i+1]
            print(wl[(wl > wl0) & (wl < wl1)])
            wlc, wlwidth = np.average([wl0, wl1]), (wl1-wl0)

            f_obs = wlprojection(
                arr=self.data, wl=self.restwl, wl0=wlc, fwhm=wlwidth,
                filtertype='box')
            f_cont = wlprojection(
                arr=self.cont, wl=contwl, wl0=wlc, fwhm=wlwidth,
                filtertype='box')
            f = f_obs - f_cont

            mask = (f < sigma) | np.isnan(f)
            channel = ma.array(f, mask=mask)

            if logFlux:
                channel = np.log10(channel)

            y, x = np.indices(np.array(f.shape) + 1) - 0.5
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max())

            pmap = ax.pcolormesh(x, y, channel, **plot_opts)
            ax.set_aspect('equal', 'datalim')
            ax.annotate(
                '{:.0f}'.format((wlc - lambda0)/lambda0*2.99792e+5),
                xy=(0.1, 0.8), xycoords='axes fraction',
                color=text_color)
            if i % side != 0:
                ax.set_yticklabels([])
            if i / float((otherside-1) * side) < 1:
                ax.set_xticklabels([])
            channelMaps += [channel]
            pmaps += [pmap]

        fig.subplots_adjust(wspace=wspace, hspace=hspace)

        plt.show()

        return channelMaps, axes, pmaps

    def gaussian_smooth(self, sigma=2, writefits=False, outfile=None,
                        clobber=False):
        """
        Performs a spatial gaussian convolution on the data cube.

        Parameters
        ----------
        sigma : number
          Sigma of the gaussian kernel.
        """

        if writefits and outfile is None:
            raise RuntimeError('Output file name not given.')

        gdata = np.zeros_like(self.data)
        gvar = np.zeros_like(self.noise_cube)

        i = 0

        while i < len(self.wl):

            tmp_data = nan_to_nearest(self.data[i])
            tmp_var = nan_to_nearest(self.noise_cube[i]) ** 2

            gdata[i] = gaussian_filter(tmp_data, sigma)
            gvar[i] = np.sqrt(gaussian_filter(tmp_var, sigma))

            i += 1

        if writefits:

            hdulist = pf.open(self.fitsfile)
            hdr = hdulist[0].header

            hdr['SPSMOOTH'] = ('Gaussian', 'Type of spatial smoothing.')
            hdr['GSMTHSIG'] = (sigma, 'Sigma of the gaussian kernel')

            hdulist[self.dataext].data = gdata
            hdulist[self.var_ext].data = gvar

            hdulist.writeto(outfile)

        return gdata, gvar

    def voronoi_binning(self, targetsnr=10.0, writefits=False,
                        outfile=None, clobber=False, writevortab=True,
                        dataext=1):
        """
        Applies Voronoi binning to the data cube, using Cappellari's
        Python implementation.

        Parameters:
        -----------
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

        Returns:
        --------
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

        s, n = deepcopy(self.signal), deepcopy(self.noise)

        s[s <= 0] = np.average(self.signal[self.signal > 0])
        n[n <= 0] = np.average(self.signal[self.signal > 0])*.5

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
            hdulist = pf.open(self.fitsfile)
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
            tbhdu = pf.BinTableHDU.from_columns(
                [
                    pf.Column(name='xcoords', format='i8', array=x),
                    pf.Column(name='ycoords', format='i8', array=y),
                    pf.Column(name='binNum', format='i8', array=binNum),
                ], name='VOR')

            tbhdu_plus = pf.BinTableHDU.from_columns(
                [
                    pf.Column(name='ubin', format='i8',
                              array=np.unique(binNum)),
                    pf.Column(name='xNode', format='F16.8', array=xNode),
                    pf.Column(name='yNode', format='F16.8', array=yNode),
                    pf.Column(name='xBar', format='F16.8', array=xBar),
                    pf.Column(name='yBar', format='F16.8', array=yBar),
                    pf.Column(name='sn', format='F16.8', array=sn),
                    pf.Column(name='nPixels', format='i8', array=nPixels),
                ], name='VORPLUS')

            hdulist.append(tbhdu)
            hdulist.append(tbhdu_plus)

            if outfile is None:
                outfile = '{:s}bin.fits'.format(self.fitsfile[:-4])

            hdulist.writeto(outfile, clobber=clobber)

        self.binned_cube = b_data

    def write_binnedspec(self, dopcor=False, writefits=False):
        """
        Writes only one spectrum for each bin in a FITS file.
        """

        xy = self.spec_indices
        unique_indices = xy[
            np.unique(self.data[1400, :, :], return_index=True)[1]]

        if dopcor:

            try:
                np.shape(self.em_model)
            except AttributeError:
                print(
                    'ERROR! This function requires the gmosdc.em_model' +
                    ' attribute to be defined.')
                return

            for k, i, j in enumerate(unique_indices):
                z = self.em_model[0, i, j]/2.998e+5
                interp_spec = interp1d(self.restwl/(1.+z), self.data[i, j])
                if k == 0:
                    specs = interp_spec(self.restwl)
                else:
                    specs = np.row_stack([specs, interp_spec(self.restwl)])

        else:
            specs = np.row_stack(
                [self.data[:, i, j] for i, j in unique_indices])

        return specs

    def ppxf_kinematics(self, fitting_window, base_wl, base_spec,
                        base_cdelt, writefits=True, outimage=None,
                        vel=0, sigma=180, fwhm_gal=2, fwhm_model=1.8,
                        noise=0.05, individual_spec=False, plotfit=False,
                        quiet=False, deg=4, mask=None, cushion=100.,
                        moments=4):
        """
        Executes pPXF fitting of the stellar spectrum over the whole
        data cube.

        Parameters
        ----------
        fitting_window : array-like
            Initial and final values of wavelength for fitting.
        base_wl : array
            Wavelength coordinates of the base spectra.
        base_spec : array
            Flux density coordinates of the base spectra.
        base_cdelt : number
            Step in wavelength coordinates.

        Returns
        -------
        Nothing

        Description
        -----------
        This function is merely a wrapper for Michelle Capellari's pPXF
        Python algorithm for penalized pixel fitting of stellar
        spectra.
        """

        try:
            import ppxf
        except ImportError:
            raise ImportError(
                'Could not find the ppxf module. '
                'Please add it to your PYTHONPATH.')

        try:
            import ppxf_util
        except ImportError:
            raise ImportError(
                'Could not find the ppxf_util module. '
                'Please add it to your PYTHONPATH.')

        w0, w1 = fitting_window
        fw = (self.restwl >= w0) & (self.restwl < w1)

        baseCut = (base_wl > w0 - cushion) & (base_wl < w1 + cushion)

        if not np.any(baseCut):
            raise RuntimeError(
                'The interval defined by fitting_window lies outside '
                'the range covered by base_wl. Please review your base '
                'and/or fitting window.')

        base_spec = base_spec[:, baseCut]
        base_wl = base_wl[baseCut]

        lamRange1 = self.restwl[fw][[0, -1]]
        centerSpaxel = np.array(
            [int(i / 2) for i in np.shape(self.data[0])], dtype='int')
        gal_lin = deepcopy(self.data[fw, centerSpaxel[0], centerSpaxel[1]])

        galaxy, logLam1, velscale = ppxf_util.log_rebin(
            lamRange1, gal_lin)

        # Here we use the goodpixels as the fitting window
        # gp = np.arange(np.shape(self.data)[0])[fw]
        gp = np.arange(len(logLam1))
        lam1 = np.exp(logLam1)

        if mask is not None:
            if len(mask) == 1:
                gp = gp[
                    (lam1 < mask[0][0]) | (lam1 > mask[0][1])]
            else:
                m = np.array([
                    (lam1 < i[0]) | (lam1 > i[1]) for i in mask])
                gp = gp[np.sum(m, 0) == m.shape[0]]

        lamRange2 = base_wl[[0, -1]]
        ssp = base_spec[0]

        sspNew, logLam2, velscale = ppxf_util.log_rebin(
            lamRange2, ssp, velscale=velscale)
        templates = np.empty((sspNew.size, len(base_spec)))

        # Convolve the whole Vazdekis library of spectral templates
        # with the quadratic difference between the SAURON and the
        # Vazdekis instrumental resolution. Logarithmically rebin
        # and store each template as a column in the array TEMPLATES.

        # Quadratic sigma difference in pixels Vazdekis --> SAURON
        # The formula below is rigorously valid if the shapes of the
        # instrumental spectral profiles are well approximated by
        # Gaussians.

        FWHM_dif = np.sqrt(fwhm_gal**2 - fwhm_model**2)
        # Sigma difference in pixels
        sigma = FWHM_dif/2.355/base_cdelt

        for j in range(len(base_spec)):
            ssp = base_spec[j]
            ssp = ndimage.gaussian_filter1d(ssp, sigma)
            sspNew, logLam2, velscale = ppxf_util.log_rebin(
                lamRange2, ssp, velscale=velscale)
            # Normalizes templates
            templates[:, j] = sspNew/np.median(sspNew)

        c = constants.c.value * 1.e-3
        dv = (logLam2[0] - logLam1[0]) * c  # km/s
        # z = np.exp(vel/c) - 1

        # Here the actual fit starts.
        start = [vel, 180.]  # (km/s), starting guess for [V,sigma]

        # Assumes uniform noise accross the spectrum
        noise = np.zeros(len(galaxy), dtype=galaxy.dtype) + noise

        if self.binned:
            vor = self.voronoi_tab
            xy = np.column_stack([
                vor[coords][np.unique(vor['binNum'], return_index=True)[1]]
                for coords in ['ycoords', 'xcoords']])
        else:
            xy = self.spec_indices

        if individual_spec:
            xy = [individual_spec[::-1]]

        ppxf_sol = np.zeros(
            (4, np.shape(self.data)[1], np.shape(self.data)[2]),
            dtype='float64')
        ppxf_spec = np.zeros(
            (len(galaxy), np.shape(self.data)[1], np.shape(self.data)[2]),
            dtype='float64')
        ppxf_model = np.zeros(np.shape(ppxf_spec), dtype='float64')

        nspec = len(xy)

        for k, h in enumerate(xy):
            progress(k, nspec, 10)
            i, j = h

            gal_lin = deepcopy(self.data[fw, i, j])
            galaxy, logLam1, velscale = ppxf_util.log_rebin(
                lamRange1, gal_lin)
            normFactor = np.nanmean(galaxy)
            galaxy = galaxy / normFactor

            if np.any(np.isnan(galaxy)):
                pp = nanSolution()
                pp.ppxf(ppxf_sol[:, 0, 0], galaxy, galaxy)
            else:
                pp = ppxf.ppxf(
                    templates, galaxy, noise, velscale, start, goodpixels=gp,
                    plot=plotfit, moments=moments, degree=deg, vsyst=dv,
                    quiet=quiet)
                if plotfit:
                    plt.show()

            if self.binned:

                binNum = vor[
                    (vor['xcoords'] == j) & (vor['ycoords'] == i)]['binNum']
                sameBinNum = vor['binNum'] == binNum
                sameBinX = vor['xcoords'][sameBinNum]
                sameBinY = vor['ycoords'][sameBinNum]

                for l, m in np.column_stack([sameBinY, sameBinX]):
                    ppxf_sol[:, l, m] = pp.sol
                    ppxf_spec[:, l, m] = pp.galaxy
                    ppxf_model[:, l, m] = pp.bestfit

            else:
                ppxf_sol[:, i, j] = pp.sol
                ppxf_spec[:, i, j] = pp.galaxy
                ppxf_model[:, i, j] = pp.bestfit

        self.ppxf_sol = ppxf_sol
        self.ppxf_spec = ppxf_spec * normFactor
        self.ppxf_model = ppxf_model * normFactor
        self.ppxf_wl = np.e ** logLam1
        self.ppxf_goodpixels = gp

        if writefits:

            # Basic tests and first header
            if outimage is None:
                outimage = self.fitsfile.replace(
                    '.fits', '_ppxf.fits')
            hdr = deepcopy(self.header_data)
            try:
                hdr['REDSHIFT'] = self.redshift
            except KeyError:
                hdr['REDSHIFT'] = (self.redshift, 'Redshift used in GMOSDC')

            # Creates MEF output.
            h = pf.HDUList()
            h.append(pf.PrimaryHDU(header=hdr))
            h[0].name = ''
            print(h.info())

            # Creates the fitted spectrum extension
            hdr = pf.Header()
            hdr['object'] = ('spectrum', 'Data in this extension')
            hdr['CRPIX3'] = (1, 'Reference pixel for wavelength')
            hdr['CRVAL3'] = (self.restwl[0], 'Reference value for wavelength')
            hdr['CD3_3'] = (np.average(np.diff(self.restwl)), 'CD3_3')
            h.append(
                pf.ImageHDU(data=self.ppxf_spec, header=hdr, name='SCI'))

            # Creates the residual spectrum extension
            hdr = pf.Header()
            hdr['object'] = ('residuals', 'Data in this extension')
            hdr['CRPIX3'] = (1, 'Reference pixel for wavelength')
            hdr['CRVAL3'] = (self.ppxf_wl[0], 'Reference value for wavelength')
            hdr['CD3_3'] = (np.average(np.diff(self.ppxf_wl)), 'CD3_3')
            h.append(
                pf.ImageHDU(
                    data=self.ppxf_spec - self.ppxf_model, header=hdr,
                    name='RES'))

            # Creates the fitted model extension.
            hdr['object'] = 'model'
            h.append(
                pf.ImageHDU(
                    data=self.ppxf_model, header=hdr, name='MODEL'))

            # Creates the solution extension.
            hdr['object'] = 'parameters'
            h.append(
                pf.ImageHDU(
                    data=self.ppxf_sol, header=hdr, name='SOL'))

            # Creates the wavelength extension.
            hdr['object'] = 'wavelength'
            h.append(
                pf.ImageHDU(
                    data=self.ppxf_wl, header=hdr, name='WAVELEN'))

            # Creates the goodpixels extension.
            hdr['object'] = 'goodpixels'
            h.append(
                pf.ImageHDU(
                    data=self.ppxf_goodpixels, header=hdr, name='GOODPIX'))

            h.writeto(outimage)

    def ppxf_plot(self, xy, axis=None):

        if axis is None:
            fig = plt.figure(1)
            plt.clf()
            ax = fig.add_subplot(111)
        else:
            ax = axis
            ax.cla()

        gp = self.ppxf_goodpixels

        ax.plot(self.ppxf_wl[gp], self.ppxf_spec[gp, xy[1], xy[0]])
        ax.plot(self.ppxf_wl, self.ppxf_spec[:, xy[1], xy[0]])
        ax.plot(self.ppxf_wl, self.ppxf_model[:, xy[1], xy[0]])

        print(
            ('Velocity: {:.2f}\nSigma: {:.2f}\nh3: {:.2f}\nh4: {:.2f}').
            format(*self.ppxf_sol[:, xy[1], xy[0]]))

    def lineflux(self, amplitude, sigma):
        """
        Calculates the flux in a line given the amplitude and sigma
        of the gaussian function that fits it.
        """

        lf = amplitude * abs(sigma) * np.sqrt(2. * np.pi)

        return lf
