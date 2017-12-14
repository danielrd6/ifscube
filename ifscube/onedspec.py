from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from astropy import wcs
import astropy.io.fits as pf

from . import stats, spectools
from . import elprofile as lprof


def scale_bounds(bounds, scale_factor, npars_pc):

    b = np.array(deepcopy(bounds))
    for j in b[::npars_pc]:
        for k in (0, 1):
            if j[k] is not None:
                j[k] /= scale_factor

    return b


class Spectrum():

    def __init__(self, *args, **kwargs):

        if len(args) > 0:
            self.__load__(*args, **kwargs)

    def __accessory_data__(self, variance, flags, stellar):

        if variance is not None:
            assert variance.shape == self.data.shape, 'Variance spectrum must'\
                ' have the same shape of the spectrum itself.'
            self.variance = variance
        else:
            self.variance = np.ones_like(self.data)

        if flags is not None:
            assert flags.shape == self.data.shape, 'Flag spectrum must have '\
                'the same shape of the spectrum itself.'
            self.flags = flags
        else:
            self.flags = np.zeros_like(self.data)

        if stellar is not None:
            assert flags.shape == self.data.shape, 'Stellar population'\
                ' spectrum must have the same shape of the spectrum itself.'
            self.stellar = stellar
        else:
            self.stellar = np.zeros_like(self.data)

    def __load__(self, fname, ext=0, redshift=0, variance=None,
                 flags=None, stellar=None):

        with pf.open(fname) as hdu:
            self.data = hdu[ext].data
            self.header = hdu[ext].header
            self.wcs = wcs.WCS(self.header)

        self.__accessory_data__(variance, flags, stellar)

        self.wl = self.wcs.wcs_pix2world(np.arange(len(self.data)), 0)[0]
        self.delta_lambda = self.wcs.pixel_scale_matrix[0, 0]

        if redshift != 0:
            self.restwl = self.__dopcor__()
        else:
            self.restwl = self.wl

    def __dopcor__(self):

        self.restwl = self.wl / (1. + self.redshift)

    def guessParser(self, p):

        npars = len(self.parnames)
        for i in p[::npars]:
            if i == 'peak':
                p[p.index(i)] = self.data[self.valid_pixels].max()
            elif i == 'mean':
                p[p.index(i)] = self.data[self.valid_pixels].mean()

        return p

    def linefit(self, p0, function='gaussian', fitting_window=None,
                writefits=False, outimage=None, variance=None,
                constraints=(), bounds=None, inst_disp=1.0,
                min_method='SLSQP', minopts={'eps': 1e-3}, copts=None,
                weights=None, verbose=False, fit_continuum=False):
        """
        Fits a spectral features.

        Parameters
        ----------
        p0 : iterable
            Initial guess for the fitting funcion, consisting of a list
            of N*M parameters for M components of **function**. In the
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
            self.parnames = ('A', 'wl', 's')
        elif function == 'gauss_hermite':
            fit_func = lprof.gausshermite
            self.fit_func = lprof.gausshermite
            npars_pc = 5
            self.parnames = ('A', 'wl', 's', 'h3', 'h4')
        else:
            raise NameError('Unknown function "{:s}".'.format(function))

        # Sets a pre-made nan vector for nan solutions.
        npars = len(p0)
        self.em_model = np.array([np.nan for i in range(npars + 1)])

        if fitting_window is None:
            fw = np.ones_like(self.data).astype('bool')
        else:
            fw = (self.restwl > fitting_window[0]) &\
                 (self.restwl < fitting_window[1])
        if not np.any(fw):
            raise RuntimeError(
                'Fitting window outside the available wavelength range.')
        zero_spec = np.zeros_like(self.restwl[fw])

        valid_pixels = (self.flags == 0) & fw
        self.valid_pixels = valid_pixels

        wl = deepcopy(self.restwl[valid_pixels])
        data = deepcopy(self.data[valid_pixels])
        stellar = deepcopy(self.stellar[valid_pixels])

        self.fitspec = self.data[fw]
        self.resultspec = zero_spec
        self.fitcont = zero_spec
        self.fitwl = self.restwl[fw]
        self.fitstellar = self.stellar[fw]
        self.r = None

        p0 = np.array(self.guessParser(p0))
        self.initial_guess = p0
        self.fitbounds = bounds

        #
        # Avoids fit if more than 80% of the pixels are flagged.
        #
        if np.sum(~valid_pixels[fw]) > 0.8 * valid_pixels[fw].size:
            self.fit_status = 98
            return

        if weights is None:
            weights = np.ones_like(data)

        sol = np.zeros((npars + 1,))

        #
        # Pseudo continuum fitting.
        #

        if copts is None:
            copts = dict(
                niterate=5, degr=4, upper_threshold=2, lower_threshold=2)

        copts.update(dict(returns='polynomial'))

        try:
            cont = self.continuum[valid_pixels]
            self.fitcont = self.continuum[fw]
        except AttributeError:
            if fit_continuum:
                pcont = spectools.continuum(wl, data - stellar, **copts)
                self.fitcont = np.polyval(pcont, self.restwl[fw])
                cont = np.polyval(pcont, wl)
            else:
                cont = np.zeros_like(data)

        #
        # Short alias for the spectrum that is going to be fitted.
        #
        s = data - cont - stellar
        w = weights
        v = self.variance[valid_pixels]

        #
        # Checks for the presence of negative or zero variance pixels.
        #
        if not np.all(v > 0):
            self.fit_status = 96
            return
        #
        # Checks for the presence of negative weight values.
        #
        if np.any(w < 0):
            self.fit_status = 95
            return
        #
        # Scaling the flux in the spectrum, the initial guess and the
        # bounds to bring everything close to unity.
        #
        scale_factor = np.mean(s)
        if scale_factor <= 0:
            self.fit_status = 97
            return
        s /= scale_factor
        if not np.all(v == 1.):
            v /= scale_factor ** 2
        p0[::npars_pc] /= scale_factor
        sbounds = scale_bounds(bounds, scale_factor, npars_pc)

        #
        # Here the actual fit begins
        #
        def res(x):
            m = fit_func(wl, x)
            # Should I divide this by the sum of the weights?
            a = w * (s - m) ** 2
            b = a / v
            rms = np.sqrt(np.sum(b))
            return rms

        r = minimize(res, x0=p0, method=min_method, bounds=sbounds,
                     constraints=constraints, options=minopts)
        self.r = r

        if verbose and r.status != 0:
            print(r.message, r.status)

        # Reduced chi squared of the fit.
        chi2 = np.sum((s - fit_func(wl, r.x)) ** 2 / v)
        nu = len(s) / inst_disp - npars - len(constraints) - 1
        red_chi2 = chi2 / nu

        self.fit_status = r.status

        p0[::npars_pc] *= scale_factor
        p = np.append(r['x'], red_chi2)
        p[0:-1:npars_pc] *= scale_factor

        self.resultspec = self.fitstellar + self.fitcont\
            + fit_func(self.fitwl, r['x']) * scale_factor

        self.em_model = p

        if writefits:

            # Basic tests and first header
            if outimage is None:
                outimage = self.fitsfile.replace('.fits',
                                                 '_linefit.fits')
            hdr = deepcopy(self.header)

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
            hdr['nfunc'] = (len(p) / npars_pc, 'Number of functions')
            h.append(pf.ImageHDU(data=sol, header=hdr))

            # Creates the minimize's exit status extension
            hdr['object'] = 'status'
            h.append(pf.ImageHDU(data=self.fit_status, header=hdr))

            h.writeto(outimage)

        return sol

    def fit_uncertainties(self, snr=10):

        if self.fit_func == lprof.gauss:

            peak = self.em_model[0]
            flux = self.em_model[0] * self.em_model[2] * np.sqrt(2. * np.pi)

        elif self.fit_func == lprof.gausshermite:

            # This is a crude estimate for the peak, and only works
            # for small values of h3 and h4.

            peak = self.em_model[0] / self.em_model[2] / np.sqrt(2. * np.pi)
            flux = self.em_model[0]

        try:
            s_peak = interp1d(self.wl, self.err)(self.em_model[1])
        except AttributeError:
            s_peak = peak / snr

        # The third element in em_model is always the sigma
        fwhm = self.em_model[2] * 2. * np.sqrt(2. * np.log(2.))

        sf = stats.line_flux_error(flux, fwhm, self.delta_lambda, peak, s_peak)

        self.flux_err = sf

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

        self.fitwl = spectools.get_wl(
            fname, pix0key='crpix0', wl0key='crval0', dwlkey='cd1_1', hdrext=1,
            dataext=1)
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

    def plotfit(self, show=True, axis=None, output='stdout'):
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

        p = self.em_model[:-1]
        c = self.fitcont
        wl = self.fitwl
        f = self.fit_func
        s = self.fitspec

        ax.plot(wl, c + f(wl, p))
        ax.plot(wl, c)
        ax.plot(wl, s)

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
                ax.plot(wl, c + f(wl, p[i: i + npars]), 'k--')

        pars = (npars * '{:10s}' + '\n').format(*parnames)
        for i in np.arange(0, len(p), npars):
            pars += (
                ('{:10.2e}' + (npars - 1) * '{:10.2f}' + '\n')
                .format(*p[i:i + npars]))

        if show:
            plt.show()

        if output == 'stdout':
            print(pars)
        if output == 'return':
            return pars

        return

    def plotspec(self, overplot=True):

        if overplot:
            try:
                ax = plt.gca()
            except:
                fig = plt.figure()
                ax = fig.add_subplot(111)

        ax.plot(self.wl, self.data)
        plt.show()
