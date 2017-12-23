# STDLIB
from copy import deepcopy

# THIRD PARTY
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from astropy import wcs, table
from astropy.io import fits

# LOCAL
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

        self.fitsfile = fname
        self.redshift = redshift
        with fits.open(fname) as hdu:
            self.data = hdu[ext].data
            self.header = hdu[0].header
            self.header_data = hdu[ext].header
            self.wcs = wcs.WCS(self.header_data)

        self.__accessory_data__(variance, flags, stellar)

        self.wl = self.wcs.wcs_pix2world(np.arange(len(self.data)), 0)[0]
        self.delta_lambda = self.wcs.pixel_scale_matrix[0, 0]

        if redshift != 0:
            self.__dopcor__()
        else:
            self.restwl = self.wl

    def __dopcor__(self):

        self.restwl = self.wl / (1. + self.redshift)

    def __fitTable__(self):

        cnames = self.component_names
        pnames = self.parnames

        c = np.array([[i for j in pnames] for i in cnames]).flatten()
        p = np.array([[i for i in pnames] for j in cnames]).flatten()

        t = table.Table([c, p], names=('component', 'parameter'))
        h = fits.table_to_hdu(t)

        return h

    def __write_linefit__(self, args):

        outimage = args['outimage']
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
        h = fits.HDUList()
        hdu = fits.PrimaryHDU(header=self.header)
        hdu.name = 'PRIMARY'
        h.append(hdu)

        # Creates the fitted spectrum extension
        hdr = fits.Header()
        hdr['object'] = ('spectrum', 'Data in this extension')
        hdr['CRPIX3'] = (1, 'Reference pixel for wavelength')
        hdr['CRVAL3'] = (self.fitwl[0], 'Reference value for wavelength')
        hdr['CD3_3'] = (np.average(np.diff(self.fitwl)), 'CD3_3')
        hdu = fits.ImageHDU(data=self.fitspec, header=hdr)
        hdu.name = 'FITSPEC'
        h.append(hdu)

        # Creates the fitted continuum extension.
        hdr['object'] = 'continuum'
        hdu = fits.ImageHDU(data=self.fitcont, header=hdr)
        hdu.name = 'FITCONT'
        h.append(hdu)

        # Creates the stellar continuum extension.
        hdr['object'] = 'stellar'
        hdu = fits.ImageHDU(data=self.fitstellar, header=hdr)
        hdu.name = 'STELLAR'
        h.append(hdu)

        # Creates the fitted function extension.
        hdr['object'] = 'modeled_spec'
        hdu = fits.ImageHDU(data=self.resultspec, header=hdr)
        hdu.name = 'MODEL'
        h.append(hdu)

        # Creates the solution extension.
        hdr = fits.Header()
        function = args['function']
        total_pars = self.em_model.shape[0] - 1

        hdr['object'] = 'parameters'
        hdr['function'] = (function, 'Fitted function')
        hdr['nfunc'] = (int(total_pars / self.npars), 'Number of functions')
        hdu = fits.ImageHDU(data=self.em_model, header=hdr)
        hdu.name = 'SOLUTION'
        h.append(hdu)

        # Equivalent width extensions
        hdr['object'] = 'eqw_model'
        hdu = fits.ImageHDU(data=self.eqw_model, header=hdr)
        hdu.name = 'EQW_M'
        h.append(hdu)

        hdr['object'] = 'eqw_direct'
        hdu = fits.ImageHDU(data=self.eqw_direct, header=hdr)
        hdu.name = 'EQW_D'
        h.append(hdu)

        # # Creates the minimize's exit status extension
        # hdr['object'] = 'status'
        # hdu = fits.ImageHDU(data=self.fit_status, header=hdr)
        # hdu.name = 'STATUS'
        # h.append(hdu)

        # Creates component and parameter names table.
        hdr['object'] = 'parameter names'
        hdu = self.__fitTable__()
        hdu.name = 'PARNAMES'
        h.append(hdu)

        h.writeto(outimage, overwrite=args['overwrite'])

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
                weights=None, verbose=False, fit_continuum=False,
                component_names=None, overwrite=False, eqw_opts={}):
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
        self.npars = npars_pc

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

        if component_names is None:
            self.component_names = [
                'C_{:03d}'.format(i) for i in range(int(npars / npars_pc))]
        else:
            self.component_names = component_names

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
        self.eqw(**eqw_opts)

        if writefits:
            self.__write_linefit__(args=locals())
        return sol

    def eqw(self, sigma_factor=5, continuum_windows=None):
        """
        Evaluates the equivalent width of a previous linefit.

        Parameters
        ----------
        component : number
            Component of emission model
        sigma_factor : number
            Radius of integration as a number of line sigmas.
        windows : iterable
          Continuum fitting windows in the form
          [blue0, blue1, red0, red1].

        Returns
        -------
        eqw : numpy.ndarray
          Equivalent widths measured on the emission line model and
          directly on the observed spectrum, respectively.
        """

        eqw_model = np.zeros((len(self.component_names),))
        eqw_direct = np.zeros_like(eqw_model)

        npars = self.npars

        for component in self.component_names:

            component_index = self.component_names.index(component)
            par_indexes = np.arange(npars) + npars * component_index

            center_index = 1 + npars * component_index
            sigma_index = 2 + npars * component_index

            # Wavelength vector of the line fit
            fwl = self.fitwl
            # Center wavelength coordinate of the fit
            cwl = self.em_model[center_index]
            # Sigma of the fit
            sig = self.em_model[sigma_index]
            # Just a short alias for the sigma_factor parameter
            sf = sigma_factor

            nandata_flag = np.any(np.isnan(self.em_model[par_indexes]))
            nullcwl_flag = cwl == 0

            if nandata_flag or nullcwl_flag:

                eqw_model = np.nan
                eqw_direct = np.nan

            else:

                low_wl = cwl - sf * sig
                up_wl = cwl + sf * sig

                cond = (fwl > low_wl) & (fwl < up_wl)

                fit = self.fit_func(
                    fwl[cond], self.em_model[par_indexes])
                syn = self.fitstellar
                fitcont = self.fitcont
                data = self.fitspec

                # If the continuum fitting windows are set, use that
                # to define the weights vector.
                cwin = continuum_windows[component]
                if cwin is not None:
                    assert len(cwin) == 4, 'Windows must be an '\
                        'iterable of the form (blue0, blue1, red0, red1)'
                    weights = np.zeros_like(self.fitwl)
                    cwin_cond = (
                        ((fwl > cwin[0]) & (fwl < cwin[1])) |
                        ((fwl > cwin[2]) & (fwl < cwin[3]))
                    )
                    weights[cwin_cond] = 1
                else:
                    weights = np.ones_like(self.fitwl)

                cont = spectools.continuum(
                    fwl, syn + fitcont, weights=weights,
                    degr=1, niterate=3, lower_threshold=3,
                    upper_threshold=3, returns='function')[1][cond]

                # Remember that 1 - (g + c)/c = -g/c, where g is the
                # line profile and c is the local continuum level.
                #
                # That is why we can shorten the equivalent width
                # definition in the eqw_model integration below.
                ci = component_index
                eqw_model[ci] = trapz(- fit / cont, x=fwl[cond])
                eqw_direct[ci] = trapz(1. - data[cond] / cont, x=fwl[cond])

            self.eqw_model = eqw_model
            self.eqw_direct = eqw_direct

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
        self.fitspec = fits.getdata(fname, ext=1)
        self.fitcont = fits.getdata(fname, ext=2)
        self.resultspec = fits.getdata(fname, ext=3)

        self.em_model = fits.getdata(fname, ext=4)
        self.fit_status = fits.getdata(fname, ext=5)

        fit_info = {}
        func_name = fits.getheader(fname, ext=4)['function']
        fit_info['function'] = func_name

        if func_name == 'gaussian':
            self.fit_func = lprof.gauss
            self.npars = 3
            self.parnames = ('A', 'wl', 's')
        elif func_name == 'gauss_hermite':
            self.fit_func = lprof.gausshermite
            self.npars = 5
            self.parnames = ('A', 'wl', 's', 'h3', 'h4')
        else:
            raise RuntimeError('Unkonwn function {:s}'.format(func_name))

        fit_info['parameters'] = self.npars
        fit_info['components'] = (self.em_model.shape[0] - 1) / self.npars

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

        npars = self.npars
        parnames = self.parnames

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
