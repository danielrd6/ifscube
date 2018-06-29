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
            self._load(*args, **kwargs)

    def _accessory_data(self, hdu, variance, flags, stellar):

        def shmess(name):
            s = '{:s} spectrum must have the same shape of the spectrum itself'
            return s.format(name)

        self.variance = np.ones_like(self.data)
        self.flags = np.zeros_like(self.data)
        self.stellar = np.zeros_like(self.data)

        acc_data = [self.variance, self.flags, self.stellar]
        ext_names = [variance, flags, stellar]
        labels = ['Variance', 'Flags', 'Synthetic']

        for i, j, lab in zip(acc_data, ext_names, labels):

            if j is not None:
                if isinstance(j, str):
                    if j in hdu:
                        assert hdu[j].data.shape == self.data.shape,\
                            shmess(lab)
                        i[:] = hdu[j].data
                elif isinstance(j, np.ndarray):
                    i[:] = j

    def _load(self, fname, scidata='SCI', variance=None,
              flags=None, stellar=None, primary='PRIMARY',
              redshift=None):
        self.fitsfile = fname

        with fits.open(fname) as hdu:
            self.data = hdu[scidata].data
            self.header = hdu[primary].header
            self.header_data = hdu[scidata].header
            self.wcs = wcs.WCS(self.header_data)

            self._accessory_data(hdu, variance, flags, stellar)

        self.wl = self.wcs.wcs_pix2world(np.arange(len(self.data)), 0)[0]
        self.delta_lambda = self.wcs.pixel_scale_matrix[0, 0]
        try:
            if self.header_data['cunit1'] == 'm':
                self.wl *= 1.e+10
                self.delta_lambda *= 1.e+10
        except KeyError:
            pass

        # Setting the redshift.
        # Redshift from arguments takes precedence over redshift
        # from the image header.
        if redshift is not None:
            self.redshift = redshift
        elif 'redshift' in self.header:
            self.redshift = self.header['REDSHIFT']
        else:
            self.redshift = 0

        self.restwl = self.dopcor(
            self.redshift, self.wl, self.data)

    @staticmethod
    def dopcor(z, wl, flux):

        restwl = wl / (1. + z)
        return restwl

    def _fitTable(self):

        cnames = self.component_names
        pnames = self.parnames

        c = np.array([[i for j in pnames] for i in cnames]).flatten()
        p = np.array([[i for i in pnames] for j in cnames]).flatten()

        t = table.Table([c, p], names=('component', 'parameter'))
        h = fits.table_to_hdu(t)

        return h

    def _write_linefit(self, args):

        suffix = args['suffix']
        outimage = args['outimage']
        # Basic tests and first header
        if outimage is None:
            if suffix is None:
                suffix = '_linefit'
            outimage = self.fitsfile.replace('.fits', suffix + '.fits')

        hdr = self.header
        try:
            hdr['REDSHIFT'] = self.redshift
        except KeyError:
            hdr['REDSHIFT'] = (self.redshift,
                               'Redshift used in IFSCUBE')

        # Creates MEF output.
        h = fits.HDUList()
        hdu = fits.PrimaryHDU(header=self.header)
        hdu.name = 'PRIMARY'
        h.append(hdu)

        # Creates the fitted spectrum extension
        hdr = fits.Header()
        hdr['object'] = ('spectrum', 'Data in this extension')
        hdr['CRPIX1'] = (1, 'Reference pixel for wavelength')
        hdr['CRVAL1'] = (self.fitwl[0], 'Reference value for wavelength')
        hdr['CD1_1'] = (np.average(np.diff(self.fitwl)), 'CD1_1')
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
        hdr['fitstat'] = self.fit_status
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
        hdu = self._fitTable()
        hdu.name = 'PARNAMES'
        h.append(hdu)

        h.writeto(outimage, overwrite=args['overwrite'])

    def guess_parser(self, p):

        npars = len(self.parnames)
        for i in p[::npars]:
            if i == 'peak':
                p[p.index(i)] = self.data[self.valid_pixels].max()
            elif i == 'mean':
                p[p.index(i)] = self.data[self.valid_pixels].mean()
            elif i == 'median':
                p[p.index(i)] = np.median(self.data[self.valid_pixels])

        return p

    def feature_slice(self, wl, lam, s):

        low_lam = (lam - s)
        up_lam = (lam + s)

        try:
            wl_lims = [
                wl[wl < low_lam][-1],
                wl[wl > up_lam][0]]
            idx = [np.where(wl == i)[0][0] for i in wl_lims]
            ws = slice(idx[0], idx[1])

        except IndexError:
            ws = None

        return ws

    def guess_parameters(self, data, wl, p0, npars_pc):

        new_p0 = deepcopy(p0)

        for i in range(0, p0.size, npars_pc):

            ws = self.feature_slice(wl, p0[i + 1], p0[i + 2] * 3)

            if ws is not None:
                new_p0[i] = np.max(data[ws])
                new_p0[i + 1] = np.average(wl[ws], weights=data[ws])
                new_p0[i + 2] = np.average(
                    abs(wl[ws] - wl[ws].mean()), weights=data[ws])

        return new_p0

    def optimize_mask(self, data, wl, p0, width=20, catch_error=False):

        npars_pc = len(self.parnames)
        npars = len(p0)

        mask = np.zeros_like(wl).astype(bool)

        for i in range(0, npars, npars_pc):
            lam = p0[i + 1]
            s = p0[i + 2]

            low_lam = (lam - width * s)
            up_lam = (lam + width * s)

            if catch_error:

                assert low_lam > wl[0],\
                    'ERROR in optimization mask. Lower limit in optimization '\
                    'window is below the lowest available wavelength.'

                assert up_lam < wl[-1],\
                    'ERROR in optimization mask. Upper limit in optimization '\
                    'window is above the highest available wavelength.'

            else:
                if (low_lam < wl[0]) or (up_lam > wl[-1]):
                    continue

            wl_lims = [
                wl[wl < low_lam][-1],
                wl[wl > up_lam][0]]
            idx = [np.where(wl == i)[0][0] for i in wl_lims]

            ws = slice(idx[0], idx[1])

            mask[ws] = True

        return mask

    def linefit(self, p0, function='gaussian', fitting_window=None,
                writefits=False, outimage=None, variance=None,
                constraints=(), bounds=None, inst_disp=1.0,
                min_method='SLSQP', minopts={'eps': 1e-3}, copts=None,
                weights=None, verbose=False, fit_continuum=False,
                component_names=None, overwrite=False, eqw_opts={},
                trivial=False, suffix=None, optimize_fit=False,
                optimization_window=10, guess_parameters=False):
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
        trivial : boolean
            Attempts a fit with a trivial solution, and if the rms is
            smaller than the fit with the intial guess, selects the
            trivial fit as the correct solution.

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

        assert self.restwl[fw].min() < np.min(p0[1::npars_pc]),\
            'Attempting to fit a spectral feature below the fitting window.'

        assert self.restwl[fw].max() > np.max(p0[1::npars_pc]),\
            'Attempting to fit a spectral feature above the fitting window.'

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
        self.eqw_model = np.nan
        self.eqw_direct = np.nan

        p0 = np.array(self.guess_parser(p0))
        self.initial_guess = p0
        self.fitbounds = bounds

        #
        # Avoids fit if more than 80% of the pixels are flagged.
        #
        if np.sum(~valid_pixels[fw]) > (0.8 * valid_pixels[fw].size):
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
        scale_factor = np.abs(np.mean(s))
        # if scale_factor <= 0:
        #     self.fit_status = 97
        #     return
        s /= scale_factor
        if not np.all(v == 1.):
            v /= scale_factor ** 2
        p0[::npars_pc] /= scale_factor
        sbounds = scale_bounds(bounds, scale_factor, npars_pc)

        #
        # Optimization mask
        #
        if optimize_fit:
            opt_mask = self.optimize_mask(
                s, wl, p0, width=optimization_window)
        else:
            opt_mask = np.ones_like(wl).astype(bool)

        if guess_parameters:
            p0 = self.guess_parameters(s, wl, p0, npars_pc)
        #
        # Here the actual fit begins
        #

        def res(x):
            m = fit_func(wl[opt_mask], x)
            a = w[opt_mask] * (s[opt_mask] - m) ** 2
            b = a / v[opt_mask]
            rms = np.sqrt(np.sum(b))
            return rms

        r = minimize(res, x0=p0, method=min_method, bounds=sbounds,
                     constraints=constraints, options=minopts)

        # Perform the fit a second time with the RMS as the flux
        # initial guess. This was added after a number of fits returned
        # high flux values even when no lines were present.
        fit_rms = res(r.x)
        trivial_p0 = [
            fit_rms if self.parnames[i % npars_pc] == 'A' else p0[i]
            for i in range(len(p0))]
        if res(trivial_p0) < res(r.x):
            r = minimize(res, x0=trivial_p0, method=min_method,
                         bounds=sbounds, constraints=constraints,
                         options=minopts)

        for i in range(0, r.x.size, npars_pc):
            if (r.x[i + 1] > wl.max()) or (r.x[i + 1] < wl.min()):
                r.x[i:i + npars_pc] = np.nan

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
            self._write_linefit(args=locals())
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
            nullcwl_flag = (cwl == 0) or (cwl == np.nan)

            if nandata_flag or nullcwl_flag:

                eqw_model[component_index] = np.nan
                eqw_direct[component_index] = np.nan

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
                if continuum_windows is not None:
                    if component in continuum_windows:
                        cwin = continuum_windows[component]
                    else:
                        cwin = None
                else:
                    cwin = None

                if cwin is not None:
                    assert len(cwin) == 4, 'Windows must be an '\
                        'iterable of the form (blue0, blue1, red0, red1)'
                    weights = np.zeros_like(self.fitwl)
                    cwin_cond = (
                        ((fwl > cwin[0]) & (fwl < cwin[1])) |
                        ((fwl > cwin[2]) & (fwl < cwin[3]))
                    )
                    weights[cwin_cond] = 1
                    nite = 1
                else:
                    weights = np.ones_like(self.fitwl)
                    nite = 3

                cont = spectools.continuum(
                    fwl, syn + fitcont, weights=weights,
                    degr=1, niterate=nite, lower_threshold=3,
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

        # self.fitwl = spectools.get_wl(
        #     fname, pix0key='crpix0', wl0key='crval0', dwlkey='cd1_1',
        #     hdrext=1, dataext=1)
        h = fits.open(fname)

        self.header = h['PRIMARY'].header
        self.fitspec = h['FITSPEC'].data
        self.fitcont = h['FITCONT'].data
        self.resultspec = h['MODEL'].data
        self.em_model = h['SOLUTION'].data
        self.fit_status = h['SOLUTION'].header['fitstat']
        self.fitstellar = h['STELLAR'].data
        func_name = h['SOLUTION'].header['function']

        self.eqw_model = h['EQW_M'].data
        self.eqw_direct = h['EQW_D'].data

        fitwcs = wcs.WCS(h['FITSPEC'].header)
        self.fitwl = fitwcs.wcs_pix2world(np.arange(len(self.fitspec)), 0)[0]

        fit_info = {}
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

        try:
            par_table = h['PARNAMES'].data
            self.component_names = par_table['component'][::self.npars]
        except KeyError:
            pass

        h.close()

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
        star = self.fitstellar

        ax.plot(wl, s)
        ax.plot(wl, star)
        ax.plot(wl, c + star)
        ax.plot(wl, c + star + f(wl, p))

        npars = self.npars
        parnames = self.parnames

        # NOTE: This is only here for backwards compatibility with
        # fits that were run before component names were written to
        # a FITS table.
        if not hasattr(self, 'component_names'):
            self.component_names = [str(i) for i in range(0, len(p) / npars)]

        if len(p) > npars:
            for i in np.arange(0, len(p), npars):
                ax.plot(wl, c + star + f(wl, p[i: i + npars]), 'k--')

        pars = ((npars + 1) * '{:12s}' + '\n').format('Name', *parnames)
        for i in np.arange(0, len(p), npars):
            pars += (
                ('{:12s}{:12.2e}' + (npars - 1) * '{:12.2f}' + '\n')
                .format(self.component_names[int(i / npars)], *p[i:i + npars]))

        if output == 'stdout':
            print(pars)

        if show:
            plt.show()

        if output == 'return':
            return pars
        else:
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
