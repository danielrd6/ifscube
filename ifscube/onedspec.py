# STDLIB
from copy import deepcopy
import warnings

# THIRD PARTY
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from astropy import wcs, table, constants, units
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


class Spectrum:

    def __init__(self, *args, **kwargs):

        self.component_names = None
        self.em_model = None
        self.eqw_direct = None
        self.eqw_model = None
        self.feature_wl = None
        self.fit_func = None
        self.fit_info = None
        self.fit_status = None
        self.fitbounds = None
        self.fitcont = None
        self.fitspec = None
        self.fitstellar = None
        self.fitwl = None
        self.flux_err = None
        self.header = None
        self.initial_guess = None
        self.npars = None
        self.parnames = None
        self.r = None
        self.resultspec = None
        self.valid_pixels = None

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
                        assert hdu[j].data.shape == self.data.shape, \
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

        self.restwl = self.dopcor(self.redshift, self.wl)

    @staticmethod
    def dopcor(z, wl):

        restwl = wl / (1. + z)
        return restwl

    @staticmethod
    def sigma_lambda(sigma_vel, rest_wl):
        return sigma_vel * rest_wl / constants.c.to('km/s').value

    def center_wavelength(self, idx):
        j = idx * self.npars
        lam = (self.em_model[j + 1] * units.km / units.s).to(
            units.angstrom, equivalencies=units.doppler_relativistic(
                self.feature_wl[idx] * units.angstrom))
        return lam.value

    def _fit_table(self):

        cnames = self.component_names
        pnames = self.parnames

        c = np.array([[i for _ in pnames] for i in cnames]).flatten()
        p = np.array([[i for i in pnames] for _ in cnames]).flatten()

        t = table.Table([c, p], names=('component', 'parameter'))
        h = fits.table_to_hdu(t)

        return h

    def _write_linefit(self, args):

        suffix = args['suffix']
        out_image = args['out_image']
        # Basic tests and first header
        if out_image is None:
            if suffix is None:
                suffix = '_linefit'
            out_image = self.fitsfile.replace('.fits', suffix + '.fits')

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
        hdu = self._fit_table()
        hdu.name = 'PARNAMES'
        h.append(hdu)

        h.writeto(out_image, overwrite=args['overwrite'])

    @staticmethod
    def _within_bounds(x, bounds):

        low, high = bounds

        if (low is not None) and (high is not None):
            return (x > low) & (x < high)

        elif (low is not None) and (high is None):
            return x > low

        elif (low is None) and (high is not None):
            return x < high

        else:
            return True

    def guess_parser(self, p):

        npars = len(self.parnames)
        for i in p[::npars]:
            try:
                if i == 'peak':
                    p[p.index(i)] = self.data[self.valid_pixels].max()
                elif i == 'mean':
                    p[p.index(i)] = self.data[self.valid_pixels].mean()
                elif i == 'median':
                    p[p.index(i)] = np.median(self.data[self.valid_pixels])
            except ValueError:
                p[p.index(i)] = np.nan

        return p

    @staticmethod
    def feature_slice(wl, lam, s):

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

    def guess_parameters(self, data, wl, p0, npars_pc, bounds):

        new_p0 = deepcopy(p0)

        for i in range(0, p0.size, npars_pc):

            ws = self.feature_slice(wl, p0[i + 1], p0[i + 2] * 3)

            if ws is not None:
                new_p0[i] = np.max(data[ws])
                new_p0[i + 1] = np.average(wl[ws], weights=data[ws])
                new_p0[i + 2] = np.average(
                    abs(wl[ws] - wl[ws].mean()), weights=data[ws])

            for j in range(i, i + 3):
                if not self._within_bounds(new_p0[j], bounds[j]):
                    new_p0[j] = deepcopy(p0[j])

        return new_p0

    @staticmethod
    def optimize_mask(wl, feature_wl, sigma, width=20, catch_error=False):
        """
        Creates the fit optimization window by selecting the appropriate portions of the spectrum.

        Parameters
        ----------
        wl: numpy.ndarray
            Wavelength coordinates
        feature_wl: numpy.ndarray
            Central wavelength of the spectral features to be fit.
        sigma: numpy.ndarray
            Sigma of each of the spectral features.
        width: float
            Number of sigmas to each side that will be considered in the fit.
        catch_error: bool
            Interrupts the program when the optimization window falls outside the available wavelengths.

        Returns
        -------
        mask: numpy.ndarray
            True for all wavelengths within the window of interest for the fit.
        """

        mask = np.zeros_like(wl).astype(bool)

        for lam, s in zip(feature_wl, sigma):

            low_lam = (lam - width * s)
            up_lam = (lam + width * s)

            if catch_error:
                assert low_lam > wl[0], 'Lower limit in optimization window is below the lowest available wavelength.'
                assert up_lam < wl[-1], 'Upper limit in optimization window is above the highest available wavelength.'
            else:
                if low_lam < wl[0]:
                    low_lam = wl[0]
                if up_lam > wl[-1]:
                    up_lam = wl[-1]

            wl_lims = [
                wl[wl <= low_lam][-1],
                wl[wl >= up_lam][0]]
            idx = [np.where(wl == i)[0][0] for i in wl_lims]

            ws = slice(idx[0], idx[1])

            mask[ws] = True

        return mask

    def linefit(self, p0, feature_wl, function='gaussian', fitting_window=None, write_fits=False, out_image=None,
                variance=None, constraints=None, bounds=None, instrument_dispersion=1.0, min_method='SLSQP',
                minopts=None, copts=None, weights=None, verbose=False, fit_continuum=False, component_names=None,
                overwrite=False, eqw_opts=None, trivial=False, suffix=None, optimize_fit=False,
                optimization_window=10.0, guess_parameters=False, test_jacobian=False, good_minfraction=.8):
        """
        Fits a spectral features.

        Parameters
        ----------
        p0: array-like
            Initial guess for the fitting funcion, consisting of a list
            of N*M parameters for M components of **function**. In the
            case of a gaussian fucntion, these parameters must be given
            as [amplitude0, center0, sigma0, amplitude1, center1, ...].
        feature_wl: array-like
            List of rest wavelengths for the spectral features to be fit.
        function: str
            The function to be fitted to the spectral features.
            Available options and respective parameters are:

                - 'gaussian': amplitude, central wavelength in angstroms, sigma in angstroms
                - 'gauss_hermite' : amplitude, central wavelength in angstroms, sigma in angstroms, h3 and h4

        fitting_window: tuple
            Lower and upper wavelength limits for the fitting
            algorithm. These limits should allow for a considerable
            portion of continuum besides the desired spectral features.
        write_fits: bool
            Writes the results in a FITS file.
        out_image: string
            Name of the FITS file in which to write the results.
        variance: float, 1D, 2D or 3D array
            The variance of the flux measurements. It can be given
            in one of four formats. If variance is a float it is
            applied as a constant to the whole spectrum. If given as 1D
            array it assumed to be a spectrum that will be applied to
            the whole cube. As 2D array, each spaxel will be applied
            equally to all wavelengths. Finally the 3D array must
            represent the variance for each element of the data cube.
            It defaults to None, in which case it does not affect the
            minimization algorithm, and the returned Chi2 will be in
            fact just the fit residuals.
        constraints: dict or sequence of dicts
            See scipy.optimize.minimize
        bounds: list
            Bounds for the fitting algorithm, given as a list of
            [xmin, xmax] pairs for each x parameter.
        instrument_dispersion: float
            Instrumental dispersion in pixel units. This argument is
            used to evaluate the reduced chi squared. If let to default
            it is assumed that each wavelength coordinate is a degree
            of freedom. The physically sound way to do it is to use the
            number of dispersion elements in a spectrum as the degrees
            of freedom.
        min_method: str
            Minimization method. See scipy.optimize.minimize.
        minopts: dict, optional
            Dictionary of options to be passed to the minimization
            routine. See scipy.optimize.minimize.
        copts: dict, optional
            Arguments to be passed to the spectools.continuum function.
        weights: numpy.ndarray, optional
            Array of weights with the same dimensions as the input spectrum.
        verbose: bool
            Prints progress messages.
        fit_continuum: bool
            If True fits the continuum just before attempting to fit
            the emission lines. Setting this option to False will
            cause the algorithm to look for self.cont, which should
            contain a data cube of continua.
        component_names: list, optional
            Names of the spectral features to be fitted.
        overwrite: bool
            Overwrites previously written output file.
        eqw_opts: dict
            Options for the equivalent width function.
        trivial: bool
            Attempts a fit with a trivial solution, and if the rms is
            smaller than the fit with the intial guess, selects the
            trivial fit as the correct solution.
        suffix: str, optional
            String to be appended to the input file name in case no output name is given.
        optimize_fit: bool
            Perform function evaluations only near the spectral features being fit.
        optimization_window: float
            Width of the window where the fit functions will be evaluated in units of sigma.
        guess_parameters: bool
            Attempt to guess to initial parameters of the fit.
        test_jacobian: bool
            Read the Jacobian matrix and set those parameters which yield zeros to nan.
        good_minfraction: float
            Minimum fraction of non-flagged pixels within the portion of the spectra selected for the fit.

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
            fit_func = lprof.gaussvel
            self.parnames = ('A', 'v', 's')
        elif function == 'gauss_hermite':
            fit_func = lprof.gausshermitevel
            self.parnames = ('A', 'v', 's', 'h3', 'h4')
        else:
            raise NameError('Unknown function "{:s}".'.format(function))

        npars_pc = len(self.parnames)

        self.fit_func = fit_func
        self.npars = npars_pc

        # Sets a pre-made nan vector for nan solutions.
        npars = len(p0)
        self.em_model = np.array([np.nan for i in range(npars + 1)])

        if fitting_window is None:
            fw = np.ones_like(self.data).astype('bool')
        else:
            fw = (self.restwl > fitting_window[0]) & \
                 (self.restwl < fitting_window[1])
        if not np.any(fw):
            raise RuntimeError(
                'Fitting window outside the available wavelength range.')
        zero_spec = np.zeros_like(self.restwl[fw])

        assert self.restwl[fw].min() < np.min(feature_wl),\
            'Attempting to fit a spectral feature below the fitting window.'

        assert self.restwl[fw].max() > np.max(feature_wl), \
            'Attempting to fit a spectral feature above the fitting window.'

        if component_names is None:
            self.component_names = [
                'C_{:03d}'.format(i) for i in range(int(npars / npars_pc))]
        else:
            self.component_names = component_names

        # NOTE: These next lines, until the good fraction test, must be
        # executed always, to avoid problems with datacube.
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
        self.initial_guess = p0
        self.eqw_model = np.nan
        self.eqw_direct = np.nan

        p0 = np.array(self.guess_parser(p0))
        self.fitbounds = bounds

        #
        # Avoids fit if more than certain fraction of the pixels are
        # flagged.
        #
        if np.sum(valid_pixels) < (good_minfraction * valid_pixels[fw].size):
            self.fit_status = 98
            warnings.warn(
                message=RuntimeWarning(
                    'Minimum fraction of good pixels not reached!\n'
                    'User set threshold: {:.2f}\n'
                    'Measured good fracion: {:.2f}'
                    .format(
                        good_minfraction,
                        np.sum(valid_pixels) / valid_pixels[fw].size)))
            return

        if weights is None:
            weights = np.ones_like(data)

        #
        # Pseudo continuum fitting.
        #

        if copts is None:
            copts = dict(
                niterate=5, degr=4, upper_threshold=2, lower_threshold=2)

        copts.update(dict(output='polynomial'))

        if hasattr(self, 'continuum'):
            cont = self.continuum[valid_pixels]
            self.fitcont = self.continuum[fw]
        else:
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

        if bounds is None:
            sbounds = [[None, None] for i in range(len(p0))]
        else:
            sbounds = scale_bounds(bounds, scale_factor, npars_pc)

        #
        # Optimization mask
        #
        if optimize_fit:
            sigma = np.array([p0[i] if sbounds[i][1] is None else sbounds[i][1] for i in range(2, len(p0), npars_pc)])
            sigma_lam = self.sigma_lambda(sigma, feature_wl)
            opt_mask = self.optimize_mask(wl=wl, feature_wl=feature_wl, sigma=sigma_lam, width=optimization_window)
            if not np.any(opt_mask):
                self.fit_status = 80
                return
        else:
            opt_mask = np.ones_like(wl).astype(bool)

        if guess_parameters:
            p0 = self.guess_parameters(s, wl, p0, npars_pc, sbounds)
        self.initial_guess = p0

        #
        # Here the actual fit begins
        #

        def res(x):
            m = fit_func(wl[opt_mask], feature_wl, x)
            a = w[opt_mask] * (s[opt_mask] - m) ** 2
            b = a / v[opt_mask]
            rms = np.sqrt(np.sum(b))
            return rms

        if minopts is None:
            minopts = {'eps': 1e-3}
        if constraints is None:
            constraints = []
        r = minimize(res, x0=p0, method=min_method, bounds=sbounds, constraints=constraints, options=minopts)

        # Perform the fit a second time with the RMS as the flux
        # initial guess. This was added after a number of fits returned
        # high flux values even when no lines were present.
        if trivial:
            fit_rms = res(r.x)
            new_p = deepcopy(r.x)
            for i in range(0, r.x.size, npars_pc):
                trivial_p = deepcopy(r.x)
                trivial_p[i] = 0
                if fit_rms > res(trivial_p):
                    new_p[i:i + npars_pc] = np.nan
            r.x = new_p

        if test_jacobian:
            for i in range(0, r.x.size, npars_pc):
                if np.any(r.jac[i:i + npars_pc] == 0):
                    r.x[i:i + npars_pc] = np.nan
                    # r.status = 94
                    # r.message = 'Jacobian has terms equal to zero.'

        self.r = r

        if verbose and r.status != 0:
            print(r.message, r.status)

        # Reduced chi squared of the fit.
        chi2 = np.sum((s - fit_func(wl, feature_wl, r.x)) ** 2 / v)
        nu = len(s) / instrument_dispersion - npars - len(constraints) - 1
        red_chi2 = chi2 / nu

        self.fit_status = r.status

        p0[::npars_pc] *= scale_factor
        p = np.append(r['x'], red_chi2)
        p[0:-1:npars_pc] *= scale_factor

        self.resultspec = self.fitstellar + self.fitcont + fit_func(self.fitwl, feature_wl, r['x']) * scale_factor

        self.em_model = p
        self.feature_wl = feature_wl
        if eqw_opts is None:
            eqw_opts = {}
        self.eqw(**eqw_opts)

        if write_fits:
            self._write_linefit(args=locals())
        return

    def eqw(self, sigma_factor=5.0, continuum_windows=None):
        """
        Evaluates the equivalent width of a previous linefit.

        Parameters
        ----------
        sigma_factor: float
            Radius of integration as a number of line sigmas.
        continuum_windows: numpy.ndarray
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

            sigma_index = 2 + npars * component_index

            # Wavelength vector of the line fit
            fwl = self.fitwl
            # Center wavelength coordinate of the fit
            cwl = self.center_wavelength(component_index)
            # Sigma of the fit
            sig = self.sigma_lambda(
                self.em_model[sigma_index], self.feature_wl[component_index])
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
                    fwl[cond],
                    np.array([self.feature_wl[component_index]]),
                    self.em_model[par_indexes])
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
                    assert len(cwin) == 4, 'Windows must be an ' \
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
                    upper_threshold=3, output='function')[1][cond]

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

    def dn4000(self):
        """
        Dn4000 index based on Balogh et al. 1999 (ApJ, 527, 54).

        The index is defined as the ratio between continuum fluxes
        at 3850A-3950A and 4000A-4100A.

        red / blue

        """

        warn_message = 'Dn4000 could not be evaluated because the ' \
                       'spectrum does not include wavelengths bluer than 3850.'

        if self.restwl[0] > 3850:
            warnings.warn(RuntimeWarning(warn_message))
            dn = np.nan

        else:
            # Mask for the blue part
            bm = (self.restwl >= 3850) & (self.restwl <= 3950)
            # Mask for the red part
            rm = (self.restwl >= 4000) & (self.restwl <= 4100)
            # Dn4000
            dn = np.sum(self.data[rm]) / np.sum(self.data[bm])

        return dn

    def fit_uncertainties(self, snr=10):

        if self.fit_func == lprof.gaussvel:
            peak = self.em_model[0]
            flux = self.em_model[0] * self.em_model[2] * np.sqrt(2. * np.pi)
        elif self.fit_func == lprof.gausshermitevel:
            # This is a crude estimate for the peak, and only works
            # for small values of h3 and h4.
            peak = self.em_model[0] / self.em_model[2] / np.sqrt(2. * np.pi)
            flux = self.em_model[0]
        else:
            raise RuntimeError('Could not understand the fit function.')

        if hasattr(self, 'err'):
            s_peak = interp1d(self.wl, self.err)(self.em_model[1])
        else:
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

        self.feature_wl = np.array([
            float(i[1]) for i in h['fitconfig'].data
            if 'rest_wavelength' in i['parameters']])

        fit_info = {'function': func_name}

        if func_name == 'gaussian':
            self.fit_func = lprof.gaussvel
            self.npars = 3
            self.parnames = ('A', 'v', 's')
        elif func_name == 'gauss_hermite':
            self.fit_func = lprof.gausshermitevel
            self.npars = 5
            self.parnames = ('A', 'v', 's', 'h3', 'h4')
        else:
            raise RuntimeError('Unknown function {:s}'.format(func_name))

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
        show: bool
            Shows the plot.
        axis: matplotlib.pyplot.Axes
            Axes instance onto which to draw the plot. If *None* a new instance will be created.
        output: str
            Output destination for the plot results. Possible values are:

                - 'stdout': prints to standard output
                - 'return': returns the results as string.

        Returns
        -------
        Nothing.
        """

        if self.fit_status != 0:
            warnings.warn(RuntimeWarning('Fit was unsuccessful with exit status {:d}.'.format(self.fit_status)))

        if axis is None:
            fig = plt.figure(1)
            plt.clf()
            ax = fig.add_subplot(111)
        else:
            ax = axis

        p = deepcopy(self.em_model[:-1])
        pp = np.array([i if np.isfinite(i) else 0.0 for i in p])
        rest_wl = self.feature_wl
        c = self.fitcont
        wl = self.fitwl
        f = self.fit_func
        s = self.fitspec
        star = self.fitstellar

        ax.plot(wl, s)
        ax.plot(wl, star)
        ax.plot(wl, c + star)
        ax.plot(wl, c + star + f(wl, rest_wl, pp))

        npars = self.npars
        parnames = self.parnames

        # NOTE: This is only here for backwards compatibility with
        # fits that were run before component names were written to
        # a FITS table.
        if not hasattr(self, 'component_names'):
            self.component_names = [str(i) for i in range(0, len(p) / npars)]

        if len(p) > npars:
            for i in np.arange(0, len(p), npars):
                rwl = np.array([rest_wl[int(i / npars)]])
                ax.plot(wl, c + star + f(wl, rwl, p[i: i + npars]), 'k--')

        pars = ((npars + 1) * '{:12s}' + '\n').format('Name', *parnames)
        for i in np.arange(0, len(p), npars):
            pars += (
                ('{:12s}{:12.2e}' + (npars - 1) * '{:12.2f}' + '\n').format(
                    self.component_names[int(i / npars)], *p[i:i + npars])
            )

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
            ax = plt.gca()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax.plot(self.wl, self.data)
        plt.show()
