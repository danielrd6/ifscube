"""
Functions for the analysis of integral field spectroscopy.

Author: Daniel Ruschel Dutra
Website: https://github.com/danielrd6/ifscube
"""

import numpy as np
import pyfits as pf
import ifscube.spectools as st
# import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from copy import deepcopy
from voronoi_2d_binning import voronoi_2d_binning
# from scipy.optimize import curve_fit
from scipy.optimize import minimize
# from scipy.ndimage import gaussian_filter as gf
# from scipy.integrate import trapz
from scipy.interpolate import interp1d
from scipy import ndimage
import ifscube.elprofile as lprof
import ppxf
import ppxf_util


class nanSolution:

    def ppxf(self, sol, galaxy, bestfit):
        self.sol = np.array([np.nan for i in range(len(sol))])
        self.galaxy = np.array([np.nan for i in range(len(galaxy))])
        self.bestfit = np.array([np.nan for i in range(len(galaxy))])


def progress(x, xmax, steps=10):
    try:
        if x % (xmax / steps) == 0:
            print('{:2.0f}%\r'.format(np.float(x) / np.float(xmax) * 100))
    except ZeroDivisionError:
        pass


class gmosdc:
    """
    A class for dealing with data cubes, originally written to work
    with GMOS IFU.
    """

    def __init__(self, fitsfile, redshift=None, vortab=None, dataext=1,
                 hdrext=0, var_ext=None, nan_spaxels='all'):
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
            g = pf.open(fitsfile)
            self.noise_cube = g[var_ext].data
            self.noise = np.nanmean(g[var_ext].data, 0)
            self.noise[self.nanSpaxels] = np.nan
            self.signal = np.nanmean(self.data, 0)
            self.signal[self.nanSpaxels] = np.nan

            self.noise[np.isinf(self.noise)] =\
                self.signal[np.isinf(self.noise)]

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

    def continuum(self, writefits=False, outimage=None, fitting_window=None,
                  copts=None):
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
            if any(s[:20]) and any(s[-20:]):
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
                    return wl, s
            else:
                c[:, i, j] = np.zeros(len(wl), dtype='float32')

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
            if any(data[snrwindow, i, j]):
                s = data[snrwindow, i, j]
                cont = st.continuum(wl, s, **copts)[1]
                noise[i, j] = np.nanstd(s - cont)
                signal[i, j] = np.nanmean(cont)
            else:
                noise[i, j], signal[i, j] = np.nan, np.nan

        self.noise = noise
        self.signal = signal

        return np.array([signal, noise])

    def wlprojection(self, wl0, fwhm=10, filtertype='box', writefits=False,
                     outimage='wlprojection.fits'):
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

        if filtertype == 'box':
            arrfilt = np.array(
                (self.restwl >= wl0-fwhm/2.) & (self.restwl <= wl0+fwhm/2.),
                dtype='float')
            arrfilt /= trapz(arrfilt, self.restwl)
        elif filtertype == 'gaussian':
            s = fwhm / (2. * np.sqrt(2. * np.log(2.)))
            arrfilt = 1. / np.sqrt(2 * np.pi) *\
                np.exp(-(self.restwl - wl0) ** 2 / 2. / s**2)
        else:
            print(
                'ERROR! Parameter filtertype "{:s}" not understood.'
                .format(filtertype))

        outim = np.zeros(np.shape(self.data)[1:], dtype='float32')

        for i, j in self.spec_indices:
            outim[i, j] = trapz(self.data[:, i, j]*arrfilt, self.restwl)

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

    def plotspec(self, x, y, noise_smooth=30):
        """
        Plots the spectrum at coordinates x,y.

        Parameters
        ----------
        x,y : numbers or tuple
            If x and y are numbers plots the spectrum at the specific
            spaxel. If x and y are two element tuples plots the average
            between x[0],y[0] and x[1],y[1]

        Returns
        -------
        Nothing.
        """

        # fig = plt.figure(1)
        plt.figure(1)
        ax = plt.axes()
        try:
            if len(x) == 2 and len(y) == 2:
                s = np.average(
                    np.average(self.data[:, y[0]:y[1], x[0]:x[1]], 1), 1)
        except TypeError:
            s = self.data[:, y, x]
        ax.plot(self.restwl, s)

        try:
            n = ndimage.gaussian_filter1d(self.noise_cube[:, y, x],
                                          noise_smooth)
            sg = ndimage.gaussian_filter1d(s, noise_smooth)
            ax.fill_between(self.restwl, sg - n, sg + n, edgecolor='',
                            alpha=0.2, color='green')
        except AttributeError:
            pass

        plt.show()

    def linefit(self, p0, function='gaussian', fitting_window=None,
                writefits=False, outimage=None, variance=None, constraints=(),
                bounds=None, inst_disp=1.0, individual_spec=False,
                min_method='SLSQP', minopts=None, copts=None, refit=False,
                spiral_loop=False, spiral_center=None, fit_continuum=True,
                refit_radius=3):
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

        try:
            minopts['eps']
        except TypeError:
            if minopts is None:
                minopts = {'eps': 1e-3}
            else:
                minopts['eps'] = 1e-3

        wl = deepcopy(self.restwl[fw])
        scale_factor = np.nanmean(self.data[fw, :, :])
        data = deepcopy(self.data[fw, :, :]) / scale_factor
        fit_status = np.ones(np.shape(data)[1:], dtype='float32') * -1

        if variance is None:
            try:
                vcube = self.noise_cube[fw, :, :] ** 2
            except AttributeError:
                vcube = np.ones(np.shape(data), dtype='float32')
        elif variance is not None:
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
                for coords in ['xcoords', 'ycoords']])
            vor = np.column_stack([
                v[coords] for coords in ['xcoords', 'ycoords', 'binNum']])
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

                r = minimize(res, x0=p0, method=min_method, bounds=bounds,
                             constraints=constraints, options=minopts)
                if r.status != 0:
                    print(h, r.message)
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

        func_name = pf.getheader(fname, ext=4)['function']
        if func_name == 'gaussian':
            self.fit_func = lprof.gauss
        if func_name == 'gauss_hermite':
            self.fit_func = lprof.gausshermite
        self.em_model = pf.getdata(fname, ext=4)

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
            if np.any(np.isnan(self.em_model[par_indexes, i, j])):
                eqw_model[i, j] = np.nan
                eqw_direct[i, j] = np.nan
            else:
                cond = (self.fitwl > self.em_model[center_index, i, j] -
                        sigma_factor * self.em_model[sigma_index, i, j]) &\
                    (self.fitwl < self.em_model[center_index, i, j] +
                     sigma_factor * self.em_model[sigma_index, i, j])

                cond_data = (
                    self.restwl > self.em_model[center_index, i, j] -
                    sigma_factor * self.em_model[sigma_index, i, j]) & \
                    (
                        self.restwl < self.em_model[center_index, i, j] +
                        sigma_factor * self.em_model[sigma_index, i, j])

                fit = self.fit_func(self.fitwl[cond],
                                    self.em_model[par_indexes, i, j])

                cont = self.fitcont[cond, i, j]
                cont_data = interp1d(
                    self.fitwl,
                    self.fitcont[:, i, j])(self.restwl[cond_data])

                eqw_model[i, j] = trapz(
                    1. - (fit + cont) / cont, x=self.fitwl[cond])

                eqw_direct[i, j] = trapz(
                    1. - self.data[cond_data, i, j] / cont_data,
                    x=self.restwl[cond_data])

        return np.array([eqw_model, eqw_direct])

    def plotfit(self, x, y):
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

        # fig = plt.figure(1)
        plt.figure(1)
        plt.clf()
        ax = plt.axes()

        p = self.em_model[:-1, y, x]
        c = self.fitcont[:, y, x]
        wl = self.fitwl
        f = self.fit_func
        s = self.fitspec[:, y, x]

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
                ax.plot(wl, c + f(wl, p[i: i+npars]), 'k--')

        pars = (npars * '{:10s}' + '\n').format(*parnames)
        for i in np.arange(0, len(p), npars):
            pars += (
                ('{:10.2e}' + (npars-1) * '{:10.2f}' + '\n')
                .format(*p[i:i+npars]))

        print(pars)
        plt.show()

    def channelmaps(self, channels=6, lambda0=None, velmin=None, velmax=None,
                    continuum_width=300, continuum_opts=None, sigma=1e-16):
        """
        Creates velocity channel maps from a data cube.

        Parameters
        ----------
            channels : integer
                Number of channel maps to build
            lambda0 : number
                Central wavelength of the desired spectral feature
            vmin : number
                Mininum velocity in kilometers per second
            vmax : number
                Maximum velocity in kilometers per second
            continuum_width : number
                Width in wavelength for the continuum evaluation window
            continuum_opts : dictionary
                Dicitionary of options to be passed to the
                spectools.continuum function

        Returns
        -------
        """
        # Converting from velocities to wavelength
        wlmin, wlmax = lambda0*(np.array([velmin, velmax])/2.99792e+5 + 1.)
        wlstep = (wlmax - wlmin)/channels
        wl_limits = np.arange(wlmin, wlmax + wlstep, wlstep)

        side = int(np.ceil(np.sqrt(channels)))  # columns
        otherside = int(np.ceil(channels / side))  # lines
        fig = plt.figure()
        plt.clf()

        if continuum_opts is None:
            continuum_opts = {'niterate': 3, 'degr': 5,
                              'upper_threshold': 3, 'lower_threshold': 3}
        cp = continuum_opts
        cw = continuum_width
        fw = lambda0 + np.array([-cw / 2., cw / 2.])

        cont = self.continuum(niterate=cp['niterate'], degr=cp['degr'],
                              upper_threshold=cp['upper_threshold'],
                              lower_threshold=cp['lower_threshold'],
                              fitting_window=fw)
        contwl = self.wl[(self.wl > fw[0]) & (self.wl < fw[1])]
        cont_wl2pix = interp1d(contwl, np.arange(len(contwl)))

        for i in np.arange(channels):
            ax = fig.add_subplot(otherside, side, i+1)
            wl = self.restwl
            wl0, wl1 = wl_limits[i], wl_limits[i+1]
            print(wl[(wl > wl0) & (wl < wl1)])
            wlc, wlwidth = np.average([wl0, wl1]), (wl1-wl0)

            f = self.wlprojection(wlc, fwhm=wlwidth, writefits=False,
                                  filtertype='box')\
                - cont[int(round(cont_wl2pix(wlc)))]
            f[f < sigma] = np.nan
            cp = continuum_opts

            ax.imshow(f, interpolation='none', aspect=1)
            ax.annotate(
                '{:.0f}'.format((wlc - lambda0)/lambda0*2.99792e+5),
                xy=(0.1, 0.8), xycoords='axes fraction', color='k')
            if i % side != 0:
                ax.set_yticklabels([])
            if i / float((otherside-1) * side) < 1:
                ax.set_xticklabels([])

        fig.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    def voronoi_binning(self, targetsnr=10.0, writefits=False,
                        outfile=None, clobber=False, writevortab=True):
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
            x = np.shape(self.noise)
        except AttributeError:
            print(
                'This function requires prior execution of the snr_eval' +
                'method.')
            return

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

        binned = np.zeros(np.shape(self.data), dtype='float32')
        binned[:, ynan, xnan] = np.nan

        for i in np.arange(binNum.max() + 1):
            samebin = v[:, 2] == i
            samebin_coords = v[samebin, :2]

            binspec = np.average(
               self.data[:, samebin_coords[:, 0], samebin_coords[:, 1]],
               axis=1)

            for k in samebin_coords:
                binned[:, k[0], k[1]] = binspec

        if writefits:
            hdulist = pf.open(self.fitsfile)
            hdr = self.header
            try:
                hdr['REDSHIFT'] = self.redshift
            except KeyError:
                hdr['REDSHIFT'] = (self.redshift,
                                   'Redshift used in GMOSDC')
            hdr['VORBIN'] = (True, 'Processed by Voronoi binning?')
            hdr['VORTSNR'] = (targetsnr, 'Target SNR for Voronoi binning.')

            hdulist[self.hdrext].header = hdr

            tbhdu = pf.BinTableHDU.from_columns([
                pf.Column(name='xcoords', format='i8', array=x),
                pf.Column(name='ycoords', format='i8', array=y),
                pf.Column(name='binNum', format='i8', array=binNum)],
                name='VOR')

            hdulist.append(tbhdu)
            hdulist[1].data = binned

            if outfile is None:
                outfile = '{:s}bin.fits'.format(self.fitsfile[:-4])

            hdulist.writeto(outfile, clobber=clobber)

        self.binned_cube = binned

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
                        quiet=False, deg=4, mask=None, cushion=100.):
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

        w0, w1 = fitting_window
        fw = (self.wl >= w0) & (self.wl < w1)

        baseCut = (base_wl > w0 - cushion) & (base_wl < w1 + cushion)
        base_spec = base_spec[:, baseCut]
        base_wl = base_wl[baseCut]

        # Here we use the goodpixels as the fitting window
        # gp = np.arange(np.shape(self.data)[0])[fw]
        gp = np.arange(len(self.wl[fw]))

        if mask is not None:
            if len(mask) == 1:
                gp = gp[
                    (self.wl[fw] < mask[0][0]) | (self.wl[fw] > mask[0][1])]
            else:
                m = np.array([
                    (self.wl[fw] < i[0]) | (self.wl[fw] > i[1])
                    for i in mask])
                gp = gp[np.sum(m, 0) == m.shape[0]]

        lamRange1 = self.wl[fw][[1, -1]]
        centerSpaxel = np.array(np.shape(self.data[0])) / 2
        gal_lin = deepcopy(self.data[fw, centerSpaxel[0], centerSpaxel[1]])
        galaxy, logLam1, velscale = ppxf_util.log_rebin(
            lamRange1, gal_lin)

        lamRange2 = base_wl[[1, -1]]
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

        c = 299792.458
        dv = (logLam2[0]-logLam1[0])*c  # km/s
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
            dtype='float32')
        ppxf_spec = np.zeros(
            (len(galaxy), np.shape(self.data)[1], np.shape(self.data)[2]),
            dtype='float32')
        ppxf_model = np.zeros(np.shape(ppxf_spec), dtype='float32')

        nspec = len(xy)

        for k, h in enumerate(xy):
            progress(k, nspec, 10)
            i, j = h

            gal_lin = deepcopy(self.data[fw, i, j])
            galaxy, logLam1, velscale = ppxf_util.log_rebin(lamRange1, gal_lin)
            galaxy = galaxy / np.nanmean(galaxy)

            if np.any(np.isnan(galaxy)):
                pp = nanSolution()
                pp.ppxf(ppxf_sol[:, 0, 0], galaxy, galaxy)
            else:
                pp = ppxf.ppxf(
                    templates, galaxy, noise, velscale, start, goodpixels=gp,
                    plot=plotfit, moments=4, degree=deg, vsyst=dv, quiet=quiet)
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
        self.ppxf_spec = ppxf_spec
        self.ppxf_model = ppxf_model
        self.ppxf_wl = self.wl[fw]
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

            # Creates the fitted spectrum extension
            hdr = pf.Header()
            hdr['object'] = ('spectrum', 'Data in this extension')
            hdr['CRPIX3'] = (1, 'Reference pixel for wavelength')
            hdr['CRVAL3'] = (self.wl[0], 'Reference value for wavelength')
            hdr['CD3_3'] = (np.average(np.diff(self.wl)), 'CD3_3')
            h.append(pf.ImageHDU(data=self.ppxf_spec, header=hdr))

            # Creates the fitted model extension.
            hdr['object'] = 'model'
            h.append(pf.ImageHDU(data=self.ppxf_model, header=hdr))

            # Creates the solution extension.
            hdr['object'] = 'parameters'
            h.append(pf.ImageHDU(data=self.ppxf_sol, header=hdr))

            # Creates the wavelength extension.
            hdr['object'] = 'wavelength'
            h.append(pf.ImageHDU(data=self.ppxf_wl, header=hdr))

            # Creates the goodpixels extension.
            hdr['object'] = 'goodpixels'
            h.append(pf.ImageHDU(data=self.ppxf_goodpixels, header=hdr))

            h.writeto(outimage)

    def lineflux(self, amplitude, sigma):
        """
        Calculates the flux in a line given the amplitude and sigma
        of the gaussian function that fits it.
        """

        lf = amplitude * abs(sigma) * np.sqrt(2. * np.pi)

        return lf
