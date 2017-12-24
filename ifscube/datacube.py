# STDLIB
from copy import deepcopy

# THIRD PARTY
import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from astropy import constants, units, table
from astropy.io import fits
import progressbar

# LOCAL
from . import cubetools, spectools, onedspec
from . import plots as ifsplots
from . import elprofile as lprof


class Cube:

    """
    A class for dealing with IFS data cubes.
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

    def __set_spec_indices__(self):

        if self.spatial_mask is None:
            self.spatial_mask = np.zeros_like(self.data[0]).astype('bool')

        self.spec_indices = np.column_stack([
            np.ravel(
                np.indices(np.shape(self.data)[1:])[0][~self.spatial_mask]),
            np.ravel(
                np.indices(np.shape(self.data)[1:])[1][~self.spatial_mask]),
        ])

    def __arg2cube__(self, arg, cube):

        if len(np.shape(arg)) == 0:
            cube *= arg
        elif len(np.shape(arg)) == 1:
            for i, j in self.spec_indices:
                cube[:, i, j] = arg
        elif len(np.shape(arg)) == 2:
            for i, j in enumerate(cube):
                cube[i] = arg

        return cube

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
        function = args['function']
        total_pars = self.em_model.shape[0] - 1

        hdr['object'] = 'parameters'
        hdr['function'] = (function, 'Fitted function')
        hdr['nfunc'] = (total_pars / self.npars, 'Number of functions')
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

        # Creates the minimize's exit status extension
        hdr['object'] = 'status'
        hdu = fits.ImageHDU(data=self.fit_status, header=hdr)
        hdu.name = 'STATUS'
        h.append(hdu)

        # Creates the spatial mask extension
        hdr['object'] = 'spatial mask'
        hdu = fits.ImageHDU(data=self.spatial_mask.astype(int), header=hdr)
        hdu.name = 'MASK2D'
        h.append(hdu)

        # Creates the spaxel indices extension as fits.BinTableHDU.
        hdr['object'] = 'spaxel_coords'
        t = table.Table(self.spec_indices, names=('row', 'column'))
        hdu = fits.table_to_hdu(t)
        hdu.name = 'SPECIDX'
        h.append(hdu)

        # Creates component and parameter names table.
        hdr['object'] = 'parameter names'
        hdu = self.__fitTable__()
        hdu.name = 'PARNAMES'
        h.append(hdu)

        h.writeto(outimage, overwrite=args['overwrite'])

    def __write_eqw__(self, eqw, args):

        outimage = args['outimage']
        # Basic tests and first header

        hdr = fits.Header()

        # Creates MEF output.
        h = fits.HDUList()
        h.append(fits.PrimaryHDU(header=self.header))
        h[0].name = 'PRIMARY'

        # Creates the model equivalent width extension
        hdr['object'] = ('eqw_model', 'EqW based on emission model.')
        hdr['sigwidth'] = (
            args['sigma_factor'], 'Line width in units of sigma.')
        hdr['bunit'] = ('angstrom', 'Unit of pixel values.')
        hdr['l_idx'] = (args['component'], 'Line number in fit output')

        if hasattr(self, 'component_names'):
            hdr['l_name'] = (
                self.component_names[args['component']],
                'Line name')

        if args['windows'] is not None:
            hdr['c_blue0'] = (
                args['windows'][0], 'lower limit, blue continuum section')
            hdr['c_blue1'] = (
                args['windows'][1], 'upper limit, blue continuum section')
            hdr['c_red0'] = (
                args['windows'][2], 'lower limit, red continuum section')
            hdr['c_red1'] = (
                args['windows'][3], 'upper limit, red continuum section')

        h.append(fits.ImageHDU(data=eqw[0], header=hdr, name='EQW_M'))

        # Creates the direct equivalent width extension
        hdr['object'] = (
            'eqw_direct', 'EqW measured directly on the spectrum.')
        hdr['sigwidth'] = (
            args['sigma_factor'], 'Line width in units of sigma.')
        h.append(fits.ImageHDU(data=eqw[1], header=hdr, name='EQW_D'))

        # Creates the minimize's exit status extension
        hdr['object'] = 'fit_status'
        h.append(
            fits.ImageHDU(data=self.fit_status, header=hdr, name='STATUS'))

        h.writeto(outimage)

    def __spiral__(self, xy, spiral_center=None):

        if self.binned:
            y, x = xy[:, 0], xy[:, 1]
        else:
            y, x = self.spec_indices[:, 0], self.spec_indices[:, 1]

        if spiral_center is None:
            spiral_center = (x.max() / 2., y.max() / 2.)

        r = np.sqrt(
            (x - spiral_center[0]) ** 2 + (y - spiral_center[1]) ** 2)

        t = np.arctan2(y - spiral_center[1], x - spiral_center[0])
        t[t < 0] += 2 * np.pi

        b = np.array([
            (np.ravel(r)[i], np.ravel(t)[i]) for i in
            range(len(np.ravel(r)))], dtype=[
                ('radius', 'f8'), ('angle', 'f8')])

        s = np.argsort(b, axis=0, order=['radius', 'angle'])
        xy = np.column_stack([np.ravel(y)[s], np.ravel(x)[s]])

        return xy

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

        c = np.zeros_like(data)

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
                    cont = spectools.continuum(wl, s, **copts)
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

            fits.writeto(outimage, data=c, header=hdr)

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
                cont = spectools.continuum(wl, s, **copts)[1]
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

        outim = cubetools.wlprojection(
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

            fits.writeto(outimage, data=outim, header=hdr)

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

            n = gaussian_filter(n, noise_smooth)
            sg = gaussian_filter(s, noise_smooth)

            ax.fill_between(self.restwl, sg - n, sg + n, edgecolor='',
                            alpha=0.2, color='green')

        plt.show()

    def linefit(self, p0, writefits=False, outimage=None, overwrite=False,
                individual_spec=False, refit=False,
                update_bounds=False, bound_range=.1, spiral_loop=False,
                spiral_center=None, refit_radius=3, sig_threshold=0,
                par_threshold=0, verbose=False, **kwargs):
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
        sig_threshold : number
            Fits which return *par_threshold* below this number of
            times the local noise will be set to nan. If set to 0 this
            criteria is ignored.
        par_threshold : integer
            Parameter which must be above the noise threshold to be
            considered a valid fit.

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

        fitting_window = kwargs.get('fitting_window', None)
        if fitting_window is not None:
            fw_mask = (
                (self.restwl > fitting_window[0])
                & (self.restwl < fitting_window[1]))
            fit_npixels = np.sum(fw_mask)
        else:
            fw_mask = np.ones_like(self.restwl).astype('bool')
            fit_npixels = self.restwl.size
        fit_shape = (fit_npixels,) + self.data.shape[1:]

        self.fit_status = np.ones(np.shape(self.data)[1:], dtype='int') * -1

        #
        # Sets the variance cube
        #
        vcube = self.variance
        variance = kwargs.get('variance', None)
        if variance is not None:
            vcube = self.__arg2cube__(variance, vcube)

        #
        # Set the weight cube.
        #
        wcube = self.weights
        weights = kwargs.get('weights', None)
        if weights is not None:
            wcube = self.__arg2cube__(weights, wcube)

        #
        # Set the flags cube.
        #
        flag_cube = self.flags
        flags = kwargs.get('flags', None)
        if flags is not None:
            flag_cube = self.__arg2cube__(flags, flag_cube)

        npars = len(p0)
        sol = np.zeros((npars + 1,) + self.data.shape[1:])
        self.fitcont = np.zeros(fit_shape)
        self.fitspec = np.zeros(fit_shape)
        self.fitstellar = np.zeros(fit_shape)
        self.resultspec = np.zeros(fit_shape)
        self.fitweights = wcube
        self.initial_guess = np.zeros((npars,) + self.data.shape[1:])
        self.fitbounds = np.zeros((npars * 2,) + self.data.shape[1:])

        if self.binned:
            v = self.voronoi_tab
            xy = np.column_stack([
                v[np.unique(v['binNum'], return_index=True)[1]][coords]
                for coords in ['ycoords', 'xcoords']])
            vor = np.column_stack([
                v[coords] for coords in ['ycoords', 'xcoords', 'binNum']])
        else:
            xy = self.spec_indices

        # Saves the original bounds in case the bound updater is used.
        original_bounds = deepcopy(kwargs.get('bounds', None))

        Y, X = np.indices(fit_shape[1:])
        if individual_spec:
            if individual_spec == 'peak':
                xy = [cubetools.peak_spaxel(self.data[fw_mask])[::-1]]
                if verbose:
                    print(
                        'Individual spaxel: {:d}, {:d}\n'.format(*xy[0][::-1]))
            else:
                xy = [individual_spec[::-1]]
        elif spiral_loop:
            if spiral_center == 'peak':
                spiral_center = cubetools.peak_spaxel(self.data[fw_mask])
            xy = self.__spiral__(xy, spiral_center=spiral_center)

        if verbose:
            iterador = progressbar.ProgressBar()(xy)
        else:
            iterador = xy

        is_first_spec = True
        for h in iterador:

            i, j = h
            if self.binned:
                binNum = vor[(vor[:, 0] == i) & (vor[:, 1] == j), 2]

            cube_slice = (Ellipsis, i, j)

            spec = onedspec.Spectrum()
            spec.restwl = self.restwl
            spec.data = self.data[cube_slice]
            spec.variance = self.variance[cube_slice]
            spec.flags = self.flags[cube_slice]
            spec.stellar = self.stellar[cube_slice]

            if refit and not is_first_spec:

                radsol = np.sqrt((Y - i)**2 + (X - j)**2)
                nearsol = sol[
                    :-1, (radsol < refit_radius) & (self.fit_status == 0)]

                if np.shape(nearsol) == (5, 1):
                    p0 = deepcopy(nearsol.transpose())
                elif np.any(nearsol):
                    p0 = deepcopy(np.average(nearsol.transpose(), 0))

                    if update_bounds:
                        bounds = cubetools.bound_updater(
                            p0, bound_range, bounds=original_bounds)

            spec.linefit(p0, **kwargs)

            # If successful, sets is_first_spec to False.
            if is_first_spec and (spec.fit_status == 0):
                is_first_spec = False

            if not hasattr(self, 'eqw_model'):
                self.eqw_model = np.zeros(
                    (len(spec.component_names),) + self.fit_status.shape)
                self.eqw_direct = np.zeros_like(self.eqw_model)

            self.fit_status[i, j] = spec.fit_status
            self.eqw_model[:, i, j] = spec.eqw_model
            self.eqw_direct[:, i, j] = spec.eqw_direct

            if self.binned:
                for l, m in vor[vor[:, 2] == binNum, :2]:
                    sol[:, l, m] = spec.em_model
                    self.fitcont[:, l, m] = spec.fitcont
                    self.fitspec[:, l, m] = spec.fitspec
                    self.resultspec[:, l, m] = spec.resultspec
                    self.fitstellar[:, l, m] = spec.fitstellar
                    self.eqw_model
                    self.initial_guess[:, l, m] = spec.initial_guess
                    self.fitbounds[:, l, m] = [
                        k if k is not None else np.nan
                        for k in np.array(spec.fitbounds).flatten()]
            else:
                sol[:, i, j] = spec.em_model
                self.fitcont[:, i, j] = spec.fitcont
                self.fitspec[:, i, j] = spec.fitspec
                self.fitstellar[:, i, j] = spec.fitstellar
                self.resultspec[:, i, j] = spec.resultspec
                self.initial_guess[:, i, j] = spec.initial_guess
                self.fitbounds[:, i, j] = [
                    k if k is not None else np.nan
                    for k in np.array(spec.fitbounds).flatten()]

        self.fitwl = spec.fitwl
        self.fit_func = spec.fit_func
        self.parnames = spec.parnames
        self.component_names = spec.component_names
        if spec.fit_func == lprof.gauss:
            function = 'gaussian'
        elif spec.fit_func == lprof.gausshermite:
            function = 'gauss_hermite'
        self.npars = len(spec.parnames)

        self.em_model = sol

        if writefits:
            self.__write_linefit__(args=locals())

        if individual_spec:
            return (
                spec.fitwl, spec.fitspec, spec.fitcont, spec.resultspec,
                spec.r)
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

        fitfile = fits.open(fname)
        if not hasattr(self, 'header'):
            self.header = fitfile[0].header

        self.fitwl = spectools.get_wl(
            fname, pix0key='crpix3', wl0key='crval3', dwlkey='cd3_3',
            hdrext=1, dataext=1)
        self.fitspec = fitfile['FITSPEC'].data
        self.fitcont = fitfile['FITCONT'].data
        self.resultspec = fitfile['MODEL'].data

        self.em_model = fitfile['SOLUTION'].data
        self.fit_status = fitfile['STATUS'].data
        self.fitstellar = fitfile['STELLAR'].data

        try:
            self.eqw_model = fitfile['EQW_M'].data
        except KeyError:
            pass

        try:
            self.eqw_direct = fitfile['EQW_D'].data
        except KeyError:
            pass

        try:
            self.spatial_mask = fitfile['MASK2D'].data.astype('bool')
        except KeyError:
            pass

        try:
            t = fitfile['SPECIDX'].data
            self.spec_indices = np.array([i for i in t])
        except KeyError:
            pass

        fit_info = {}
        func_name = fitfile['SOLUTION'].header['function']
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
            raise IOError('Unkwon function name "{:s}"'.format(func_name))

        try:
            par_table = fitfile['PARNAMES'].data
            self.component_names = par_table['component'][::self.npars]
        except KeyError:
            pass

        fit_info['parameters'] = self.npars
        fit_info['components'] = (self.em_model.shape[0] - 1) / self.npars

        self.fit_info = fit_info

        fitfile.close()

    def w80(self, component, sigma_factor=5, individual_spec=False,
            verbose=False, smooth=0, remove_components=[]):

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

        if center_index > self.em_model.shape[0]:
            raise RuntimeError(
                'Specified component number is higher than the total number '
                'of components.')

        for i, j in xy:

            if verbose:
                print(i, j)

            # Wavelength vector of the line fit
            fwl = self.fitwl
            # Rest wavelength vector of the whole data cube
            # rwl = self.restwl
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

                fit = self.fit_func(
                    fwl[cond], self.em_model[par_indexes, i, j])
                obs_spec = deepcopy(self.fitspec[cond, i, j])

                cont = self.fitcont[cond, i, j]

                # Evaluates the W80 over the modeled emission line.
                w80_model[i, j], m0, m1, mv, ms = spectools.w80eval(
                    fwl[cond], fit, cwl)

                # Evaluating the W80 over the observed spectrum
                # directly is a bit more complicated due to the overlap
                # between neighbouring spectral features. The solution
                # here is to remove the undesired components from the
                # observed spectrum.
                if len(remove_components) > 0:
                    for component in remove_components:
                        ci = component * npars
                        obs_spec -= self.fit_func(
                            fwl[cond], self.em_model[ci:ci + npars, i, j],
                        )
                # And now for the actual W80 evaluation.
                w80_direct[i, j], d0, d1, dv, ds = spectools.w80eval(
                    fwl[cond], obs_spec - cont, cwl, smooth=smooth,
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
        star = self.fitstellar[:, y, x]

        median_spec = np.median(s)

        if median_spec > 0:
            norm_factor = np.int(np.log10(median_spec))
        else:
            return ax

        ax.plot(wl, s / 10. ** norm_factor)
        ax.plot(wl, star / 10. ** norm_factor)
        ax.plot(wl, (star + c) / 10. ** norm_factor)
        ax.plot(wl, (c + star + f(wl, p)) / 10. ** norm_factor)

        ax.set_xlabel(r'Wavelength (${\rm \AA}$)')
        ax.set_ylabel(
            'Flux density ($10^{{{:d}}}\, {{\\rm erg\,s^{{-1}}\,cm^{{-2}}'
            '\,\AA^{{-1}}}}$)'.format(norm_factor))

        npars = self.npars
        parnames = self.parnames

        if len(p) > npars:
            for i in np.arange(0, len(p), npars):
                modeled_spec = (c + star + f(wl, p[i: i + npars]))\
                    / 10. ** norm_factor
                ax.plot(wl, modeled_spec, 'k--')

        pars = ('Red_Chi2: {:.3f}\n'.format(self.em_model[-1, y, x]))
        pars += (npars * '{:10s}' + '\n').format(*parnames)
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

        wlstep = (wlmax - wlmin) / channels
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
            ax = fig.add_subplot(otherside, side, i + 1)
            axes += [ax]
            wl = self.restwl
            wl0, wl1 = wl_limits[i], wl_limits[i + 1]
            print(wl[(wl > wl0) & (wl < wl1)])
            wlc, wlwidth = np.average([wl0, wl1]), (wl1 - wl0)

            f_obs = cubetools.wlprojection(
                arr=self.data, wl=self.restwl, wl0=wlc, fwhm=wlwidth,
                filtertype='box')
            f_cont = cubetools.wlprojection(
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
                '{:.0f}'.format((wlc - lambda0) / lambda0 * 2.99792e+5),
                xy=(0.1, 0.8), xycoords='axes fraction',
                color=text_color)
            if i % side != 0:
                ax.set_yticklabels([])
            if i / float((otherside - 1) * side) < 1:
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

            tmp_data = cubetools.nan_to_nearest(self.data[i])
            tmp_var = cubetools.nan_to_nearest(self.noise_cube[i]) ** 2

            gdata[i] = gaussian_filter(tmp_data, sigma)
            gvar[i] = np.sqrt(gaussian_filter(tmp_var, sigma))

            i += 1

        if writefits:

            hdulist = fits.open(self.fitsfile)
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
        n[n <= 0] = np.average(self.signal[self.signal > 0]) * .5

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
            hdulist = fits.open(self.fitsfile)
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
            tbhdu = fits.BinTableHDU.from_columns(
                [
                    fits.Column(name='xcoords', format='i8', array=x),
                    fits.Column(name='ycoords', format='i8', array=y),
                    fits.Column(name='binNum', format='i8', array=binNum),
                ], name='VOR')

            tbhdu_plus = fits.BinTableHDU.from_columns(
                [
                    fits.Column(name='ubin', format='i8',
                                array=np.unique(binNum)),
                    fits.Column(name='xNode', format='F16.8', array=xNode),
                    fits.Column(name='yNode', format='F16.8', array=yNode),
                    fits.Column(name='xBar', format='F16.8', array=xBar),
                    fits.Column(name='yBar', format='F16.8', array=yBar),
                    fits.Column(name='sn', format='F16.8', array=sn),
                    fits.Column(name='nPixels', format='i8', array=nPixels),
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
                z = self.em_model[0, i, j] / 2.998e+5
                interp_spec = interp1d(self.restwl / (1. + z), self.data[i, j])
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
        sigma = FWHM_dif / 2.355 / base_cdelt

        for j in range(len(base_spec)):
            ssp = base_spec[j]
            ssp = gaussian_filter(ssp, sigma)
            sspNew, logLam2, velscale = ppxf_util.log_rebin(
                lamRange2, ssp, velscale=velscale)
            # Normalizes templates
            templates[:, j] = sspNew / np.median(sspNew)

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

        for k, h in enumerate(xy):
            i, j = h

            gal_lin = deepcopy(self.data[fw, i, j])
            galaxy, logLam1, velscale = ppxf_util.log_rebin(
                lamRange1, gal_lin)
            normFactor = np.nanmean(galaxy)
            galaxy = galaxy / normFactor

            if np.any(np.isnan(galaxy)):
                pp = cubetools.nanSolution()
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
            h = fits.HDUList()
            h.append(fits.PrimaryHDU(header=hdr))
            h[0].name = ''
            print(h.info())

            # Creates the fitted spectrum extension
            hdr = fits.Header()
            hdr['object'] = ('spectrum', 'Data in this extension')
            hdr['CRPIX3'] = (1, 'Reference pixel for wavelength')
            hdr['CRVAL3'] = (self.restwl[0], 'Reference value for wavelength')
            hdr['CD3_3'] = (np.average(np.diff(self.restwl)), 'CD3_3')
            h.append(
                fits.ImageHDU(data=self.ppxf_spec, header=hdr, name='SCI'))

            # Creates the residual spectrum extension
            hdr = fits.Header()
            hdr['object'] = ('residuals', 'Data in this extension')
            hdr['CRPIX3'] = (1, 'Reference pixel for wavelength')
            hdr['CRVAL3'] = (self.ppxf_wl[0], 'Reference value for wavelength')
            hdr['CD3_3'] = (np.average(np.diff(self.ppxf_wl)), 'CD3_3')
            h.append(
                fits.ImageHDU(
                    data=self.ppxf_spec - self.ppxf_model, header=hdr,
                    name='RES'))

            # Creates the fitted model extension.
            hdr['object'] = 'model'
            h.append(
                fits.ImageHDU(
                    data=self.ppxf_model, header=hdr, name='MODEL'))

            # Creates the solution extension.
            hdr['object'] = 'parameters'
            h.append(
                fits.ImageHDU(
                    data=self.ppxf_sol, header=hdr, name='SOL'))

            # Creates the wavelength extension.
            hdr['object'] = 'wavelength'
            h.append(
                fits.ImageHDU(
                    data=self.ppxf_wl, header=hdr, name='WAVELEN'))

            # Creates the goodpixels extension.
            hdr['object'] = 'goodpixels'
            h.append(
                fits.ImageHDU(
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
