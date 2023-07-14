import warnings
from copy import deepcopy
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from numpy import ma
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, center_of_mass

from . import channel_maps
from . import cubetools, spectools, onedspec


class Cube(onedspec.Spectrum):
    """
    A class for dealing with IFS data cubes.
    """

    def __init__(self, fname: str = None, scidata: Union[str, int] = 'SCI', variance: Union[str, int] = None,
                 flags: Union[str, int] = None, stellar: Union[str, int] = None, primary: Union[str, int] = 'PRIMARY',
                 redshift: float = None, wcs_axis: int = 3, spatial_mask: Union[np.ndarray, str] = None,
                 nan_spaxels: str = 'all', wavelength: Union[str, int] = None):
        """
        Instantiates the class. If any arguments are given they will be
        passed to the _load method.
        """
        super().__init__(fname=fname, scidata=scidata, variance=variance, flags=flags, stellar=stellar, primary=primary,
                         redshift=redshift, wcs_axis=wcs_axis, wavelength=wavelength)

        if spatial_mask is None:
            self.spatial_mask = np.zeros(self.data.shape[1:]).astype(bool)
        else:
            self.spatial_mask = spatial_mask
            if nan_spaxels == 'all':
                self.nan_mask = np.isnan(self.data).all(axis=0) | np.isnan(self.stellar).all(axis=0)
            elif nan_spaxels == 'any':
                self.nan_mask = np.isnan(self.data).any(axis=0) | np.isnan(self.stellar).any(axis=0)
            else:
                self.nan_mask = np.zeros(self.data.shape[1:]).astype('bool')
            self.spatial_mask |= self.nan_mask

        try:
            if self.header['VORBIN']:
                vortab = fits.getdata(fname, 'VOR')
                self.voronoi_tab = vortab
                self.binned = True
        except KeyError:
            self.binned = False

        self.binned_cube = None
        self.noise = None
        self.signal = None

        if fname is not None:
            self._set_spec_indices()

    @property
    def spatial_mask(self):
        return self._spatial_mask

    @spatial_mask.setter
    def spatial_mask(self, value: Union[np.ndarray, str]):
        if isinstance(value, str):
            if ',' in value:
                ext_name = value.split(',')[0].strip()
                ext_version = int(value.split(',')[1])
            else:
                ext_name = value.strip()
                ext_version = None
            self._spatial_mask = fits.getdata(self.fitsfile, ext_name, ext_version)
        elif isinstance(value, int):
            self._spatial_mask = fits.getdata(self.fitsfile, value).astype('bool')
        else:
            self._spatial_mask = value.astype('bool')
        assert self._spatial_mask.shape == self.data.shape[1:],\
            'Spatial mask must match the last two dimensions of the data cube.'

    def _set_spec_indices(self):
        self.spec_indices = np.column_stack([
            np.ravel(np.indices(np.shape(self.data)[1:])[0][~self.spatial_mask]),
            np.ravel(np.indices(np.shape(self.data)[1:])[1][~self.spatial_mask]),
        ])

    def _arg2cube(self, arg, cube):

        if len(np.shape(arg)) == 0:
            cube *= arg
        elif len(np.shape(arg)) == 1:
            for i, j in self.spec_indices:
                cube[:, i, j] = arg
        elif len(np.shape(arg)) == 2:
            for i, j in enumerate(cube):
                cube[i] = arg

        return cube

    def _write_eqw(self, eqw, args):

        outimage = args['outimage']
        # Basic tests and first header

        hdr = fits.Header()

        # Creates MEF output.
        h = fits.HDUList()
        h.append(fits.PrimaryHDU(header=self.header))
        h[0].name = 'PRIMARY'

        # Creates the model equivalent width extension
        hdr['object'] = ('eqw_model', 'EqW based on emission model.')
        hdr['sigwidth'] = (args['sigma_factor'], 'Line width in units of sigma.')
        hdr['bunit'] = ('angstrom', 'Unit of pixel values.')
        hdr['l_idx'] = (args['component'], 'Line number in fit output')

        if hasattr(self, 'component_names'):
            hdr['l_name'] = (self.component_names[args['component']], 'Line name')

        if args['windows'] is not None:
            hdr['c_blue0'] = (args['windows'][0], 'lower limit, blue continuum section')
            hdr['c_blue1'] = (args['windows'][1], 'upper limit, blue continuum section')
            hdr['c_red0'] = (args['windows'][2], 'lower limit, red continuum section')
            hdr['c_red1'] = (args['windows'][3], 'upper limit, red continuum section')

        h.append(fits.ImageHDU(data=eqw[0], header=hdr, name='EQW_M'))

        # Creates the direct equivalent width extension
        hdr['object'] = ('eqw_direct', 'EqW measured directly on the spectrum.')
        hdr['sigwidth'] = (args['sigma_factor'], 'Line width in units of sigma.')
        h.append(fits.ImageHDU(data=eqw[1], header=hdr, name='EQW_D'))

        # Creates the minimize's exit status extension
        hdr['object'] = 'fit_status'
        h.append(fits.ImageHDU(data=self.fit_status, header=hdr, name='STATUS'))

        h.writeto(outimage)

    def _masked_to_nan(self):
        for c, j in np.ndenumerate(self.spatial_mask):
            if j:
                self.em_model[:, c[0], c[1]] = np.nan

    def write(self, file_name: str, overwrite=False):

        hdr = deepcopy(self.header_data)
        hdr['REDSHIFT'] = 0.0

        # Creates MEF output.
        h = fits.HDUList()
        hdu = fits.PrimaryHDU(header=self.header)
        hdu.header['REDSHIFT'] = 0.0
        hdu.name = 'PRIMARY'
        h.append(hdu)

        hdr = fits.Header()
        hdr['CRPIX3'] = (1, 'Reference pixel for wavelength')
        hdr['CRVAL3'] = (self.rest_wavelength[0], 'Reference value for wavelength')
        hdr['CD3_3'] = (np.average(np.diff(self.rest_wavelength)), 'CD3_3')
        hdu = fits.ImageHDU(data=self.data, header=hdr)
        hdu.name = 'SCI'
        h.append(hdu)

        hdu = fits.ImageHDU(data=self.variance, header=hdr)
        hdu.name = 'VAR'
        h.append(hdu)

        hdu = fits.ImageHDU(data=self.spatial_mask.astype(int), header=hdr)
        hdu.name = 'MASK2D'
        h.append(hdu)

        if hasattr(self, 'flags'):
            hdu = fits.ImageHDU(data=self.flags.astype(int), header=hdr)
            hdu.name = 'FLAGS'
            h.append(hdu)

        if hasattr(self, 'stellar'):
            # noinspection PyTypeChecker
            hdu = fits.ImageHDU(data=self.stellar, header=hdr)
            hdu.name = 'STELLAR'
            h.append(hdu)

        if self.ppxf_sol is not None:
            # noinspection PyTypeChecker
            hdu = fits.ImageHDU(data=self.ppxf_sol, header=hdr)
            hdu.name = 'PPXFSOL'
            h.append(hdu)

        with fits.open(self.fitsfile) as original_cube:
            for ext_name in ['vor', 'vorplus']:
                if ext_name in original_cube:
                    h.append(original_cube[ext_name])

            h.writeto(file_name, overwrite=overwrite)

    def continuum(self, write_fits=False, output_image=None, fitting_window=None, continuum_options=None):
        """
        Evaluates a polynomial continuum for the whole cube and stores
        it in self.cont.

        Parameters
        ----------
        write_fits : bool
            Write the output in a FITS file
        output_image : str
            Name of the output FITS file.
        fitting_window : list
            A list containing the starting and ending wavelengths.
        continuum_options : dict
            Dictionary of continuum fitting options.
        """

        if self.binned:
            v = self.voronoi_tab
            xy = np.column_stack(
                [v[np.unique(v['binNum'], return_index=True)[1]][coords] for coords in ['xcoords', 'ycoords']])
        else:
            v = None
            xy = self.spec_indices

        fw = fitting_window
        fwidx = (self.rest_wavelength > fw[0]) & (self.rest_wavelength < fw[1])

        wl = deepcopy(self.rest_wavelength[fwidx])
        data = deepcopy(self.data[fwidx])

        c = np.zeros_like(data)

        if continuum_options is None:
            continuum_options = {'degree': 3, 'upper_threshold': 2, 'lower_threshold': 2, 'n_iterate': 5}

        try:
            continuum_options['output']
        except KeyError:
            continuum_options['output'] = 'function'

        for k, h in enumerate(xy):
            i, j = h
            s = deepcopy(data[:, i, j])
            if (any(s[:20]) and any(s[-20:])) or (any(np.isnan(s[:20])) and any(np.isnan(s[-20:]))):
                try:
                    cont = spectools.continuum(wl, s, **continuum_options)
                    if v is not None:
                        for l, m in v[v[:, 2] == k, :2]:
                            c[:, l, m] = cont[1]
                    else:
                        c[:, i, j] = cont[1]
                except TypeError:
                    print('Could not find a solution for {:d},{:d}.'.format(i, j))
                    c[:, i, j] = np.nan
                except ValueError:
                    c[:, i, j] = np.nan
            else:
                c[:, i, j] = np.nan

        self.cont = c

        if write_fits:
            if output_image is None:
                output_image = self.fitsfile.replace('.fits', '_continuum.fits')

            hdr = deepcopy(self.header_data)

            try:
                hdr['REDSHIFT'] = self.redshift
            except KeyError:
                hdr['REDSHIFT'] = (self.redshift, 'Redshift used in GMOSDC')

            hdr['CRVAL3'] = wl[0]
            hdr['CONTDEGR'] = (continuum_options['degree'], 'Degree of continuum polynomial')
            hdr['CONTNITE'] = (continuum_options['n_iterate'], 'Continuum rejection iterations')
            hdr['CONTLTR'] = (continuum_options['lower_threshold'], 'Continuum lower threshold')
            hdr['CONTHTR'] = (continuum_options['upper_threshold'], 'Continuum upper threshold')

            fits.writeto(output_image, data=c, header=hdr)

        return c

    def snr_eval(self, wl_range=(6050, 6200), continuum_options=None):
        """Measures the signal to noise ratio (SNR) for each spectrum in a data cube, returning an image of the SNR.

        This method evaluates the SNR for each spectrum in a data cube by measuring the residuals
        of a polynomial continuum fit. The function CONTINUUM of the SPECTOOLS package is used to
        provide the continuum, with zero rejection iterations and a 3 order polynomial.

        Parameters
        -----------
        wl_range : array like
            An array like object containing two wavelength coordinates
            that define the SNR window at the rest frame.
        continuum_options : dictionary
            Options for the continuum fitting function.

        Returns
        --------
        snr : numpy.ndarray
            Image of the SNR for each spectrum.
        """

        snr_window = (self.rest_wavelength >= wl_range[0]) & (self.rest_wavelength <= wl_range[1])

        # FIXME: This is only here because I am always setting
        # a variance attribute, when it really shouldn't be.
        # The correct behaviour should be to check if variance is set.
        # if hasattr(self, 'variance'):
        if not np.all(self.variance == 1.):
            noise = np.nanmean(np.sqrt(self.variance[snr_window, :, :]), axis=0)
            signal = np.nanmean(self.data[snr_window, :, :], axis=0)

        else:
            noise = np.zeros(np.shape(self.data)[1:])
            signal = np.zeros(np.shape(self.data)[1:])
            data = deepcopy(self.data)

            wl = self.rest_wavelength[snr_window]

            if continuum_options is None:
                continuum_options = {
                    'n_iterate': 0, 'degree': 1, 'upper_threshold': 3, 'lower_threshold': 3, 'output': 'function'}
            else:
                continuum_options['output'] = 'function'

            for i, j in self.spec_indices:
                if any(data[snr_window, i, j]) and all(~np.isnan(data[snr_window, i, j])):
                    s = data[snr_window, i, j]
                    cont = spectools.continuum(wl, s, **continuum_options)[1]
                    noise[i, j] = np.nanstd(s - cont)
                    signal[i, j] = np.nanmean(cont)
                else:
                    noise[i, j], signal[i, j] = np.nan, np.nan

        signal[signal == 0.0] = np.nan
        noise[(noise == 0.0) | (noise == 1.0)] = np.nan

        self.noise = noise
        self.signal = signal

        return np.array([signal, noise])

    def wlprojection(self, wl0, fwhm, filtertype='box', writefits=False,
                     outimage='outimage.fits'):
        """Writes a projection of the data cube along the wavelength coordinate.

        Parameters
        ----------
        wl0: float
            Central wavelength at the rest frame.
        fwhm: float
            Full width at half maximum. See 'filtertype'.
        filtertype: string
            Type of function to be multiplied by the spectrum to return
            the argument for the integral. Should be one of

                - 'box': Box function that is zero everywhere and 1 between :math:`\\lambda_0 \\pm {\\rm FWHM}/2`
                - 'gaussian': Normalized gaussian function with center at
                    :math:`\\lambda_0` and :math:`\\sigma = {\\rm FWHM}/(2\\sqrt{2\\log(2)})`

        writefits: bool
            Writes the output to a FITS file.
        outimage : string
            Name of the output image

        Returns
        -------
        Nothing.
        """

        output_image = cubetools.wlprojection(
            arr=self.data, wl=self.rest_wavelength, wl0=wl0, fwhm=fwhm, filtertype=filtertype)

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

            fits.writeto(outimage, data=output_image, header=hdr)

        return output_image

    def aperture_spectrum(self, radius=1.0, x0=None, y0=None, flag_threshold=0.5, combine: str = 'sum'):
        """
        Makes an aperture spectrum out of the data cube.

        Parameters
        ----------
        radius : float
            Radius of the virtual aperture in pixels.
        x0, y0 : float or None
            Central coordinates of the aperture in pixels. If both are
            set to *None* the center of the datacube will be used.
        flag_threshold : float
            Amount of flagged pixels for the output spectrum to also
            be flagged in this pixel.
        combine : str
            Combination type. Can be either 'sum', 'mean' or 'median'.

        Returns
        -------
        s : ifscube.onedspec.Spectrum
            Spectrum object.
        """

        if x0 is None:
            x0 = int(self.spec_indices[:, 1].mean())
        if y0 is None:
            y0 = int(self.spec_indices[:, 0].mean())

        sci, npix_sci = cubetools.aperture_spectrum(
            self.data, x0=x0, y0=y0, radius=radius, combine=combine)
        var, npix_var = cubetools.aperture_spectrum(
            self.variance, x0=x0, y0=y0, radius=radius, combine='sum')
        if np.all(self.variance == 1.0):
            var = self.variance[:, 0, 0]
        ste, npix_ste = cubetools.aperture_spectrum(self.stellar, x0=x0, y0=y0, radius=radius, combine=combine)
        fla, npix_fla = cubetools.aperture_spectrum(
            (self.flags.astype('bool')).astype('float64'), x0=x0, y0=y0, radius=radius, combine='mean')

        # NOTE: This only makes sense when the flags are only ones
        # and zeros, that is why the flag combination has to ensure
        # the boolean character of the flags.
        fla = fla > flag_threshold

        s = onedspec.Spectrum()
        s.data = sci
        s.variance = var
        s.stellar = ste
        s.flags = fla

        keys = ['wl', 'rest_wavelength', 'redshift', 'header']

        for i in keys:
            s.__dict__[i] = self.__dict__[i]

        return s

    def plotspec(self, x, y, show_noise: bool = False, noise_smooth: float = 0.0, ax: plt.Axes = None):
        """
        Plots the spectrum at coordinates x,y.

        Parameters
        ----------
        x,y : numbers or iterables
            If x and y are numbers plots the spectrum at the specific
            spaxel. If x and y are two element tuples plots the average
            between x[0],y[0] and x[1],y[1]
        show_noise: bool
            Displays the noise spectrum as a filled area.
        noise_smooth: float
            Sigma of the gaussian kernel for the noise smoothing.
        ax: matplotlib.pyplot.Axes, optional
            Axes instance in which to plot the spectrum. If *None* a new
            instance will be created.

        Returns
        -------
        Nothing.
        """

        # fig = plt.figure(1)
        if ax is None:
            fig = plt.figure(1)
            ax = fig.add_subplot(111)

        if hasattr(x, '__iter__') and hasattr(y, '__iter__'):
            s = np.average(np.average(self.data[:, y[0]:y[1], x[0]:x[1]], 1), 1)
        elif hasattr(x, '__iter__') and not hasattr(y, '__iter__'):
            s = np.average(self.data[:, y, x[0]:x[1]], 1)
        elif not hasattr(x, '__iter__') and hasattr(y, '__iter__'):
            s = np.average(self.data[:, y[0]:y[1], x], 1)
        else:
            s = self.data[:, y, x]

        if hasattr(x, '__iter__') and hasattr(y, '__iter__'):
            syn = np.average(np.average(self.stellar[:, y[0]:y[1], x[0]:x[1]], 1), 1)
        elif hasattr(x, '__iter__') and not hasattr(y, '__iter__'):
            syn = np.average(self.stellar[:, y, x[0]:x[1]], 1)
        elif not hasattr(x, '__iter__') and hasattr(y, '__iter__'):
            syn = np.average(self.stellar[:, y[0]:y[1], x], 1)
        else:
            syn = self.stellar[:, y, x]

        ax.plot(self.rest_wavelength, s)
        ax.plot(self.rest_wavelength, syn)

        if show_noise and (self.noise_cube is not None):

            if hasattr(x, '__iter__') and hasattr(y, '__iter__'):
                n = np.average(np.average(self.noise_cube[:, y[0]:y[1], x[0]:x[1]], 1), 1)
            elif hasattr(x, '__iter__') and not hasattr(y, '__iter__'):
                n = np.average(self.noise_cube[:, y, x[0]:x[1]], 1)
            elif not hasattr(x, '__iter__') and hasattr(y, '__iter__'):
                n = np.average(self.noise_cube[:, y[0]:y[1], x], 1)
            else:
                n = self.noise_cube[:, y, x]

            n = gaussian_filter(n, noise_smooth)
            sg = gaussian_filter(s, noise_smooth)

            ax.fill_between(self.rest_wavelength, sg - n, sg + n, edgecolor='', alpha=0.2, color='green')

        if hasattr(self, 'flags') and not hasattr(x, '__iter__') and not hasattr(y, '__iter__'):
            sflags = self.flags[:, y, x].astype('bool')
            ax.scatter(self.rest_wavelength[sflags], s[sflags], marker='x', color='red')

        plt.show()

    def channel_maps(self, *args, **kwargs):

        channel_maps.channelmaps(self, *args, **kwargs)

    def peak_coords(self, wl_center, wl_width, center_type='peak_cen', spatial_center=None, spatial_width=10):
        """
        Returns the coordinates of the centroid or the peak of a 2D
        flux distrubution.

        Parameters
        ----------
        wl_center : number
            The central wavelenght of the spectral region over which the
            cube is to be inegrated.
        wl_width : number
            The wavelenght width of the spectral region over which the cube
            is to be integrated.
        center_type : string
            Type of centering algorithm emploied. Options are: centroid,
            peak and peak_cen.
            Where,
            'peak_cen' returns the centroid on a box 'spatial_width' wide,
            centered on the pixel corresponding to the peak value,
            'centroid' returns position of the centroid of the values in
            the region, and,
            'peak' returns the pixel position of the maximum value in the
            region.
        spatial_center: tuple
            Central position of the spatial region where the center is
            calculated.
        spatial_width : number
            Side size of the square spatial region where the center is
            calculated.
        """
        if spatial_center is None:
            spatial_center = [
                int(self.data.shape[1] / 2.), int(self.data.shape[2] / 2.)]

        # wavelength = self.wl
        projection = cubetools.wlprojection(
            arr=self.data, wl=self.wl, wl0=wl_center, fwhm=wl_width,
            filtertype='box')
        projection_crop = projection[
                          int(spatial_center[0] - spatial_width / 2):
                          int(spatial_center[0] + spatial_width / 2) + 1,
                          int(spatial_center[1] - spatial_width / 2):
                          int(spatial_center[1] + spatial_width / 2) + 1]
        if center_type == 'peak_cen':
            idx = np.nanargmax(projection_crop, axis=None)
            coords = np.unravel_index(idx, projection_crop.shape)
            spatial_center[0] = int(
                spatial_center[0] - spatial_width / 2 + coords[0])
            spatial_center[1] = int(
                spatial_center[1] - spatial_width / 2 + coords[1])
            projection_crop = projection[
                              int(spatial_center[0] - spatial_width / 2):
                              int(spatial_center[0] + spatial_width / 2) + 1,
                              int(spatial_center[1] - spatial_width / 2):
                              int(spatial_center[1] + spatial_width / 2) + 1
                              ]
            coords = center_of_mass(ma.masked_invalid(projection_crop))
        elif center_type == 'peak':
            idx = np.nanargmax(projection_crop, axis=None)
            coords = np.unravel_index(idx, projection_crop.shape)
        elif center_type == 'centroid':
            pass
            coords = center_of_mass(ma.masked_invalid(projection_crop))
        else:
            raise ValueError('Parameter center_type "{:s}" not understood.'.format(center_type))

        coords = coords + np.array([
            int(spatial_center[0] - spatial_width / 2),
            int(spatial_center[1] - spatial_width / 2)])

        return coords

    def spatial_rebin(self, xbin, ybin, combine='mean'):
        """
        Spatial undersampling of the datacube.

        Parameters
        ----------
        xbin: int
            Size of the bin in the horizontal direction.
        ybin: int
            Size of the bin in the vertical direction.
        combine: str
            Type of spectral combination.
                - 'mean': The spectral flux is averaged over the spatial bin.
                - 'sum': The spectral flux is summed over the spatial bin.

        Returns
        -------
        None.
        """

        m = self.flags.astype('bool')

        self.data = cubetools.rebin(self.data, xbin, ybin, combine=combine, mask=m)
        self.stellar = cubetools.rebin(self.stellar, xbin, ybin, combine=combine, mask=m)
        self.variance = cubetools.rebin(self.flags, xbin, ybin, combine='mean')
        self.flags = (cubetools.rebin(self.flags, xbin, ybin, combine='sum') == xbin * ybin).astype('int')

        if hasattr('self', 'noise_cube'):
            self.noise_cube = np.sqrt(cubetools.rebin(np.square(self.noise_cube), xbin, ybin, combine='sum', mask=m))

            if combine == 'mean':
                self.noise_cube /= self.ncubes

        self.spatial_mask = np.zeros(self.data.shape[1:], dtype=bool)
        self._set_spec_indices()

        return

    def gaussian_smooth(self, sigma=2, write_fits=False, outfile=None):
        """
        Performs a spatial gaussian convolution on the data cube.

        Parameters
        ----------
        sigma: float
            Sigma of the gaussian kernel.
        write_fits: bool
            Writes the output to a FITS file.
        outfile: str
            Name of the output file.

        Returns
        -------
        gdata: numpy.ndarray
            Gaussian smoothed data.
        gvar: numpy.ndarray
            Smoothed variance.

        See also
        --------
        scipy.ndimage.gaussian_filter
        """

        if write_fits and outfile is None:
            raise RuntimeError('Output file name not given.')

        gdata = np.zeros_like(self.data)
        gvar = np.zeros_like(self.noise_cube)

        i = 0

        while i < len(self.wl):
            tmp_data = cubetools.nan_to_nearest(self.data[i])
            # noinspection PyTypeChecker
            tmp_var = cubetools.nan_to_nearest(self.noise_cube[i]) ** 2

            gdata[i] = gaussian_filter(tmp_data, sigma)
            gvar[i] = np.sqrt(gaussian_filter(tmp_var, sigma))

            i += 1

        if write_fits:
            hdulist = fits.open(self.fitsfile)
            hdr = hdulist[0].header

            hdr['SPSMOOTH'] = ('Gaussian', 'Type of spatial smoothing.')
            hdr['GSMTHSIG'] = (sigma, 'Sigma of the gaussian kernel')

            hdulist[self.extension_names['scidata']].data = gdata
            hdulist[self.extension_names['variance']].data = gvar

            hdulist.writeto(outfile)

        return gdata, gvar

    def voronoi_binning(self, target_snr=10.0, write_fits=False, outfile=None, overwrite=False, plot=False,
                        flag_threshold=0.5, **kwargs):
        """
        Applies Voronoi binning to the data cube, using Cappellari's Python implementation.

        Parameters
        ----------
        target_snr : float
            Desired signal to noise ratio of the binned pixels
        write_fits : boolean
            Writes a FITS image with the output of the binning.
        plot: bool
            Plots the binning results.
        outfile : string
            Name of the output FITS file. If 'None' then the name of
            the original FITS file containing the data cube will be used
            as a root name, with '.bin' appended to it.
        overwrite : boolean
            Overwrites files with the same name given in 'outfile'.
        flag_threshold : float
            Bins with less than this fraction of unflagged pixels will be flagged.
        **kwargs: dict
            Arguments passed to voronoi_2d_binning.

        Returns
        -------
        Nothing.

        Notes
        -----
        The output file contains two tables which outline the tesselation process. These are
        stored in the extensions 'VOR' and 'VORPLUS'.
        """

        try:
            from vorbin.voronoi_2d_binning import voronoi_2d_binning
        except ImportError:
            raise ImportError('Could not find the voronoi_2d_binning module. Please add it to your PYTHONPATH.')

        if self.noise is None:
            raise RuntimeError('This function requires prior execution of the snr_eval method.')

        # Initializing the binned arrays as zeros.
        assert hasattr(self, 'data'), 'Could not access the data attribute of the Cube object.'
        b_data = ma.zeros(self.data.shape)
        b_data.mask = self.flags.astype(bool)

        assert hasattr(self, 'variance'), 'Could not access the variance attribute of the Cube object.'
        b_variance = ma.zeros(self.variance.shape)
        b_variance.mask = self.flags.astype(bool)

        assert hasattr(self, 'flags'), 'Could not access the variance attribute of the Cube object.'
        b_flags = np.zeros_like(self.flags, dtype=int)

        valid_spaxels = np.ravel(~np.isnan(self.signal) & ~np.isnan(self.noise) & ~self.spatial_mask)

        x = np.ravel(np.indices(np.shape(self.signal))[1])[valid_spaxels]
        y = np.ravel(np.indices(np.shape(self.signal))[0])[valid_spaxels]

        s, n = deepcopy(self.signal), deepcopy(self.noise)

        s[s <= 0] = np.average(self.signal[self.signal > 0])
        n[n <= 0] = np.average(self.signal[self.signal > 0]) * .5

        signal, noise = np.ravel(s)[valid_spaxels], np.ravel(n)[valid_spaxels]

        bin_num, x_node, y_node, x_bar, y_bar, sn, n_pixels, scale = \
            voronoi_2d_binning(x, y, signal, noise, target_snr, plot=plot, quiet=0, **kwargs)
        v = np.column_stack([y, x, bin_num])

        # For every nan in the original cube, fill with nan the binned cubes.
        nan_idx = (Ellipsis,
                   np.ravel(np.indices(np.shape(self.signal))[0])[~valid_spaxels],
                   np.ravel(np.indices(np.shape(self.signal))[1])[~valid_spaxels])
        b_data[nan_idx] = np.nan
        b_variance[nan_idx] = np.nan
        b_flags[nan_idx] = 1

        for i in np.arange(bin_num.max() + 1):
            same_bin = v[:, 2] == i
            same_bin_coordinates = v[same_bin, :2]

            for k in same_bin_coordinates:
                binned_idx = (Ellipsis, k[0], k[1])
                unbinned_idx = (Ellipsis, same_bin_coordinates[:, 0], same_bin_coordinates[:, 1])

                b_data[binned_idx] = ma.mean(self.data[unbinned_idx], axis=1)
                b_variance[binned_idx] = ma.mean(self.variance[unbinned_idx], axis=1)
                b_flags[binned_idx] = (np.mean(self.flags[unbinned_idx], axis=1) >= flag_threshold).astype(int)

        b_data = b_data.data
        b_variance = b_variance.data

        if write_fits:

            h = fits.HDUList()
            hdu = fits.PrimaryHDU(header=self.header)
            hdu.name = 'PRIMARY'
            hdu.header['VORBIN'] = (True, 'Processed by Voronoi binning?')
            hdu.header['VORTSNR'] = (target_snr, 'Target SNR for Voronoi binning.')
            h.append(hdu)

            hdr = self.header_data
            # noinspection PyTypeChecker
            hdu = fits.ImageHDU(data=b_data, header=hdr)
            hdu.name = 'SCI'
            h.append(hdu)

            # noinspection PyTypeChecker
            hdu = fits.ImageHDU(data=b_variance, header=hdr)
            hdu.name = 'VAR'
            h.append(hdu)

            # noinspection PyTypeChecker
            hdu = fits.ImageHDU(data=b_flags, header=hdr)
            hdu.name = 'FLAGS'
            h.append(hdu)

            tbhdu = fits.BinTableHDU.from_columns(
                [
                    fits.Column(name='xcoords', format='i8', array=x),
                    fits.Column(name='ycoords', format='i8', array=y),
                    fits.Column(name='binNum', format='i8', array=bin_num),
                ], name='VOR')

            tbhdu_plus = fits.BinTableHDU.from_columns(
                [
                    fits.Column(name='ubin', format='i8', array=np.unique(bin_num)),
                    fits.Column(name='xNode', format='F16.8', array=x_node),
                    fits.Column(name='yNode', format='F16.8', array=y_node),
                    fits.Column(name='xBar', format='F16.8', array=x_bar),
                    fits.Column(name='yBar', format='F16.8', array=y_bar),
                    fits.Column(name='sn', format='F16.8', array=sn),
                    fits.Column(name='nPixels', format='i8', array=n_pixels),
                ], name='VORPLUS')

            h.append(tbhdu)
            h.append(tbhdu_plus)

            if outfile is None:
                outfile = self.fitsfile.replace('.fits', '_vor.fits')

            h.writeto(outfile, overwrite=overwrite)

        self.binned_cube = b_data

    def write_binnedspec(self, doppler_correction=False):
        """
        Writes only one spectrum for each bin in a FITS file.
        
        Parameters
        ----------
        doppler_correction: bool
            Apply Doppler correction.
        """

        xy = self.spec_indices
        unique_indices = xy[np.unique(self.data[1400, :, :], return_index=True)[1]]

        if doppler_correction:

            assert hasattr(self, 'em_model'), 'This function requires the Cube.em_model attribute to be defined.'

            specs = np.array([])
            for k, i, j in enumerate(unique_indices):
                z = self.em_model[0, i, j] / 2.99792458e+5
                interp_spec = interp1d(self.rest_wavelength / (1. + z), self.data[i, j])
                specs = np.row_stack([specs, interp_spec(self.rest_wavelength)])

        else:
            specs = np.row_stack([self.data[:, i, j] for i, j in unique_indices])

        return specs

    @staticmethod
    def lineflux(amplitude, sigma):
        """
        Calculates the flux in a line given the amplitude and sigma
        of the gaussian function that fits it.
        """

        lf = amplitude * abs(sigma) * np.sqrt(2. * np.pi)

        return lf
