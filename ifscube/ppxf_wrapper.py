# This module is a wrapper for the ppxf package available in PyPI
import copy
import glob

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from astropy import wcs, constants
from astropy.io import fits
from numpy import ma
from pkg_resources import resource_filename
from ppxf import ppxf, ppxf_util
from scipy.ndimage import gaussian_filter

from . import spectools
from .onedspec import Spectrum


def spectrum_kinematics(spectrum, fitting_window=None, **kwargs):
    """
    Executes pPXF fitting of the stellar spectrum over the whole
    data cube.

    Parameters
    ----------
    spectrum : onedspec.Spectrum
        Spectrum object.
    fitting_window: tuple or list
        Initial and final values of wavelength for fitting.
    **kwargs
        Passed directly to ppxf_wrapper.Fit.fit.

    Returns
    -------
    output : onedspec.Spectrum
        Output spectrum instance.


    Notes
    -----
    This function is merely a wrapper for Michelle Capellari's
    pPXF Python algorithm for penalized pixel fitting of stellar
    spectra.

    See also
    --------
    ppxf
    """

    if fitting_window is None:
        fitting_window = [spectrum.rest_wavelength[i] for i in [0, -1]]

    fit = Fit(fitting_window=fitting_window)

    fw = (spectrum.rest_wavelength >= fitting_window[0]) & (spectrum.rest_wavelength <= fitting_window[1])
    spectrum_length = np.sum(fw)

    if 'mask' in kwargs:
        mask = copy.deepcopy(kwargs['mask'])
        kwargs.pop('mask')
    else:
        mask = None

    if ('noise' not in kwargs) and (spectrum.variance is not None):
        kwargs['noise'] = np.sqrt(spectrum.variance)

    if (mask is not None) and ((spectrum.flags is not None) and np.any(spectrum.flags)):
        m = mask + spectools.flags_to_mask(spectrum.rest_wavelength, spectrum.flags)
    elif spectrum.flags is not None:
        m = spectools.flags_to_mask(spectrum.rest_wavelength, spectrum.flags)
    else:
        m = None

    if np.sum(spectrum.flags[fw]) / spectrum_length > 0.5:
        raise RuntimeError('Skipping spectrum due to too many (> 0.5) flagged pixels.')

    pp = fit.fit(spectrum.rest_wavelength, spectrum.data, mask=m, **kwargs)

    output = Spectrum()
    output.sol = pp.sol
    output.data = np.interp(
        spectrum.rest_wavelength[fw], fit.obs_wavelength, pp.galaxy * fit.normalization_factor)
    output.stellar = np.interp(
        spectrum.rest_wavelength[fw], fit.obs_wavelength, pp.bestfit * fit.normalization_factor)

    output.rest_wavelength = spectrum.rest_wavelength[fw]
    output.flags = spectrum.flags[fw]
    output.variance = spectrum.variance[fw]

    output.header = copy.deepcopy(spectrum.header)

    if hasattr(output, 'header_data'):
        output.header_data = copy.deepcopy(spectrum.header_data)
        output.header_data['CRPIX3'] = (1, 'Reference pixel for wavelength')
        output.header_data['CRVAL3'] = (spectrum.rest_wavelength[fw][0], 'Reference value for wavelength')
        output.header_data['CD3_3'] = (np.mean(np.diff(spectrum.rest_wavelength)), 'CD3_3')
    else:
        w = wcs.WCS(naxis=1)
        w.wcs.crpix = np.array([1.])
        w.wcs.crval = np.array([spectrum.rest_wavelength[fw][0]])
        w.wcs.cdelt = np.array([np.mean(np.diff(spectrum.rest_wavelength))])
        output.header_data = w.to_header()

    return output


def cube_kinematics(cube, fitting_window, individual_spec=None, verbose=False, **kwargs):
    """
    Executes pPXF fitting of the stellar spectrum over the whole
    data cube.

    Parameters
    ----------
    cube : datacube.Cube
        Input data cube.
    fitting_window: tuple
        Initial and final values of wavelength for fitting.
    individual_spec : tuple
        (x, y) coordinates of the individual spaxel to be fit.
    verbose : bool
        Prints progress messages.
    **kwargs
        Arguments passed directly to ppxf_wrapper.Fit.fit.

    Returns
    -------
    Nothing

    Notes
    -----
    This function is merely a wrapper for Michelle Capellari's
    pPXF Python algorithm for penalized pixel fitting of stellar
    spectra.

    See also
    --------
    ppxf
    """

    vor = None
    if cube.binned:
        vor = cube.voronoi_tab
        xy = np.column_stack(
            [vor[coords][np.unique(vor['binNum'], return_index=True)[1]] for coords in ['ycoords', 'xcoords']])
    else:
        xy = cube.spec_indices

    if individual_spec is not None:
        xy = [individual_spec[::-1]]

    fit = Fit(fitting_window=fitting_window)

    fw = (cube.rest_wavelength >= fitting_window[0]) & (cube.rest_wavelength <= fitting_window[1])
    wavelength = cube.rest_wavelength[fw]

    sol = np.zeros((4, np.shape(cube.data)[1], np.shape(cube.data)[2]), dtype='float64')
    data = cube.data[fw]
    model = np.zeros_like(data)
    noise = np.sqrt(cube.variance[fw])
    flags = cube.flags[fw]

    if 'mask' in kwargs:
        mask = copy.deepcopy(kwargs['mask'])
        kwargs.pop('mask')
    else:
        mask = None

    if individual_spec is None:
        summed_flags = flags.sum(axis=0)
        spatial_mask = np.array([summed_flags[i[0], i[1]] / len(wavelength) > 0.2 for i in xy])
        xy = xy[~spatial_mask]

    if verbose:
        # noinspection PyTypeChecker
        iterator = tqdm.tqdm(xy, desc='pPXF fitting.', unit='spectrum')
    else:
        # noinspection PyTypeChecker
        iterator = xy

    for h in iterator:
        i, j = h

        pop_it_later = False
        if ('noise' not in kwargs) and ~np.all(noise[:, i, j] == 1.0):
            kwargs['noise'] = noise[:, i, j]
            pop_it_later = True

        if (mask is not None) and ((cube.flags is not None) and np.any(cube.flags[:, i, j])):
            m = mask + spectools.flags_to_mask(wavelength, flags[:, i, j])
        elif cube.flags is not None:
            m = spectools.flags_to_mask(wavelength, flags[:, i, j])
            if not m:
                m = None
        else:
            m = None

        pp = fit.fit(wavelength, data[:, i, j], mask=m, **kwargs)
        if len(pp.sol) < 4:
            pp.sol = np.concatenate([pp.sol, (4 - len(pp.sol)) * [0.,]])

        if pop_it_later:
            kwargs.pop('noise')

        if vor is not None:

            bin_num = vor[(vor['xcoords'] == j) & (vor['ycoords'] == i)]['binNum']
            same_bin_num = vor['binNum'] == bin_num
            same_bin_x = vor['xcoords'][same_bin_num]
            same_bin_y = vor['ycoords'][same_bin_num]

            for l, m in np.column_stack([same_bin_y, same_bin_x]):
                sol[:, l, m] = pp.sol
                data[:, l, m] = np.interp(wavelength, fit.obs_wavelength, pp.galaxy * fit.normalization_factor)
                model[:, l, m] = np.interp(wavelength, fit.obs_wavelength, pp.bestfit * fit.normalization_factor)

        else:
            sol[:, i, j] = pp.sol
            data[:, i, j] = np.interp(wavelength, fit.obs_wavelength, pp.galaxy * fit.normalization_factor)
            model[:, i, j] = np.interp(wavelength, fit.obs_wavelength, pp.bestfit * fit.normalization_factor)

    output = copy.deepcopy(cube)

    output.data = data
    output.stellar = model
    output.flags = flags
    output.variance = np.square(noise)
    output.ppxf_sol = sol
    output.rest_wavelength = wavelength

    return output


class Fit(object):

    def __init__(self, fitting_window, cushion=100.0):

        self.fitting_window = fitting_window
        self.mask = None

        self.base = np.array([])
        self.base_wavelength = np.array([])
        self.obs_wavelength = np.array([])
        self.obs_flux = np.array([])
        self.noise = np.array([])
        self.flags = np.array([])
        self.good_pixels = np.array([])
        self.solution = None

        self.base_delta = 1.0
        self.normalization_factor = 1.0

        self.load_miles_models()

        self._cut_base(self.fitting_window[0], self.fitting_window[1], cushion=cushion)

    def load_miles_models(self):

        path = resource_filename('ppxf', 'miles_models')
        base_files = glob.glob(path + '/*.fits')
        base_files.sort()

        w = wcs.WCS(base_files[0], naxis=1)
        spectrum = fits.getdata(base_files[0])
        wavelength = w.wcs_pix2world(np.arange(spectrum.size), 0)[0]

        base = spectrum.reshape((1, spectrum.size))

        for file in base_files:
            base = np.row_stack([base, fits.getdata(file)])

        self.base = base
        self.base_wavelength = wavelength

        d = np.diff(self.base_wavelength)
        if np.std(d) > (1.e-6 * d.mean()):
            raise UserWarning('Base is not evenly sampled in wavelength.')

        self.base_delta = np.mean(np.diff(self.base_wavelength))

    def _cut_base(self, start_wavelength, end_wavelength, cushion=100.0):

        base_cut = (self.base_wavelength > start_wavelength - cushion) \
                   & (self.base_wavelength < end_wavelength + cushion)

        if not np.any(base_cut):
            raise RuntimeError(
                'The interval defined by fitting_window lies outside the range covered by base_wl. Please review your'
                'base and/or fitting window.')

        self.base = self.base[:, base_cut]
        self.base_wavelength = self.base_wavelength[base_cut]

    def fit(self, wavelength, data, mask=None, initial_velocity=0.0, initial_sigma=150.0, fwhm_gal=2, fwhm_model=1.8,
            noise=0.05, plot_fit=False, quiet=False, deg=4, moments=4, **kwargs):
        """
        Performs the pPXF fit.
        
        Parameters
        ----------
        wavelength : numpy.ndarray
            Wavelength coordinates of the data.
        data : numpy.ndarray
            Input spectrum flux vector.
        mask : list
            List of masked regions, as pairs of wavelength coordinates.
        initial_velocity : float
            Initial guess for radial velocity.
        initial_sigma : float
            Initial guess for the velocity dispersion.
        fwhm_gal : float
            Full width at half maximum of a resolution element in the observed spectrum in units of pixels.
        fwhm_model : float
            The same as the above for the models.
        noise : float or numpy.ndarray
            If float it as assumed as the signal to noise ratio, and will be horizontally applied to the whole
            spectrum. If it is an array, it will be interpreted as individual noise values for each pixel.
        plot_fit : bool
            Plots the resulting fit.
        quiet : bool
            Prints information on the fit.
        deg : int
            Degree of polynomial function to be fit in addition to the stellar population spectrum.
        moments : int
            Number of moments in the Gauss-Hermite polynomial. A simple Gaussian would be 2.
        kwargs
            Additional keyword arguments passed directly to ppxf.

        Returns
        -------
        pp
            pPXF output object.

        See Also
        --------
        ppxf, ppxf_util
        """

        self.mask = mask

        fw = (wavelength >= self.fitting_window[0]) & (wavelength < self.fitting_window[1])

        lam_range1 = wavelength[fw][[0, -1]]
        gal_lin = copy.deepcopy(data[fw])

        self.obs_flux = gal_lin

        galaxy, log_lam1, velscale = ppxf_util.log_rebin(lam_range1, gal_lin)

        # Here we use the goodpixels as the fitting window
        gp = np.arange(len(log_lam1))
        lam1 = np.exp(log_lam1)
        self.obs_wavelength = lam1

        if self.mask is not None:
            if len(self.mask) == 1:
                gp = gp[(lam1 < self.mask[0][0]) | (lam1 > self.mask[0][1])]
            else:
                m = np.array([(lam1 < i[0]) | (lam1 > i[1]) for i in self.mask])
                gp = gp[np.sum(m, 0) == m.shape[0]]

        self.good_pixels = gp

        lam_range2 = self.base_wavelength[[0, -1]]
        ssp = self.base[0]

        ssp_new, log_lam2, velscale = ppxf_util.log_rebin(lam_range2, ssp, velscale=velscale)
        templates = np.empty((ssp_new.size, len(self.base)))

        fwhm_dif = np.sqrt(fwhm_gal ** 2 - fwhm_model ** 2)
        # Sigma difference in pixels
        sigma = fwhm_dif / 2.355 / self.base_delta

        for j in range(len(self.base)):
            ssp = self.base[j]
            ssp = gaussian_filter(ssp, sigma)
            ssp_new, log_lam2, velscale = ppxf_util.log_rebin(lam_range2, ssp, velscale=velscale)
            # Normalizes templates
            templates[:, j] = ssp_new / np.median(ssp_new)

        c = constants.c.value * 1.e-3
        dv = (log_lam2[0] - log_lam1[0]) * c  # km/s
        # z = np.exp(vel/c) - 1

        # Here the actual fit starts.
        start = [initial_velocity, initial_sigma]  # (km/s), starting guess for [V,sigma]

        # Assumes uniform noise accross the spectrum
        def make_noise(galaxy, noise):
            noise = galaxy * noise
            noise_mask = (~np.isfinite(noise)) | (noise <= 0.0)
            mean_noise = np.mean(noise[~noise_mask])
            noise[noise_mask] = mean_noise
            return noise
        if isinstance(noise, float):
            noise = make_noise(galaxy, noise)
        elif isinstance(noise, np.ndarray):
            noise, log_lam1, velscale = ppxf_util.log_rebin(lam_range1, copy.deepcopy(noise)[fw])
        self.noise = noise

        self.normalization_factor = np.nanmean(galaxy)
        galaxy = copy.deepcopy(ma.getdata(galaxy / self.normalization_factor))
        noise = copy.deepcopy(ma.getdata(np.abs(noise / self.normalization_factor)))

        assert np.all((noise > 0) & np.isfinite(noise)), 'Invalid values encountered in noise spectrum.'
        pp = ppxf.ppxf(templates, galaxy, noise, velscale, start, goodpixels=gp, moments=moments, degree=deg, vsyst=dv,
                       quiet=quiet, **kwargs)

        self.solution = pp

        if plot_fit:
            self.plot_fit()

        return pp

    def plot_fit(self):

        fig = plt.figure(1)
        plt.clf()
        ax = fig.add_subplot(111)

        if self.mask is not None:
            for region in self.mask:
                ax.axvspan(region[0], region[1], color='grey', alpha=0.1)

        ax.plot(self.obs_wavelength, self.solution.galaxy)
        ax.plot(self.obs_wavelength, self.solution.bestfit)

        ax.set_xlabel(r'Wavelength')
        ax.set_ylabel(r'Normalized flux')

        ax.set_ylim(self.solution.bestfit.min() * 0.8, self.solution.bestfit.max() * 1.2)

        if len(self.solution.sol) == 4:
            print('Velocity: {:.2f}\nSigma: {:.2f}\nh3: {:.2f}\nh4: {:.2f}'.format(*self.solution.sol))
        elif len(self.solution.sol) == 2:
            print('Velocity: {:.2f}\nSigma: {:.2f}'.format(*self.solution.sol))

        plt.show()
