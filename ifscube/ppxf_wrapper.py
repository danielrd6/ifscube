# This module is a wrapper for the ppxf package available in PyPI
import copy
import glob

import matplotlib.pyplot as plt
import numpy as np
from astropy import wcs, constants
from astropy.io import fits
from pkg_resources import resource_filename
from ppxf import ppxf, ppxf_util
from scipy.ndimage import gaussian_filter


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
            noise=0.05, plot_fit=False, quiet=False, deg=4, moments=4):
        """
        Performs the pPXF fit.
        
        Parameters
        ----------
        wavelength : numpy.ndarray
            Wavelength coordinates of the data.
        data : numpy.ndarray
            Input spectrum flux vector.

        Returns
        -------
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
        if isinstance(noise, float):
            noise = np.zeros(len(galaxy), dtype=galaxy.dtype) + noise
        elif isinstance(noise, np.ndarray):
            noise, log_lam1, velscale = ppxf_util.log_rebin(lam_range1, copy.deepcopy(noise)[fw])
        self.noise = noise

        self.normalization_factor = np.nanmean(galaxy)
        galaxy = galaxy / self.normalization_factor
        noise = np.abs(noise / self.normalization_factor)

        pp = ppxf.ppxf(
            templates, galaxy, noise, velscale, start, goodpixels=gp, moments=moments, degree=deg,
            vsyst=dv, quiet=quiet,
        )

        self.solution = pp

        if plot_fit:
            self.plot_fit()

        return pp

    def plot_fit(self):

        fig = plt.figure(1)
        plt.clf()
        ax = fig.add_subplot(111)

        gp = self.good_pixels

        if self.mask is not None:
            for region in self.mask:
                ax.axvspan(region[0], region[1], color='grey', alpha=0.1)

        ax.plot(self.obs_wavelength, self.solution.galaxy)
        ax.plot(self.obs_wavelength, self.solution.bestfit)

        ax.set_xlabel(r'Wavelength')
        ax.set_ylabel(r'Normalized density')

        ax.set_ylim(self.solution.bestfit.min() * 0.8, self.solution.bestfit.max() * 1.2)

        print('Velocity: {:.2f}\nSigma: {:.2f}\nh3: {:.2f}\nh4: {:.2f}'.format(*self.solution.sol))

        plt.show()
