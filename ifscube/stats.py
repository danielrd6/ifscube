#
# Statistics subroutines for profile fitting
#
import numpy as np


def line_flux_error(flux, fwhm, delta_l, peak, s_peak):
    """
    Evaluates the uncertainty in the flux of the fitted line.
    Based on Lenz & Ayres 1992 and Wesson 2015

    Parameters
    ----------
    flux : number
      Integrated flux in the line.
    fwhm : number
      Full width at half maximum.
    delta_l : number
      Sampling step in wavelength, usually in angstroms per pixel.
    peak : number
      The peak value of the flux.
    s_peak : number
      Uncertainty in the peak flux.


    Returns
    -------
    s_flux : number
      Uncertainty in the integrated flux of the line.
    """

    cx = 0.67

    s_flux = flux / (cx * np.sqrt(fwhm / delta_l) * (peak / s_peak))

    return s_flux
