#
# Statistics subroutines for profile fitting
#
import numpy as np


def line_flux_error(flux, fwhm, sampling_interval, amplitude, amplitude_error):
    """
    Evaluates the uncertainty in the flux of the fitted line.
    Based on Lenz & Ayres 1992 and Wesson 2015

    Parameters
    ----------
    flux : float
      Integrated flux in the line.
    fwhm : float
      Full width at half maximum.
    sampling_interval : float
      Sampling step in wavelength, usually in angstroms per pixel.
    amplitude : float
      The peak value of the flux.
    amplitude_error : float
      Uncertainty in the peak flux.


    Returns
    -------
    flux_error : number
      Uncertainty in the integrated flux of the line.
    """

    # FIXME: What is this?
    cx = 0.67

    flux_error = flux / (cx * np.sqrt(fwhm / sampling_interval) * (amplitude / amplitude_error))

    return flux_error
