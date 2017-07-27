from astropy.modeling import Fittable1DModel, Parameter
import numpy as np


class GaussHermite1D(Fittable1DModel):
    """
    Gauss-Hermite Series up to the fourth order.

    Parameters
    ----------
    amplitude : float
        Integrated flux of the profile.
    mean : float
        Coordinate of the profile center.
    stddev : float
        Sigma
    h3 : float
        Coefficient of the third order element.
    h4 : float
        Coefficient of the fourth order element.

    Description
    -----------
    Taken from Riffel, R. A. 2010 and van der Marel & Franx 1993.
    """

    inputs = ('x',)
    outputs = ('y',)

    amplitude = Parameter()
    mean = Parameter()
    stddev = Parameter()
    h3 = Parameter()
    h4 = Parameter()

    @staticmethod
    def evaluate(x, amplitude, mean, stddev, h3, h4):

        w = (x - mean) / stddev

        alphag = np.exp(-w**2. / 2.) / np.sqrt(2. * np.pi)
        hh3 = h3 * np.sqrt(2.) / np.sqrt(6.) * (2. * w**3 - 3. * w)
        hh4 = h4 / np.sqrt(24.) * (4. * w**4 - 12. * w**2 + 3.)

        return amplitude * alphag / stddev * (1. + hh3 + hh4)

    # @staticmethod
    # def fit_deriv(x, a, x0, s, h3, h4):
    #
    #     # Amplitude derivative

    #     ag = np.exp(-((x-x0)/s)**2. / 2.) / np.sqrt(2. / np.pi)

    #     hh3 = (h3 * ((2**(3 / 2) * (x - x0)**3) / s**3
    #            - (3 * np.sqrt(2) * (x - x0)) / s)) /\
    #         (np.sqrt(6) * np.sqrt(np.pi)

    #     hh4 = (h4 * (3 - (12 * (x - x0)**2) / s**2
    #            + (4 * (x - x0)**4) / s**4)) / (2 * np.sqrt(6))

    #     d_a = ag / s * (1 + hh4 + hh3)

    #     # Mean derivative

    #     d_x0 = (
    #         a * (1 + (
    #             h4 * (3 - (12 * (x - x0)**2) / s**2
    #             + (4 * (x - x0)**4) / s**4)) /
    #             (2 * np.sqrt(6)) + (
    #                 h3 * ((2**(3 / 2) * (x - x0)**3) / s**3
    #                 - (3 * np.sqrt(2) * (x - x0)) / s)) /
    #             (np.sqrt(6) * np.sqrt(np.pi)))
    #         * (x - x0) * np.exp(-(x - x0)**2 / (2 * s**2))) /\
    #         (np.sqrt(2) * np.sqrt(np.pi) * s**3) +\
    #         (
    #             a * (
    #                 (
    #                     h3 * ((3 * np.sqrt(2)) / s - (3 * 2**(3 / 2)
    #                     * (x - x0)**2) / s**3))
    #             / (np.sqrt(6) * np.sqrt(np.pi)) +
    #             (h4 * ((24 * (x - x0)) / s**2 - \
    #              (16 * (x - x0)**3) / s**4)) /\
    #             (2 * np.sqrt(6))) * np.exp(-(x - x0)**2 / (2 * s**2))) /\
    #         (np.sqrt(2) * np.sqrt(np.pi) * s)

    #

    #     return [d_a, d_x0]
