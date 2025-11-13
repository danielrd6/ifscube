from astropy.modeling import Fittable1DModel, Fittable2DModel, Parameter
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

    amplitude = Parameter(default=1)
    mean = Parameter(default=0)
    stddev = Parameter(default=1)
    h3 = Parameter(default=0)
    h4 = Parameter(default=0)

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

class DiskRotation(Fittable2DModel):

    """
    Observed radial velocity according to Bertola et al 1991.

    Notes
    -----
    This model returns a two-dimensional velocity field of observed
    radial velocities based on a rotation curve equal to

    v_c(r) = A * r / ( (r ** 2 + c_0 ** 2) ** (p / 2) )

    This equation was taken from Bertola et al. 1991 (ApJ, 373, 369B).
    """

    # inputs = ('x', 'y',)
    n_outputs = 1
    n_inputs = 2

    amplitude = Parameter(default=100.0)
    c_0 = Parameter(default=1.0)
    p = Parameter(default=1.25)
    phi_0 = Parameter(default=0.7854)
    theta = Parameter(default=0.7854)
    v_sys = Parameter(default=0.0)
    x_0 = Parameter(default=0.0)
    y_0 = Parameter(default=0.0)

    @staticmethod
    def evaluate(x, y, amplitude, c_0, p, phi_0, theta, v_sys, x_0, y_0):
        """
        Parameters
        ----------
        amplitude: float
            Integrated flux of the profile.
        c_0: float
            Concentration index.
        p: float
            Velocity curve exponent.
        phi_0: float
            Position angle of the line of nodes in radians.
        theta: float
            Inclination of the disk in radians(theta = 0 for a face-on disk).

        Returns
        -------
        v: numpy.ndarray
            Velocity at coordinates x, y.
        """

        r = np.sqrt((x - x_0) ** 2 + (y - y_0) ** 2)
        phi = np.arctan2((y - y_0), (x - x_0)) + (np.pi / 2.0)

        cop = np.cos(phi - phi_0)
        sip = np.sin(phi - phi_0)
        cot = np.cos(theta)
        sit = np.sin(theta)
        
        d = amplitude * r * cop * sit * cot**p
        n = r**2 * (sip**2 + cot**2 * cop**2) + c_0**2 * cot**2
        v = v_sys + d / n**(p / 2)

        return v
