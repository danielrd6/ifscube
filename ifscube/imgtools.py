import copy

import numpy as np
from numpy import ma
from scipy.optimize import minimize


def gauss2d(x, y, p):
    """
    Returns a 2-dimensional gaussian function, following the equation

    Parameters
    ----------
    x, y: numpy.ndarray
        The coordinates, or array of coordinates, of the point
        of interest.
    p: list
        Parameters of the gaussian function in the following order:
        [a, b, x0, y0, sx, sy]

    Returns
    -------
    g: numpy.ndarray
        2D Gaussian.

    Notes
    -----
    g(x, y) = a * exp(-(x-x0)**2/(2 * sx**2) - (y-y0)**2/(2 * sy**2))
    """

    amplitude, background, x0, y0, sx, sy, t = p

    a = np.cos(t) ** 2 / 2 / sx ** 2 + np.sin(t) ** 2 / 2 / sy ** 2
    b = -np.sin(2 * t) / 4 / sx ** 2 + np.sin(2 * t) / 4 / sy ** 2
    c = np.sin(t) ** 2 / 2 / sx ** 2 + np.cos(t) ** 2 / 2 / sy ** 2

    g = amplitude * \
        np.exp(
            - (
                    a * (x - x0) ** 2 + 2. * b * (x - x0) * (y - y0)
                    + c * (y - y0) ** 2)
        ) + background

    return g


def match_mosaic(img, method="2dgaussian"):
    """
    Attempts to match different parts of an image mosaic based on
    the assumption of a smooth distribution of surface brightness.

    Parameters
    ----------
    img : numpy.ndarray
        Array containing the mosaic to be matched
    method : string
        Definition of the method to be employed. Possible options are:
            2dgaussian : Fits a 2D gaussian to the image and returns a
                ratio between the fit and the mosaic. This is best
                used when the source is known to be spatially
                unresolved, like the broad line region of an AGN, or a
                star.

    Returns
    -------
    sol : numpy.ndarray
        Array containing the ratio image that best corrects for
        discrepancies in the flux of the mosaic.
    r : minimize.optimize.OptimizeResult
        Optimization result.

    See Also
    --------
    scipy.optimize.minimize

    """

    im = copy.deepcopy(img)

    if method == "2dgaussian":
        scale_factor = np.nanmean(im)
        im /= scale_factor

        p0 = np.zeros(7)
        p0[0] = im[~np.isnan(im)].max()  # gaussian amplitude
        p0[1] = 0  # background level
        p0[2:4] = np.array(np.shape(im)) / 2  # center
        p0[4:6] = p0[2:4] / 2  # sigma
        p0[6] = 0  # position angle

        x, y = np.indices(np.shape(im))

        def res(p):
            residual = np.sum((im[~np.isnan(im)] - gauss2d(x, y, p)[~np.isnan(im)]) ** 2)
            return residual

        r = minimize(res, x0=p0, method='slsqp', options={'disp': True})

        sol = im / gauss2d(x, y, r['x'])

    else:
        raise NotImplementedError('Other methods are not yet implemented.')

    return sol, r


def pixel_distance(x, y, x0=0, y0=0):
    r = np.sqrt(np.square(x - x0) + np.square(y - y0))

    return r


def rebin(arr, xbin, ybin, combine='sum', mask=None):
    """
    Re-samples images or data cubes spatially.

    Parameters
    ----------
    arr : numpy.ndarray
        Input data to be re-sampled.
    xbin, ybin : int
        Re-sampling factor in the horizontal and vertical directions.
    combine : str
        Type of combination. Can be one of
            - sum : add the data in each bin
            - mean : averages the data in each bin
    mask : numpy.ndarray
        Array of the same shape of *arr* in which masked elements are
        evaluated as True.

    Returns
    -------
    new : numpy.ndarray
        Resulting re-sampled image or data cube.
    """

    assert combine in ['sum', 'mean'], 'The combine parameter must be "sum" or "mean".'

    old_shape = copy.copy(arr.shape)
    new_shape = (int(np.ceil(old_shape[0] / ybin)), int(np.ceil(old_shape[1] / xbin)))

    new = np.zeros(new_shape)

    if mask is not None:
        data = ma.array(data=arr, mask=mask)
    else:
        data = ma.array(data=arr)

    def comb_sum(x, m, n):
        return x[m * ybin:(m + 1) * ybin, n * xbin:(n + 1) * xbin].sum()

    def comb_avg(x, m, n):
        return x[m * ybin:(m + 1) * ybin, n * xbin:(n + 1) * xbin].mean()

    comb_fun = dict(sum=comb_sum, mean=comb_avg)

    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            new[i, j] = comb_fun[combine](data, i, j)

    return new


def image2slit(data, x0=None, y0=None, pa=0, slit_width=1):
    """
    Produces a virtual slit image out of a 2D image.

    Parameters
    ----------
    data : numpy.ndarray
        Input image.
    x0, y0 : float
        Center coordinates of the virtual slit.
    pa : float
        Position angle in degrees, with North being zero, and increasing
        towards the east.
    slit_width : float
        Slit width in pixels.

    Returns
    -------
    mask : numpy.ndarray
        Masked used to generate the virtual slit.
    slit_x, slit_y : numpy.ndarray
        Horizontal and vertical coordinates of the slit.
    slit_r : numpy.ndarray
        Distance from the slit center.
    slit_z : numpy.ndarray
        Data values at the virtual slit positions.
    """
    if not isinstance(data, ma.MaskedArray):
        data = ma.masked_invalid(data)

    y, x = [i.astype('float') for i in np.indices(data.shape)]

    if x0 is None:
        x0 = (x.max() - x.min()) / 2.
    if y0 is None:
        y0 = (y.max() - y.min()) / 2.

    x -= x0
    y -= y0

    r = pixel_distance(x, y, x0=0, y0=0)

    im_pa = np.deg2rad(pa - 90)

    slit_r = np.arange(-r.max(), r.max(), 1.)
    slit_x = slit_r * np.cos(im_pa)
    slit_y = slit_r * np.sin(im_pa)

    mask = np.zeros_like(data, dtype='bool')

    slit_z = []
    final_r = []

    for sx, sy, sr in zip(slit_x, slit_y, slit_r):

        cr = pixel_distance(x, y, x0=sx, y0=sy)
        current_mask = cr <= slit_width

        if np.any(current_mask & ~data.mask):
            mask |= current_mask
            final_r += [sr]
            slit_z += [ma.mean(data[current_mask])]

    slit_z = np.array(slit_z)

    return mask, slit_x, slit_y, final_r, slit_z
