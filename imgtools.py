import numpy as np
from copy import deepcopy
from scipy.optimize import minimize

def gauss2d(x, y, p):
    """
    Returns a 2-dimensional gaussian function, following the equation
  
                          (x-x0)**2   (y-y0)**2
    g(x,y) = a * exp( - ( --------- + --------- ) ) + b
                           2*sx**2     2*sy**2
  
    Parameters:
    -----------
    As described in the formula above.
  
    Returns:
    --------
    x,y : nd.array
      The coordinates, or array of coordinates, of the point
      of interest.
    p : list
      Parameters of the gaussian function in the following order:
      [a,b,x0,y0,sx,sy]
    """
  
    A, B, x0, y0, sx, sy, t = p
  
    a = np.cos(t)**2/2/sx**2 + np.sin(t)**2/2/sy**2
    b = -np.sin(2*t)/4/sx**2 + np.sin(2*t)/4/sy**2
    c = np.sin(t)**2/2/sx**2 + np.cos(t)**2/2/sy**2
  
    g = A*np.exp(-( a*(x-x0)**2 + 2.*b*(x-x0)*(y-y0) + c*(y-y0)**2 )) + B

    return g

def match_mosaic(img, method="2dgaussian"):
    """
    Attempts to match diferent parts of an image mosaic based on
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

    """

    im = deepcopy(img)
    sol = im*0.0

    if method == "2dgaussian":
        scale_factor = np.nanmean(im)
        im /= scale_factor

        p0 = np.zeros(7)
        p0[0] = im[~np.isnan(im)].max()     # gaussian amplitude
        p0[1] = 0                           # background level
        p0[2:4] = np.array(np.shape(im))/2  # center
        p0[4:6] = p0[2:4]/2                 # sigma
        p0[6] = 0                           # position angle

        x, y = np.indices(np.shape(im))

        res = lambda p : np.sum( (im[~np.isnan(im)]
            - gauss2d(x, y, p)[~np.isnan(im)])**2 )
        r = minimize(res, x0=p0, method='slsqp', options={'disp':True})

        sol = im/gauss2d(x, y, r['x'])

    return sol, r


