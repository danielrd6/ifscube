import numpy as np
import astropy.io.fits as pf
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter as gf


def bertola(r,phi,x):
    eta = np.sin(phi - x[2]) ** 2 + np.cos(x[3]) ** 2 *\
        np.cos(phi - x[2]) ** 2
    v = x[0] +\
        (
            x[1] * r * np.cos(phi - x[2]) *
            np.sin(x[3]) * np.cos(x[3]) ** x[4]) /\
        (r ** 2 * eta + x[5] ** 2 * np.cos(x[3]) ** 2) ** (x[4] ** 2.)
    return v


def bertola_freecenter(x):

    r = np.sqrt( (vx-x[6])**2 + (vy-x[7])**2 )
    phi = arctan2((vy-x[7]), -(vx-x[6]))
    eta = np.sin(phi-x[2])**2 + np.cos(x[3])**2*np.cos(phi-x[2])**2
    v = x[0] + ( x[1] * r * np.cos(phi-x[2]) * np.sin(x[3]) * np.cos(x[3])**x[4] )\
        / ( r**2 * eta + x[5]**2 * np.cos(x[3])**2)**(x[4]**2.)
    return v, r, phi


def rotationfit():
    
    # new rscale = 576.928  # parsec/arcsec
    pxscale = 0.1        # arcsec/pixel
    rscale = 0.587         # kiloparsec/arcsec
    
    data = pf.getdata('am2306-721r4.snr10.harvel.fits') + 0.029751*2.99792e+5
    #data = pf.getdata('slits.rvel.fits')
    
    
    y, x = indices(shape(data))
    x0, y0 = 33, 23.5
    rproj = sqrt( (x-x0)**2 + (y-y0)**2 )
    # This projection of the phi angle has the zero at the WCS north,
    # or the negative x in the image, and increases towards the WCS east,
    # or the positive y in the image.
    phi = arctan2((y-y0), -(x-x0))
    phi[phi < 0] += 2*pi
    
    vcoords = loadtxt('voronoi_binning_snr10.dat')
    bins = reshape(vcoords[:,2], shape(x))
    vx, vy = zeros(shape(x)), zeros(shape(y))
    for i in unique(vcoords[:,2]):
        vx[bins == i] = average(x[bins == i])
        vy[bins == i] = average(y[bins == i])
    vrproj = sqrt( (vx-x0)**2 + (vy-y0)**2 )
    vphi = arctan2((vy-y0), -(vx-x0))
    res = lambda x : sum( ( data - bertola_freecenter(x)[0] )**2 \
        / variance )
    
    p0 = [9000, 60, np.deg2rad(64.), np.deg2rad(-57.), .7, 1, 33, 22]
    b = [
        [8000, 10000],
        [0, 400],
        [np.deg2rad(50.), np.deg2rad(90.)],
        [np.deg2rad(-57.5), np.deg2rad(-56.5)],
        [.6, .8],
        [0, 30],
        [20, 40],
        [20, 40]]

    r = minimize( res, x0=p0, method='SLSQP', options=opts, bounds=b,
        constraints=const)

    return r
