from ifscube import gmos
import numpy as np

# Definition of line centers

lines_wl = np.array([
    6548.04,  # [N II] 6548
    6562.80,  # H alpha
    6583.46,  # [N II] 6583
])

# Approximate redshift of the spectrum
z = 0.008

# Defining the initial guess for the parameters.

ncomponents = 3
npars = 3  # three parameters for each gaussian component
           # flux, wavelength, sigma

p0 = np.zeros(ncomponents * npars)

p0[0::npars] = 1e-15                # flux
p0[1::npars] = lines_wl * (1. + z)  # wavelength
p0[2::npars] = 2.0                  # sigma

# Setting bounds

b = []
for i in range(ncomponents):
    b += [[1e-17, 1e-14]]                                            # flux
    b += [[lines_wl[i] * (1. + z) - 5, lines_wl[i] * (1. + z) + 5]]  # wl
    b += [[1.0, 4]]                                                  # sigma

# Setting the constraints

c = [
    # Keeping the same doppler shift on all lines
    {'type': 'eq', 'fun': lambda x: x[1]/lines_wl[0] - x[4]/lines_wl[1]},
    {'type': 'eq', 'fun': lambda x: x[4]/lines_wl[1] - x[7]/lines_wl[2]},

    # And the same goes for the sigmas
    {'type': 'eq', 'fun': lambda x: x[2]/x[1] - x[5]/x[4]},
    {'type': 'eq', 'fun': lambda x: x[5]/x[4] - x[8]/x[7]},
]

# Loading the spectrum.
mycube = gmos.cube('ngc3081_cube.fits', var_ext=2)

# Index of a good spectrum.
idx = (3, 3)

# Performs the fit and stores the results in the variable "x".
x = mycube.linefit(
    p0,
    fitting_window=(6500, 6700),
    function='gaussian',
    constraints=c,
    bounds=b,
    writefits=True,
    outimage='ngc3081_fit.fits',
    refit=True,
    spiral_loop=True,
    spiral_center=idx,
    minopts=dict(eps=1e-3),
    fit_continuum=True,
)

# Plots the fit for the spaxel defined in "idx".
mycube.plotfit(*idx)
