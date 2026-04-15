from ifscube import Cube
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

# The number of components is the number of spectral features to be fit.
ncomponents = 3
# Three parameters for each gaussian component: amplitude, velocity, sigma:
npars = 3

p0 = np.zeros(ncomponents * npars)

p0[0::npars] = 1e-15                # flux
p0[1::npars] = 0.0  # velocity
p0[2::npars] = 120.0                  # sigma

# Setting bounds

b = []
for i in range(ncomponents):
    b += [[1e-17, 1e-14]]   # amplitude
    b += [[-500.0, 500.0]]  # velocity
    b += [[40.0, 500.0]]    # sigma

# Setting the constraints

c = [
    # Keeping the same velocity on all lines.
    {'type': 'eq', 'fun': lambda x: x[1] - x[4]},
    {'type': 'eq', 'fun': lambda x: x[4] - x[7]},

    # And the same goes for the velocity dispersions (sigmas).
    {'type': 'eq', 'fun': lambda x: x[2] - x[5]},
    {'type': 'eq', 'fun': lambda x: x[5] - x[8]},
]

# Loading the spectrum.
mycube = Cube('ngc3081_cube.fits')

# Index of a good spectrum.
idx = (3, 3)

# Performs the fit and stores the results in the variable "x".
fit = mycube.linefit(
    p0,
    feature_wl=lines_wl,
    fitting_window=(6500, 6700),
    function='gaussian',
    constraints=c,
    bounds=b,
    write_fits=True,
    out_image='ngc3081_fit.fits',
    refit=True,
    spiral_loop=True,
    spiral_center=idx,
    minopts=dict(eps=1e-3),
    fit_continuum=True,
    overwrite=True,
)

# Plots the fit for the spaxel defined in "idx".
mycube.plotfit(*idx)
