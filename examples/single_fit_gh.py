from src import ifscube as ds
import numpy as np

# Definition of line centers

lines_wl = np.array([
    6548.04,  # [N II] 6548
    6562.80,  # H alpha
    6583.46,  # [N II] 6583
])

# Approximate redshift of the spectrum
z = 0.0037

# Defining the initial guess for the parameters.

# The number of components is the number of spectral features to be fit.
ncomponents = 3
# Five parameters for the gauss hermite polynomial amplitude, velocity, sigma, h3, h4
npars = 5

p0 = np.zeros(ncomponents * npars)

p0[0::5] = 1e-14    # amplitude
p0[1::5] = 0.0      # velocity
p0[2::5] = 100.0    # sigma
p0[3::5] = 0.0      # h3
p0[4::5] = 0.0      # h4

# Setting bounds

b = []
for i in range(ncomponents):
    b += [[0, 1e-12]]        # flux
    b += [[-500.0, +500.0]]  # wl
    b += [[40.0, 500.0]]     # sigma
    b += [[-.2, +.2]]        # h3
    b += [[-.2, +.2]]        # h4

# Setting the constraints

c = [
    # Keeping the same doppler shift on all lines
    {'type': 'eq', 'fun': lambda x: x[1] - x[6]},
    {'type': 'eq', 'fun': lambda x: x[6] - x[11]},

    # And the same goes for the sigmas
    {'type': 'eq', 'fun': lambda x: x[2] - x[7]},
    {'type': 'eq', 'fun': lambda x: x[7] - x[12]},

    # All the same h3's and h4's
    {'type': 'eq', 'fun': lambda x: x[3] - x[8]},
    {'type': 'eq', 'fun': lambda x: x[8] - x[13]},
    {'type': 'eq', 'fun': lambda x: x[4] - x[9]},
    {'type': 'eq', 'fun': lambda x: x[9] - x[14]},
]

# Loading the spectrum
myspec = ds.Spectrum('ngc6300_nuc.fits')

myspec.linefit(
    p0, fitting_window=(6500.0, 6700.0), feature_wl=lines_wl, function='gauss_hermite', constraints=c, bounds=b,
    fit_continuum=True)

myspec.plotfit()
