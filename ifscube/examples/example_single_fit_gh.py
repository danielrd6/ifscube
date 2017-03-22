import ifscube.onedspec as ds
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

ncomponents = 3
npars = 5  # five parameters for the gauss hermite polynomials
           # flux, wavelength, sigma, h3, h4

p0 = np.zeros(ncomponents * npars)

p0[0::5] = 1e-14                # flux
p0[1::5] = lines_wl * (1. + z)  # wavelength
p0[2::5] = 3.0                  # sigma
p0[3::5] = 0.0                  # h3
p0[4::5] = 0.0                  # h4

# Setting bounds

b = []
for i in range(ncomponents):
    b += [[0, 1e-12]]                                                  # flux
    b += [[lines_wl[i] * (1. + z) - 10, lines_wl[i] * (1. + z) + 10]]  # wl
    b += [[1.5, 9]]                                                    # sigma
    b += [[-.2, +.2]]                                                  # h3
    b += [[-.2, +.2]]                                                  # h4

# Setting the constraints

c = [
    # Keeping the same doppler shift on all lines
    {'type': 'eq', 'fun': lambda x: x[1]/lines_wl[0] - x[6]/lines_wl[1]},
    {'type': 'eq', 'fun': lambda x: x[6]/lines_wl[1] - x[11]/lines_wl[2]},

    # And the same goes for the sigmas
    {'type': 'eq', 'fun': lambda x: x[2]/x[1] - x[7]/x[6]},
    {'type': 'eq', 'fun': lambda x: x[7]/x[6] - x[12]/x[11]},

    # All the same h3's and h4's
    {'type': 'eq', 'fun': lambda x: x[3] - x[8]},
    {'type': 'eq', 'fun': lambda x: x[8] - x[13]},
    {'type': 'eq', 'fun': lambda x: x[4] - x[9]},
    {'type': 'eq', 'fun': lambda x: x[9] - x[14]},
]

# Loading the spectrum
myspec = ds.Spectrum('ngc6300_nuc.fits')

myspec.linefit(
    p0, fitting_window=(6500, 6700), function='gauss_hermite',
    constraints=c, bounds=b)

myspec.plotfit()
