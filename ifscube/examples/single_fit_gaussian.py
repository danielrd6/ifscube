import ifscube.onedspec as ds
import numpy as np

# Definition of line centers

lines_wl = np.array([
    6548.04,  # [N II] 6548
    6562.80,  # H alpha
    6583.46,  # [N II] 6583
])

# Approximate redshift of the spectrum
z = 0.0036

# Defining the initial guess for the parameters.

ncomponents = 3
npars = 3  # five parameters for the gauss hermite polynomials
           # flux, wavelength, sigma, h3, h4

p0 = np.zeros(ncomponents * npars)

p0[0::npars] = 1e-14                # flux
p0[1::npars] = lines_wl * (1. + z)  # wavelength
p0[2::npars] = 3.0                  # sigma

# Setting bounds

b = []
for i in range(ncomponents):
    b += [[0, 1e-12]]                                                  # flux
    b += [[lines_wl[i] * (1. + z) - 10, lines_wl[i] * (1. + z) + 10]]  # wl
    b += [[1.5, 9]]                                                    # sigma

# Setting the constraints

c = [
    # Keeping the same doppler shift on all lines
    {'type': 'eq', 'fun': lambda x: x[1]/lines_wl[0] - x[4]/lines_wl[1]},
    {'type': 'eq', 'fun': lambda x: x[4]/lines_wl[1] - x[7]/lines_wl[2]},

    # And the same goes for the sigmas
    {'type': 'eq', 'fun': lambda x: x[2]/x[1] - x[5]/x[4]},
    {'type': 'eq', 'fun': lambda x: x[5]/x[4] - x[8]/x[7]},
]

# Loading the spectrum
myspec = ds.Spectrum('ngc6300_nuc.fits')

# Creating a fake variance spectrum with signal-to-noise = 20.
myspec = ds.Spectrum('ngc6300_nuc.fits')
myspec.variance = (myspec.data / 10) ** 2

x = myspec.linefit(
    p0, fitting_window=(6500, 6700), function='gaussian',
    constraints=c, bounds=b, fit_continuum=True,)

myspec.plotfit()

myspec.fit_uncertainties()

print('Flux      : {:.2e}'.format(myspec.em_model[0]))
print('Flux error: {:.2e}'.format(myspec.flux_err))
