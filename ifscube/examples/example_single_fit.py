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

# Loading the spectrum
myspec = ds.Spectrum('ngc6300_nuc.fits')

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

myspec.linefit(
    p0, fitting_window=(6500, 6700), function='gauss_hermite')

myspec.plotfit()
