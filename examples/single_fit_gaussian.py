#!/usr/bin/env python
# THIRD PARTY
import numpy as np
from astropy.io import fits

# LOCAL
from src import ifscube as ds

if __name__ == '__main__':

    # Definition of line centers

    lines_wl = np.array([
        6548.04,  # [N II] 6548
        6562.80,  # H alpha
        6583.46,  # [N II] 6583
    ])

    # Approximate redshift of the spectrum
    z = 0.0036

    # Defining the initial guess for the parameters.
    # five parameters for the gauss hermite polynomials
    # amplitude, velocity, sigma, h3, h4

    ncomponents = 3
    npars = 3
    p0 = np.zeros(ncomponents * npars)

    p0[0::npars] = 1e-14                # flux
    p0[1::npars] = lines_wl * (1. + z)  # wavelength
    p0[2::npars] = 3.0                  # sigma

    # Setting bounds

    b = []
    for i in range(ncomponents):
        # amplitude
        b += [[0.0, 1e-12]]
        # velocity
        b += [[-300.0, +300.0]]
        # sigma
        b += [[40.0, 500.0]]

    # Setting the constraints

    c = [
        # Keeping the same doppler shift on all lines
        {'type': 'eq', 'fun': lambda x: x[1] - x[4]},
        {'type': 'eq', 'fun': lambda x: x[4] - x[7]},

        # And the same goes for the sigmas
        {'type': 'eq', 'fun': lambda x: x[2] - x[5]},
        {'type': 'eq', 'fun': lambda x: x[5] - x[8]},
    ]

    # Creating a fake variance spectrum with signal-to-noise = 20.
    myspec = ds.Spectrum('ngc6300_nuc.fits')
    myspec.variance = (myspec.data / 10) ** 2

    x = myspec.linefit(
        p0, fitting_window=(6500.0, 6700.0), feature_wl=lines_wl, function='gaussian', constraints=c, bounds=b,
        fit_continuum=True, write_fits=True, overwrite=True)

    myspec.plotfit()

    myspec.fit_uncertainties()

    print('Flux      : {:.2e}'.format(myspec.em_model[0]))
    print('Flux error: {:.2e}'.format(myspec.flux_err))

    h = fits.open('ngc6300_nuc_linefit.fits')
    h.info()
