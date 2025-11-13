#!/usr/bin/env python
# THIRD PARTY
from astropy.io import fits

# LOCAL
from src.ifscube import onedspec, parser


if __name__ == '__main__':

    # Approximate redshift of the spectrum
    z = 0.0036

    myspec = onedspec.Spectrum('ngc6300_nuc.fits', redshift=z)

    # Creating a fake variance spectrum with signal-to-noise = 20.
    myspec.variance = (myspec.data / 10) ** 2

    c = parser.LineFitParser('halpha_gauss.cfg')
    linefit_args = c.get_vars()

    x = myspec.linefit(**linefit_args)

    myspec.plotfit()

    myspec.fit_uncertainties()

    print('Flux      : {:.2e}'.format(myspec.em_model[0]))
    print('Flux error: {:.2e}'.format(myspec.flux_err))

    h = fits.open('ngc6300_nuc_linefit.fits')
    h.info()
