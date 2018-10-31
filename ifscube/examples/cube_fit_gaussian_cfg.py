#!/usr/bin/env python
# THIRD PARTY
from astropy.io import fits

# LOCAL
from ifscube import gmos, parser

if __name__ == '__main__':

    z = 0.0081

    mycube = gmos.cube(
        'ngc3081_cube.fits', redshift=z, var_ext=None, ncubes_ext=None)

    c = parser.LineFitParser('halpha_cube.cfg')
    linefit_args = c.get_vars()

    mycube.linefit(**linefit_args)

    # Plots the fit for the spaxel defined in "idx".
    mycube.plotfit(3, 3)
