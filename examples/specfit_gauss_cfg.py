#!/usr/bin/env python
# local
from src.ifscube import parser
from src.ifscube import Cube

if __name__ == '__main__':
    z = 0.0081

    mycube = Cube('ngc3081_cube.fits', redshift=z, var_ext=None, ncubes_ext=None)

    c = parser.LineFitParser('halpha_cube.cfg')
    linefit_args = c.get_vars()

    mycube.linefit(**linefit_args)

    # Plots the fit for the spaxel defined in "idx".
    mycube.plotfit(3, 3)
