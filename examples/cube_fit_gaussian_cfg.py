#!/usr/bin/env python
from src.ifscube import Cube, parser
from src.ifscube.cubetools import append_config

if __name__ == '__main__':

    z = 0.0081

    my_cube = Cube('ngc3081_cube.fits', redshift=z)

    # Loads the configuration file and stores the parameters
    # in the line_fit_args dict.
    c = parser.LineFitParser('halpha_cube.cfg')
    line_fit_args = c.get_vars()

    my_cube.linefit(**line_fit_args)

    # Append the input configuration file to the output of the fit.
    append_config(config_file='halpha_cube.cfg', fit_file='ngc3081_cube_linefit.fits')

    # Plots the fit for the spaxel defined in "idx".
    my_cube.plotfit(3, 3)
