#!/usr/bin/env python
# stdlib
import glob
import site

# third party
from astropy.io import fits
import numpy as np

# local
from ifscube import Cube, spectools


def load_models(searchPattern):

    baseFiles = glob.glob(searchPattern)
    baseFiles.sort()

    wl = spectools.get_wl(baseFiles[0], dwlkey='cdelt1')
    spec = fits.getdata(baseFiles[0])
    base = spec.reshape((1, len(spec)))

    for fileName in baseFiles[1:]:
        base = np.row_stack([base, fits.getdata(fileName)])

    return wl, base


if __name__ == '__main__':

    #
    # Voronoi binning
    #

    # Loads the datacube.
    mycube = Cube('ngc3081_cube.fits', variance='ERR', scidata='SCI')

    # Evaluates signal to noise ratio in a given wavelength window.
    mycube.snr_eval(wl_range=(5650, 5750))

    # Actual binning.
    mycube.voronoi_binning(
        target_snr=3, write_fits=True, overwrite=True, plot=False)

    #
    # pPXF fitting
    #

    # Loads base files for pPXF.
    #
    # ATTENTION: You must modify this path to match your local copy
    # of the miles models. If you have installed the pPXF python
    # package via pip, everything from site-packages on should be
    # the same.
    wl, base = load_models(
        site.getsitepackages()[0] + '/ppxf/miles_models/Mun*fits')

    # Loads the Voronoi binned cube.
    mycube = Cube('ngc3081_cube.bin.fits', variance='ERR', scidata='SCI',
        vortab='VORTAB')

    # Actual pPXF fitting.
    mycube.ppxf_kinematics(
        fitting_window=(5650, 5850),
        base_wl=wl,
        base_spec=base,
        base_cdelt=0.9,
        write_fits=True,
        deg=3,
        overwrite=True,
    )
