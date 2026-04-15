#!/usr/bin/env python
import glob

import numpy as np
from astropy.io import fits

from ifscube import Cube, spectools, ppxf_wrapper


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
    mycube.voronoi_binning(target_snr=3, write_fits=True, overwrite=True, plot=False)

    #
    # pPXF fitting
    #
    # Loads the Voronoi binned cube.
    mycube = Cube('ngc3081_cube.bin.fits', variance='ERR', scidata='SCI', vortab='VORTAB')

    mask = [
        [4856, 4870],  # Hb
        [4954, 4970],  # [O III]
        [4997, 5022],  # [O III]
        [5190, 5210],  # [N I]
        [5715, 5730],  # ?
        [5860, 5920],  # He I, Na
        [6075, 6100],  # ?
        [6290, 6325],  # [O I]
        [6355, 6393],  # [O I]
        [6430, 6445],  # ?
        [6500, 6640],  # [N II] and Ha
        [6675, 6690],  # He I
        [6710, 6750],  # [S II]
        [6800, 6860],  # Telluric
    ]

    # Actual pPXF fitting.
    ppxf_cube = ppxf_wrapper.cube_kinematics(mycube, fitting_window=(5800, 6800), deg=4, mask=mask)
    ppxf_cube.write('ngc3081_cube.bin.ppxf.fits', overwrite=True)
