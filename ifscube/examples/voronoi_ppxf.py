#!/usr/bin/env python
# local
from ifscube import Cube 


if __name__ == '__main__':

    # Loads the datacube
    mycube = Cube('ngc3081_cube.fits', variance='ERR', scidata='SCI')

    mycube.snr_eval(wl_range=(5650, 5750))
    mycube.voronoi_binning(targetsnr=3, writefits=True, overwrite=True)
    # mycube.ppxf
