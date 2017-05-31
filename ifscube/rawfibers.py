from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import pdb


class cube:

    def __init__(self, fitsfile):

        hdu = fits.open(fitsfile)
        self.data = hdu['sci'].data
        self._full_mdf = hdu['mdf'].data

        # Valid apertures, which must equal the number of lines
        # in the data array.
        apid_mask = self._full_mdf['apid'] != 0
        self.mdf = self._full_mdf[apid_mask]

        self.sky_mask = self.mdf['beam'] == 0
        self.obj_mask = self.mdf['beam'] == 1

        self._filename = fitsfile

    def imshow(self, data=None, **kwargs):

        fig = plt.figure(1)
        plt.clf()
        ax = fig.add_subplot(111)

        m = self.obj_mask

        y, x = self.mdf['yinst'][m], self.mdf['xinst'][m]

        if data is None:
            data = np.sum(self.data[m], 1)

        ax.scatter(x, y, c=data, cmap='inferno', s=120,
                   marker='H', edgecolor='none', **kwargs)
        ax.set_aspect('equal')

        plt.show()
