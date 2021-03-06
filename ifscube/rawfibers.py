from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt


class Cube:

    def __init__(self, fitsfile):

        hdu = fits.open(fitsfile)
        self.data = hdu['sci'].data
        self._full_mdf = hdu['mdf'].data

        # Valid apertures, which must equal the number of lines
        # in the data array.
        aperture_id_mask = self._full_mdf['apid'] != 0
        self.mdf = self._full_mdf[aperture_id_mask]

        self.sky_mask = self.mdf['beam'] == 0
        self.obj_mask = self.mdf['beam'] == 1

        self._filename = fitsfile

    def imshow(self, data=None, ax=None, **kwargs):

        if ax is None:
            fig = plt.figure(1)
            plt.clf()
            ax = fig.add_subplot(111)

        m = self.obj_mask

        y, x = self.mdf['yinst'][m], self.mdf['xinst'][m]

        x0 = (x.max() + x.min()) / 2.
        y0 = (y.max() + y.min()) / 2.
        
        x -= x0
        y -= y0

        if data is None:
            data = np.sum(self.data[m], 1)

        ax.scatter(x, y, c=data, cmap='inferno', s=250,
                   marker='H', edgecolor='none', **kwargs)
        ax.set_aspect('equal')

        plt.show()
