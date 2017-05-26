from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import pdb


class cube:

    def __init__(self, fitsfile):

        hdu = fits.open(fitsfile)
        self.data = hdu['sci'].data
        self.mdf = hdu['mdf'].data

        self._filename = fitsfile

    def imshow(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)

        y, x = self.mdf['yinst'], self.mdf['xinst']

        pdb.set_trace()
        ax.scatter(x, y, c=np.sum(self.data, 1), cmap='inferno')

        plt.show()
