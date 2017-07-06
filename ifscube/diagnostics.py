import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms


class whan_diagram:
    """
    See Cid Fernandes, R. et al 2011 MNRAS 413, 1687.
    """

    def __init__(self, x, y):

        self.x = x
        self.y = y

    def plot(self, ax=None, fig=None, **kwargs):

        if fig is None and ax is None:
            fig = plt.figure()

        if ax is None:
            ax = fig.add_subplot(111)
        ax.set_xlabel(r'$\log_{10}$ [N II]/H$\alpha$')
        ax.set_ylabel(r'$\log_{10} {\rm W}_{{\rm H}\alpha}$ ($\AA$)')
 
        ax.set_xlim(-1, .6)
        ax.set_ylim(-1, 2)
        inv = ax.transAxes.inverted()

        # wha < 3 ==> Retired and passive galaxies
        ax.axhline(np.log10(3))

        # 3 < wha < 6 ==> weak AGN
        xm = inv.transform(ax.transData.transform((-.4, 0)))[0]
        print(xm)
        ax.axhline(np.log10(6), xmin=xm)

        # log10([N II] / Ha) < -0.4 ==> Star forming galaxies
        ym = inv.transform(ax.transData.transform((0, np.log10(3))))[1]
        ax.axvline(-.4, ymin=ym)

        ax.text(.05, .95, 'SF', ha='left', transform=ax.transAxes)
        ax.text(.95, .95, 'sAGN', ha='right', transform=ax.transAxes)
        trans = transforms.blended_transform_factory(
                ax.transAxes, ax.transData)
        ax.text(.95, np.log10(4), 'wAGN', ha='right', transform=trans)
        ax.text(.95, .1, 'Passive galaxies', ha='right', transform=ax.transAxes)

        ax.scatter(self.x, self.y, **kwargs)
