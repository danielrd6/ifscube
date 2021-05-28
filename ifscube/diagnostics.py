import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
from matplotlib import transforms


class bpt:
    """
    See Baldwin, Philips & Terlevich 198?,
        Kauffmann et al. 2003
    """

    def __init__(self, ha, n2, hb, o3):

        for line in ('ha', 'n2', 'hb', 'o3'):
            if not isinstance(eval(line), ma.masked_array):
                self.__dict__.update({line: ma.masked_array(eval(line))})
            else:
                self.__dict__[line] = eval(line)
        
    def kauffmann2003(self):

        ax = self.ax

        x = np.linspace(ax.get_xlim()[0], -.1)
        y = 0.61 / (x - 0.05) + 1.3
        ax.plot(x, y, ls='dashed', color='C2')

        return

    def plot(self, ax=None, fig=None, xlim=(-1.5, .5),
             ylim=(-1.2, 1.5), **kwargs):
        
        if fig is None and ax is None:
            fig = plt.figure(1, figsize=(6, 6))

        if ax is None:
            ax = fig.add_subplot(111)
            
        self.ax = ax

        ax.set_xlabel(r'$\log_{10}$ [N II]/H$\alpha$')
        ax.set_ylabel(r'$\log_{10}$ [O III]/H$\beta$')

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        self.x = ma.log10(self.n2 / self.ha) 
        self.y = ma.log10(self.o3 / self.hb) 

        ax.scatter(self.x, self.y, **kwargs)

        self.kauffmann2003()

        return


class whan_diagram:
    """
    See Cid Fernandes, R. et al 2011 MNRAS 413, 1687.
    """

    def __init__(self, wha, flux_ha, flux_n2):

        for line in ('wha', 'flux_ha', 'flux_n2'):
            if not isinstance(eval(line), ma.masked_array):
                self.__dict__.update({line: ma.masked_array(eval(line))})
            else:
                self.__dict__[line] = eval(line)
 
        self.x = ma.log10(self.flux_n2 / self.flux_ha)
        self.y = ma.log10(self.wha)

    def plot(self, ax=None, fig=None, text_opts={}, xlim=None, ylim=None, **kwargs):

        if fig is None and ax is None:
            fig = plt.figure(1, figsize=(6, 6))

        if ax is None:
            ax = fig.add_subplot(111)
        ax.set_xlabel(r'$\log_{10}$ [N II]/H$\alpha$')
        ax.set_ylabel(r'$\log_{10} {\rm W}_{{\rm H}\alpha}$ ($\AA$)')

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        inv = ax.transAxes.inverted()

        # wha < 3 ==> Retired and passive galaxies
        ax.axhline(np.log10(3), color='k')

        # 3 < wha < 6 ==> weak AGN
        xm = inv.transform(ax.transData.transform((-.4, 0)))[0]
        print(xm)
        ax.axhline(np.log10(6), xmin=xm, color='k')

        # log10([N II] / Ha) < -0.4 ==> Star forming galaxies
        ym = inv.transform(ax.transData.transform((0, np.log10(3))))[1]
        # ax.axvline(-.4, ymin=ym)
        ax.axvline(-.4, color='k')

        ax.text(.05, .95, 'SF', ha='left', transform=ax.transAxes, **text_opts)
        ax.text(.95, .95, 'sAGN', ha='right', transform=ax.transAxes, **text_opts)
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(.95, np.log10(4), 'wAGN', ha='right', transform=trans, **text_opts)
        ax.text(.95, .05, 'Passive galaxies', ha='right', transform=ax.transAxes, **text_opts)

        ax.scatter(self.x, self.y, **kwargs)
