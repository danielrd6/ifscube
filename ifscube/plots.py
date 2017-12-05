import numpy as np
import ifscube.spectools as st
import matplotlib.pyplot as plt
from astropy.io import fits


def w80(x):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=14)

    # Create figure and axes.
    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(111)

    # Gets the exponent

    power = np.int(np.log10(np.mean(x[0][3])))
    w80_label = ('M', 'O') 

    for i, j in enumerate(x):

        sc = 'C{:d}'.format(i)

        # Plot the spectrum.
        ax.plot(j[2], j[3] / 10**(power), c=sc)
    
        # Draw w80 lines.
        for k in j[:2]:
            ax.axvline(k, ls='dashed', color=sc)
    
        # Draw baseline
        ax.axhline(0, ls='dashed', color=sc, alpha=.5)
    
        # Labels.
        ax.set_xlabel('Velocity (km/s)')
        ax.set_ylabel(
            '$F_\lambda$ ($10^{{{:d}}}$ erg s$^{{-1}}$'
            ' cm$^{{-2}}$ \AA$^{{-1}}$)'.format(power))
        
        # Write the W80 values
        ax.annotate(
            r'$W_{{80}} (\mathbf{{{:s}}})'
            ' = {:.2f}$ km/s'.format(w80_label[i], np.float(j[1] - j[0])),
            xy=(.65, .9 - i / 10), xycoords='axes fraction', size=12)
    
        ax.minorticks_on()
    
    # Show plot.
    plt.show()
